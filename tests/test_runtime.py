from __future__ import annotations

import unittest

import torch

from neurosim_gpu import (
    DenseOracle,
    Dynamics,
    HybridPELSRuntime,
    MatrixRuntime,
    SparseTopology,
    random_topology,
    uniform_error_budget,
)


class RuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        self.n = 32
        self.batch = 3
        base_topology = random_topology(
            self.n,
            180,
            device=self.device,
            dtype=self.dtype,
            seed=7,
        )
        self.edge_parameter = (
            base_topology.weight.detach().clone().requires_grad_(True)
        )
        self.topology = SparseTopology.from_edges(
            self.n,
            base_topology.src,
            base_topology.dst,
            self.edge_parameter,
        )
        generator = torch.Generator(device=self.device).manual_seed(11)
        self.initial = torch.rand(
            self.batch,
            self.n,
            device=self.device,
            dtype=self.dtype,
            generator=generator,
        )
        self.dynamics = Dynamics(
            alpha=torch.full(
                (self.n,), 0.93, device=self.device, dtype=self.dtype
            ),
            gain=torch.full(
                (self.n,), 0.8, device=self.device, dtype=self.dtype
            ),
        )

    def test_sparse_matrix_matches_dense_oracle(self) -> None:
        dense = DenseOracle(self.topology, self.dynamics, self.initial.clone())
        sparse = MatrixRuntime(self.topology, self.dynamics, self.initial.clone())
        for _ in range(20):
            expected = dense.step(0.15)
            actual = sparse.step(0.15)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)

    def test_theta_zero_matches_matrix_runtime(self) -> None:
        matrix = MatrixRuntime(self.topology, self.dynamics, self.initial.clone())
        pels = HybridPELSRuntime(
            self.topology,
            self.dynamics,
            self.initial.clone(),
            theta=0.0,
            propagation="matrix",
        )
        for _ in range(20):
            expected = matrix.step(0.15)
            actual = pels.step(0.15)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)

    def test_event_and_matrix_delta_paths_match(self) -> None:
        matrix = HybridPELSRuntime(
            self.topology,
            self.dynamics,
            self.initial.clone(),
            theta=0.01,
            propagation="matrix",
        )
        event = HybridPELSRuntime(
            self.topology,
            self.dynamics,
            self.initial.clone(),
            theta=0.01,
            propagation="event",
        )
        for _ in range(20):
            expected = matrix.step(0.15)
            actual = event.step(0.15)
        torch.testing.assert_close(actual, expected, rtol=1e-11, atol=1e-12)
        torch.testing.assert_close(
            event.persistent_current,
            matrix.persistent_current,
            rtol=1e-11,
            atol=1e-12,
        )

    def test_unemitted_residual_is_bounded(self) -> None:
        theta = 0.02
        pels = HybridPELSRuntime(
            self.topology,
            self.dynamics,
            self.initial.clone(),
            theta=theta,
            propagation="matrix",
        )
        for _ in range(20):
            pels.step(0.15)
            residual = (pels.state - pels.emitted_state).abs()
            self.assertLess(float(residual.detach().max()), theta + 1e-12)

    def test_theta_zero_gradient_matches_dense(self) -> None:
        dense = DenseOracle(self.topology, self.dynamics, self.initial.clone())
        pels = HybridPELSRuntime(
            self.topology,
            self.dynamics,
            self.initial.clone(),
            theta=0.0,
            propagation="matrix",
        )
        for _ in range(8):
            dense.step(0.15)
            pels.step(0.15)
        dense_loss = dense.state.square().mean()
        pels_loss = pels.state.square().mean()
        dense_gradient = torch.autograd.grad(
            dense_loss, self.edge_parameter, retain_graph=True
        )[0]
        pels_gradient = torch.autograd.grad(
            pels_loss, self.edge_parameter
        )[0]
        torch.testing.assert_close(
            pels_gradient,
            dense_gradient,
            rtol=1e-9,
            atol=1e-11,
        )

    def test_uniform_error_budget_bounds_omitted_current(self) -> None:
        budget = uniform_error_budget(
            self.topology,
            self.dynamics,
            target_state_error=1e-3,
        )
        src = self.topology.src.to(torch.int64)
        dst = self.topology.dst.to(torch.int64)
        omitted_current_bound = torch.zeros(
            self.n,
            dtype=self.dtype,
            device=self.device,
        ).index_add_(
            0,
            dst,
            self.topology.weight.detach().abs() * budget.theta[src],
        )
        torch.testing.assert_close(
            torch.minimum(omitted_current_bound, budget.current_tolerance),
            omitted_current_bound,
            rtol=1e-12,
            atol=1e-14,
        )

    def test_neutral_modulation_matches_standard_advance(self) -> None:
        current = self.initial * 0.1
        expected = self.dynamics.advance(self.initial, current, 0.15)
        actual = self.dynamics.advance_modulated(
            self.initial,
            current,
            0.15,
            drive_shift=torch.zeros_like(self.initial),
            gain_scale=torch.ones_like(self.initial),
        )
        torch.testing.assert_close(actual, expected)

    def test_source_weight_update_keeps_matrix_view_consistent(self) -> None:
        updated = self.topology.weight.detach() * 0.75
        self.topology.apply_source_weights_(updated)
        reconstructed = self.topology.in_weight[
            self.topology.out_weight_index.to(torch.int64)
        ]
        torch.testing.assert_close(reconstructed, updated)

    def test_nonfinite_persistent_threshold_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite"):
            HybridPELSRuntime(
                self.topology,
                self.dynamics,
                self.initial.clone(),
                theta=float("nan"),
            )
        with self.assertRaisesRegex(ValueError, "zero-outdegree"):
            HybridPELSRuntime(
                self.topology,
                self.dynamics,
                self.initial.clone(),
                theta=float("inf"),
            )

    def test_error_budget_infinity_is_allowed_for_inert_source(self) -> None:
        source = torch.tensor([0], device=self.device, dtype=torch.int64)
        destination = torch.tensor([1], device=self.device, dtype=torch.int64)
        weight = torch.tensor([0.05], device=self.device, dtype=self.dtype)
        topology = SparseTopology.from_edges(3, source, destination, weight)
        dynamics = Dynamics(
            alpha=torch.full((3,), 0.8, device=self.device, dtype=self.dtype),
            gain=torch.full((3,), 0.5, device=self.device, dtype=self.dtype),
        )
        budget = uniform_error_budget(topology, dynamics, 1e-3)
        self.assertTrue(bool(torch.isposinf(budget.theta[1])))
        self.assertTrue(bool(torch.isposinf(budget.theta[2])))
        runtime = HybridPELSRuntime(
            topology,
            dynamics,
            torch.full((1, 3), 0.2, device=self.device, dtype=self.dtype),
            theta=budget.theta,
            propagation="matrix",
        )
        runtime.step(0.1)

    def test_external_weight_update_invalidates_persistent_cache(self) -> None:
        runtime = HybridPELSRuntime(
            self.topology,
            self.dynamics,
            self.initial.clone(),
            theta=0.02,
            propagation="matrix",
        )
        self.topology.apply_source_weights_(self.topology.weight * 0.5)
        runtime.step(0.15)
        recomputed = runtime._propagate(runtime.emitted_state).contiguous()
        torch.testing.assert_close(
            runtime.persistent_current,
            recomputed,
            rtol=1e-11,
            atol=1e-12,
        )


if __name__ == "__main__":
    unittest.main()
