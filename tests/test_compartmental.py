from __future__ import annotations

from dataclasses import replace
import io
import math
import unittest

import torch

from neurosim_gpu import (
    DendriticDynamics, DendriticMatrixRuntime, DendriticPELSRuntime,
    Dynamics, MatrixRuntime, SparseTopology, capture_runtime,
    load_runtime_checkpoint, random_topology, restore_runtime,
    save_runtime_checkpoint,
)


class CompartmentTests(unittest.TestCase):
    def make(self, device="cpu"):
        graph = random_topology(12, 48, device=device, dtype=torch.float64, seed=17)
        soma = Dynamics(torch.full((12,), 0.7, device=device, dtype=torch.float64),
                        torch.full((12,), 0.8, device=device, dtype=torch.float64))
        dynamics = DendriticDynamics.from_time_constants(
            soma, dt=1.0, tau_dendrite=torch.linspace(2, 5, 12),
            tau_adaptation=20.0, coupling=0.6, adaptation_strength=0.3,
        )
        return graph, dynamics, torch.full((2, 12), 0.2, device=device,
                                           dtype=torch.float64)

    def test_neutral_parameters_recover_original_trajectory(self):
        graph, dynamics, initial = self.make()
        dynamics = replace(dynamics, coupling=torch.zeros(12, dtype=torch.float64),
                           adaptation_strength=torch.zeros(12, dtype=torch.float64))
        baseline = MatrixRuntime(graph, dynamics.soma, initial.clone())
        extended = DendriticMatrixRuntime(graph, dynamics, initial.clone())
        for t in range(30):
            drive = 0.15 + 0.05 * math.sin(t)
            torch.testing.assert_close(extended.step(drive), baseline.step(drive),
                                       rtol=0, atol=0)

    def check_paths(self, device):
        graph, dynamics, initial = self.make(device)
        matrix = DendriticMatrixRuntime(graph, dynamics, initial.clone())
        paths = [DendriticPELSRuntime(graph, dynamics, initial.clone(), theta=0,
                                     propagation=mode)
                 for mode in ("matrix", "event", "auto")]
        # Independent dense recurrence: catches state order and input routing.
        state, dendrite, adaptation = initial.clone(), torch.zeros_like(initial), torch.zeros_like(initial)
        weight = graph.dense_matrix()
        for t in range(20):
            current = state @ weight.T
            dendrite = (dynamics.dendrite_decay * dendrite
                        + (1 - dynamics.dendrite_decay) * torch.tanh(current + 0.1))
            combined = (0.4 * current + 0.6 * dendrite - 0.3 * adaptation)
            state = (0.7 * state + 0.3 * 0.8 * torch.tanh(combined + 0.2)).clamp(0, 1)
            adaptation = (dynamics.adaptation_decay * adaptation
                          + (1 - dynamics.adaptation_decay) * state)
            torch.testing.assert_close(matrix.step(0.2, dendritic_drive=0.1), state,
                                       rtol=1e-11, atol=1e-12)
            for path in paths:
                torch.testing.assert_close(path.step(0.2, dendritic_drive=0.1), state,
                                           rtol=1e-11, atol=1e-12)
                torch.testing.assert_close(path.dendritic_state, dendrite)
                torch.testing.assert_close(path.adaptation_state, adaptation)

    def test_dense_matrix_and_pels_paths_cpu(self):
        self.check_paths("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_dense_matrix_and_pels_paths_cuda(self):
        self.check_paths("cuda")

    def test_dendritic_impulse_has_heterogeneous_decay(self):
        _, dynamics, initial = self.make()
        dynamics = replace(dynamics, coupling=torch.ones_like(dynamics.coupling),
                           adaptation_strength=torch.zeros_like(dynamics.coupling))
        zero = torch.zeros_like(initial)
        state, dendrite, adaptation = dynamics.advance(zero, zero, zero, zero,
                                                       dendritic_drive=1.0)
        expected = (1 - dynamics.dendrite_decay) * math.tanh(1.0)
        torch.testing.assert_close(dendrite[0], expected)
        for t in range(1, 12):
            state, dendrite, adaptation = dynamics.advance(state, dendrite, adaptation, zero)
            torch.testing.assert_close(dendrite[0], expected * dynamics.dendrite_decay**t)
        self.assertGreater(float(state.sum()), 0)

    def test_adaptation_reduces_sustained_response_and_recovers(self):
        _, dynamics, initial = self.make()
        dynamics = replace(dynamics, adaptation_strength=torch.ones_like(dynamics.coupling))
        zero = torch.zeros_like(initial)
        state, dendrite, adaptation = zero, zero, zero
        trace = []
        for _ in range(200):
            state, dendrite, adaptation = dynamics.advance(state, dendrite, adaptation, zero, 0.6)
            trace.append(float(state.mean()))
        self.assertGreater(max(trace), trace[-1] + 0.05)
        peak_adaptation = adaptation.clone()
        for _ in range(200):
            state, dendrite, adaptation = dynamics.advance(state, dendrite, adaptation, zero)
        self.assertLess(float(adaptation.max()), float(peak_adaptation.min()) * 0.001)

    def test_signed_inhibition_and_batch_independence(self):
        graph, dynamics, initial = self.make()
        batched = DendriticMatrixRuntime(graph, dynamics, initial.clone())
        single = [DendriticMatrixRuntime(graph, dynamics, initial[i:i+1].clone()) for i in range(2)]
        drive = torch.tensor([[0.5], [0.1]], dtype=torch.float64)
        dendritic_drive = torch.tensor([[-0.8], [0.8]], dtype=torch.float64)
        for _ in range(20):
            actual = batched.step(drive, dendritic_drive=dendritic_drive)
            expected = torch.cat([r.step(drive[i], dendritic_drive=dendritic_drive[i])
                                  for i, r in enumerate(single)])
            torch.testing.assert_close(actual, expected)
        self.assertTrue(bool((batched.dendritic_state[0] < 0).all()))

    def test_idle_pels_still_updates_local_states(self):
        graph, dynamics, initial = self.make()
        runtime = DendriticPELSRuntime(graph, dynamics, initial, theta=2,
                                       propagation="event")
        state, stats = runtime.step(0.5, dendritic_drive=0.5, collect_stats=True)
        self.assertEqual(stats.path, "idle")
        self.assertEqual(stats.active_edges, 0)
        self.assertGreater(float(runtime.adaptation_state.sum()), 0)
        self.assertGreater(float(runtime.dendritic_state.abs().sum()), 0)
        self.assertFalse(torch.equal(state, initial))

    def test_nonzero_threshold_residual_and_external_weight_refresh(self):
        graph, dynamics, initial = self.make()
        runtime = DendriticPELSRuntime(graph, dynamics, initial, theta=0.02,
                                       propagation="event")
        for t in range(20):
            if t == 5:
                graph.apply_source_weights_(graph.weight * 0.7)
            runtime.step(0.2)
            self.assertLess(float((runtime.state - runtime.emitted_state).abs().max()), 0.02 + 1e-12)
            torch.testing.assert_close(runtime.persistent_current,
                                       runtime.emitted_state @ graph.dense_matrix().T,
                                       rtol=1e-11, atol=1e-12)

    def test_gradients_match_dense_and_finite_difference(self):
        graph, dynamics, initial = self.make()
        weights = graph.weight.detach().clone().requires_grad_()
        graph = SparseTopology.from_edges(12, graph.src, graph.dst, weights)
        drive = torch.full_like(initial, 0.3, requires_grad=True)
        matrix = DendriticMatrixRuntime(graph, dynamics, initial.clone())
        pels = DendriticPELSRuntime(graph, dynamics, initial.clone(), theta=0,
                                   propagation="event")
        state, dendrite, adaptation = initial.clone(), torch.zeros_like(initial), torch.zeros_like(initial)
        for _ in range(8):
            matrix.step(drive, dendritic_drive=0.2)
            pels.step(drive, dendritic_drive=0.2)
            state, dendrite, adaptation = dynamics.advance(
                state, dendrite, adaptation, state @ graph.dense_matrix().T,
                drive, dendritic_drive=0.2,
            )
        expected = torch.autograd.grad(state.square().sum(), (weights, drive), retain_graph=True)
        for output in (matrix.state, pels.state):
            actual = torch.autograd.grad(output.square().sum(), (weights, drive), retain_graph=True)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, rtol=1e-9, atol=1e-11)
                self.assertGreater(float(a.abs().sum()), 0)
        # Independent numerical derivative of a multi-tick observable.
        def loss(delta):
            runtime = DendriticMatrixRuntime(graph, dynamics, initial.clone())
            changed_drive = drive.detach().clone()
            changed_drive[0, 0] += delta
            for _ in range(8):
                runtime.step(changed_drive, dendritic_drive=0.2)
            return float(runtime.state.square().sum().detach())
        numerical = (loss(1e-5) - loss(-1e-5)) / 2e-5
        self.assertAlmostEqual(numerical, float(expected[1][0, 0]), places=8)

    def test_checkpoint_restores_all_states(self):
        graph, dynamics, initial = self.make()
        for runtime_type, kwargs in ((DendriticMatrixRuntime, {}),
                                     (DendriticPELSRuntime, {"theta": 0.01, "propagation": "event"})):
            with self.subTest(runtime=runtime_type):
                runtime = runtime_type(graph, dynamics, initial.clone(), **kwargs)
                for _ in range(10):
                    runtime.step(0.2, dendritic_drive=0.4)
                buffer = io.BytesIO()
                save_runtime_checkpoint(runtime, buffer)
                restored = runtime_type(graph, dynamics, initial.clone(), **kwargs)
                buffer.seek(0)
                load_runtime_checkpoint(buffer, restored)
                for _ in range(10):
                    torch.testing.assert_close(runtime.step(0.3), restored.step(0.3), rtol=0, atol=0)
                    torch.testing.assert_close(runtime.dendritic_state, restored.dendritic_state)
                    torch.testing.assert_close(runtime.adaptation_state, restored.adaptation_state)

    def test_incomplete_or_incompatible_checkpoint_rejected(self):
        graph, dynamics, initial = self.make()
        runtime = DendriticMatrixRuntime(graph, dynamics, initial.clone())
        runtime.step(0.3)
        payload = capture_runtime(runtime)
        with self.assertRaisesRegex(ValueError, "runtime type"):
            restore_runtime(DendriticPELSRuntime(graph, dynamics, initial.clone(), theta=0), payload)
        for field in ("dendrite_decay", "adaptation_decay", "coupling", "adaptation_strength"):
            other = replace(dynamics, **{field: getattr(dynamics, field) * 0.9})
            with self.assertRaisesRegex(ValueError, "parameters"):
                restore_runtime(DendriticMatrixRuntime(graph, other, initial.clone()), payload)
        del payload["tensors"]["adaptation_state"]
        before = runtime.state.clone()
        with self.assertRaisesRegex(ValueError, "missing tensor adaptation_state"):
            restore_runtime(runtime, payload)
        torch.testing.assert_close(runtime.state, before)

    def test_long_drives_keep_states_finite_and_bounded(self):
        _, dynamics, initial = self.make()
        state, dendrite, adaptation = initial, torch.zeros_like(initial), torch.zeros_like(initial)
        for t in range(1500):
            state, dendrite, adaptation = dynamics.advance(
                state, dendrite, adaptation, torch.full_like(initial, (-1)**(t//40) * 1e4),
                0.3, dendritic_drive=0.2,
            )
        for value, low in ((state, 0), (adaptation, 0), (dendrite, -1)):
            self.assertTrue(bool((torch.isfinite(value) & (value >= low) & (value <= 1)).all()))

    def test_invalid_parameters_and_state_rejected(self):
        graph, dynamics, initial = self.make()
        for field, value in (("dendrite_decay", 1.0), ("adaptation_decay", -0.1),
                             ("coupling", 1.1), ("adaptation_strength", -0.1),
                             ("coupling", float("nan")), ("adaptation_strength", float("inf"))):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                replace(dynamics, **{field: torch.full((12,), value, dtype=torch.float64)})
        with self.assertRaisesRegex(ValueError, "dtype"):
            replace(dynamics, coupling=dynamics.coupling.float())
        with self.assertRaisesRegex(ValueError, "one value per neuron"):
            replace(dynamics, coupling=torch.ones(2, dtype=torch.float64))
        for tau in (0, -1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                DendriticDynamics.from_time_constants(dynamics.soma, dt=1,
                                                     tau_dendrite=tau, tau_adaptation=10)
        with self.assertRaisesRegex(ValueError, "shape"):
            DendriticMatrixRuntime(graph, dynamics, initial[0])
        with self.assertRaisesRegex(ValueError, "finite"):
            DendriticMatrixRuntime(graph, dynamics, initial * float("nan"))
        runtime = DendriticMatrixRuntime(graph, dynamics, initial)
        with self.assertRaisesRegex(ValueError, "broadcast"):
            runtime.step(dendritic_drive=torch.ones(3, 2, 12))
        torch.testing.assert_close(runtime.dendritic_state, torch.zeros_like(initial))


if __name__ == "__main__":
    unittest.main()
