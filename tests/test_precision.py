from __future__ import annotations

import unittest

import torch

from neurosim_gpu import (
    DendriticDynamics, DendriticMatrixRuntime, DendriticPELSRuntime, Dynamics,
    SparseTopology, capture_runtime, restore_runtime,
)
from neurosim_gpu.precision import quantize_dequantize
from neurosim_gpu.scheduler import MultiRateScheduler


class PrecisionQuantizationTests(unittest.TestCase):
    def test_quantize_dequantize_modes_are_finite_and_ste_differentiable(self):
        source = torch.linspace(-2.0, 2.0, 77).reshape(7, 11)
        for precision in ("fp16", "fp8", "nvfp4"):
            value = source.clone().requires_grad_()
            decoded = quantize_dequantize(value, precision)
            self.assertEqual(decoded.shape, value.shape)
            self.assertTrue(torch.isfinite(decoded).all())
            decoded.sum().backward()
            torch.testing.assert_close(value.grad, torch.ones_like(value), rtol=0, atol=0)

    def test_zero_nvfp4_input_stays_zero(self):
        value = torch.zeros((2, 19))
        decoded = quantize_dequantize(value, "nvfp4")
        torch.testing.assert_close(decoded, value, rtol=0, atol=0)

    def test_unknown_precision_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "precision must be"):
            quantize_dequantize(torch.ones(2), "fp6")  # type: ignore[arg-type]


def _case(device: str = "cpu", *, trainable: bool = False):
    n = 48
    neurons = torch.arange(n, device=device)
    src = neurons.repeat_interleave(4)
    dst = ((neurons[:, None] + torch.tensor([1, 7, 13, 19], device=device)) % n).flatten()
    edge_weight = torch.full((src.numel(),), 0.012, device=device, requires_grad=trainable)
    graph = SparseTopology.from_edges(n, src, dst, edge_weight)
    alpha = torch.full((n,), 0.72, device=device, requires_grad=trainable)
    gain = torch.full((n,), 0.8, device=device, requires_grad=trainable)
    dynamics = DendriticDynamics.from_time_constants(
        Dynamics(alpha, gain), dt=1, tau_dendrite=4, tau_adaptation=20,
        coupling=0.45, adaptation_strength=0.15,
    )
    state = torch.full((2, n), 0.25, device=device, requires_grad=trainable)
    return graph, dynamics, state, (edge_weight, alpha, gain)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for CSR low-precision runtime tests")
class PrecisionRuntimeTests(unittest.TestCase):
    @torch.no_grad()
    def test_topology_replacement_rebuilds_low_precision_pels_current(self):
        graph, dynamics, state, _ = _case("cuda")
        replacement = SparseTopology.from_edges(
            graph.n_neurons, graph.src, (graph.dst + 1) % graph.n_neurons,
            torch.linspace(-0.3, 0.4, graph.n_edges, device="cuda"),
        )
        self.assertEqual(graph.weight_version, replacement.weight_version)
        options = dict(precision="nvfp4", propagation="matrix", theta=0.01)
        runtime = DendriticPELSRuntime(graph, dynamics, state.clone(), **options)
        oracle = DendriticPELSRuntime(replacement, dynamics, state.clone(), **options)
        MultiRateScheduler(runtime).replace_topology(replacement)
        torch.testing.assert_close(runtime.persistent_current, oracle.persistent_current, atol=0, rtol=0)
        torch.testing.assert_close(runtime.step(0.2), oracle.step(0.2), atol=0, rtol=0)

    def test_inference_formats_and_three_state_updates(self):
        for precision in ("fp32", "fp16", "fp8", "nvfp4"):
            graph, dynamics, initial, _ = _case("cuda")
            matrix = DendriticMatrixRuntime(graph, dynamics, initial.clone(), precision=precision)
            pels = DendriticPELSRuntime(
                graph, dynamics, initial.clone(), theta=0.0, propagation="matrix",
                precision=precision,
            )
            with torch.inference_mode():
                for _ in range(12):
                    matrix.step(0.2, dendritic_drive=0.05)
                    pels.step(0.2, dendritic_drive=0.05)
            for runtime in (matrix, pels):
                for name in ("state", "dendritic_state", "adaptation_state"):
                    value = getattr(runtime, name)
                    self.assertTrue(torch.isfinite(value).all())
                self.assertTrue(torch.all((runtime.state >= 0) & (runtime.state <= 1)))
                if precision != "fp32":
                    self.assertIn(precision, runtime.precision_execution)
            if precision != "fp32":
                for name in ("state", "dendritic_state", "adaptation_state"):
                    torch.testing.assert_close(getattr(matrix, name), getattr(pels, name), atol=0, rtol=0)
                _, stats = pels.step(0.2, collect_stats=True)
                self.assertEqual(stats.path, "matrix-precision")

    def test_training_backward_reaches_state_parameters_and_sparse_weights(self):
        for precision in ("fp16", "fp8", "nvfp4"):
            graph, dynamics, state, leaves = _case("cuda", trainable=True)
            runtime = DendriticMatrixRuntime(graph, dynamics, state, precision=precision)
            for _ in range(5):
                runtime.step(0.2, dendritic_drive=0.05)
            runtime.state.square().mean().backward()
            for leaf in (state, *leaves):
                self.assertIsNotNone(leaf.grad, precision)
                self.assertTrue(torch.isfinite(leaf.grad).all(), precision)
                self.assertGreater(float(leaf.grad.abs().sum()), 0.0, precision)

    def test_precision_checkpoint_cannot_restore_into_different_mode(self):
        graph, dynamics, state, _ = _case("cuda")
        source = DendriticMatrixRuntime(graph, dynamics, state.clone(), precision="fp8")
        payload = capture_runtime(source)
        target = DendriticMatrixRuntime(graph, dynamics, state.clone(), precision="fp32")
        with self.assertRaisesRegex(ValueError, "precision does not match"):
            restore_runtime(target, payload)

    @torch.no_grad()
    def test_checkpoint_resume_after_weight_change_with_warm_caches(self):
        for precision in ("fp16", "fp8", "nvfp4"):
            for kind, options in (
                (DendriticMatrixRuntime, {}),
                (DendriticPELSRuntime, {"theta": 0.01, "propagation": "matrix"}),
            ):
                with self.subTest(precision=precision, kind=kind.__name__):
                    graph, dynamics, initial, _ = _case("cuda")
                    target_graph, _, _, _ = _case("cuda")
                    source = kind(graph, dynamics, initial.clone(), precision=precision, **options)
                    target = kind(target_graph, dynamics, initial.clone(), precision=precision, **options)
                    source.step(0.2)
                    target.step(0.1)
                    graph.apply_source_weights_(graph.weight * 4)
                    restore_runtime(target, capture_runtime(source))
                    for _ in range(3):
                        source.step(0.3)
                        target.step(0.3)
                        for field in ("state", "dendritic_state", "adaptation_state"):
                            torch.testing.assert_close(
                                getattr(target, field), getattr(source, field), atol=0, rtol=0,
                            )


if __name__ == "__main__":
    unittest.main()
