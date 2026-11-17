from __future__ import annotations

import io
import os
import unittest
from dataclasses import replace

import torch

from neurosim_gpu import (
    Dynamics, DendriticDynamics, DendriticMatrixRuntime, DendriticPELSRuntime,
    NativeDendriticMatrixRuntime, NativeDendriticPELSRuntime, SparseTopology,
    random_topology, save_runtime_checkpoint, load_runtime_checkpoint,
)


@unittest.skipUnless(torch.cuda.is_available() and os.environ.get("NEUROSIM_TEST_NATIVE") == "1",
                     "set NEUROSIM_TEST_NATIVE=1 with a CUDA build environment")
class NativeCompartmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from neurosim_gpu.native import load_native_extension
        cls.extension = load_native_extension()

    def make(self, n=97, batch=3, edges=600):
        graph = random_topology(n, edges, device="cuda", seed=83)
        soma = Dynamics(torch.linspace(0.4, 0.95, n, device="cuda"),
                        torch.full((n,), 0.8, device="cuda"))
        model = DendriticDynamics.from_time_constants(soma, dt=1,
                    tau_dendrite=torch.linspace(2, 8, n), tau_adaptation=25,
                    coupling=0.6, adaptation_strength=0.3)
        state = torch.full((batch, n), 0.2, device="cuda")
        return graph, model, state

    def assert_states(self, actual, expected, atol=2e-6):
        for name in ("state", "dendritic_state", "adaptation_state"):
            torch.testing.assert_close(getattr(actual, name), getattr(expected, name),
                                       atol=atol, rtol=2e-4)

    def test_matrix_widths_and_strided_broadcast_inputs(self):
        for width in (8, 16, 32):
            graph, model, state = self.make()
            # Transposed state, strided params, Bx1 / N input; no hidden materialization.
            state = state.T.contiguous().T
            model = replace(model, coupling=torch.ones(194, device="cuda")[::2] * 0.6)
            native = NativeDendriticMatrixRuntime(graph, model, state.clone(),
                        threads_per_row=width, extension=self.extension)
            reference = DendriticMatrixRuntime(graph, model, state.clone())
            drives = [0.2, torch.tensor(0.2, device="cuda"),
                      torch.tensor([[0.1], [0.3], [0.5]], device="cuda"),
                      torch.linspace(0.1, 0.3, 194, device="cuda")[::2]]
            for value in drives:
                for _ in range(8):
                    native.step(value, dendritic_drive=-0.1)
                    reference.step(value, dendritic_drive=-0.1)
                self.assert_states(native, reference)
            self.assertEqual(native.last_backend, "native")

    def test_empty_edges_empty_batch_and_large_degree(self):
        for n, batch, edges in ((1, 2, 0), (9, 0, 20), (65, 3, 4000)):
            graph, model, state = self.make(n, batch, edges)
            for kind, kwargs in ((NativeDendriticMatrixRuntime, {}),
                                 (NativeDendriticPELSRuntime, {"theta": 0.0})):
                native = kind(graph, model, state.clone(), extension=self.extension, **kwargs)
                reference = DendriticMatrixRuntime(graph, model, state.clone())
                for _ in range(8):
                    native.step(0.2)
                    reference.step(0.2)
                self.assert_states(native, reference)

    def test_theta_zero_long_rollout_matches_matrix(self):
        graph, model, state = self.make(1024, 2, 16000)
        native = NativeDendriticPELSRuntime(graph, model, state.clone(), theta=0,
                                          extension=self.extension)
        reference = DendriticMatrixRuntime(graph, model, state.clone())
        for t in range(500):
            value = 0.2 if t % 70 < 35 else 0.05
            native.step(value, dendritic_drive=0.1)
            reference.step(value, dendritic_drive=0.1)
        self.assert_states(native, reference, atol=8e-6)
        torch.testing.assert_close(native.persistent_current,
            native._propagate(native.emitted_state), rtol=2e-4, atol=1e-5)

    def test_nonzero_threshold_emission_and_weight_mutation(self):
        graph, model, state = self.make()
        native = NativeDendriticPELSRuntime(graph, model, state.clone(), theta=0.02,
                                          extension=self.extension)
        reference = DendriticPELSRuntime(graph, model, state.clone(), theta=0.02,
                                         propagation="matrix")
        for t in range(50):
            if t == 25:
                graph.apply_source_weights_(graph.weight * 0.7)
            native.step(0.3, dendritic_drive=0.1)
            reference.step(0.3, dendritic_drive=0.1)
            self.assertLess(float((native.state-native.emitted_state).abs().max()), 0.020001)
            torch.testing.assert_close(native.persistent_current,
                native._propagate(native.emitted_state), rtol=2e-4, atol=2e-6)
        self.assert_states(native, reference)
        # Explicit statistics fallback must leave the next native tick consistent.
        _, stats = native.step(0.3, collect_stats=True)
        reference.step(0.3)
        self.assertEqual(native.last_backend, "reference-stats")
        self.assertGreaterEqual(stats.active_edges, 0)
        native.step(0.3)
        reference.step(0.3)
        self.assert_states(native, reference)

    def test_idle_updates_traces_and_retains_returned_state(self):
        graph, model, state = self.make()
        native = NativeDendriticPELSRuntime(graph, model, state.clone(), theta=2,
                                          extension=self.extension)
        previous = native.step(0.5)
        saved = previous.clone()
        current = native.persistent_current.clone()
        first_adaptation = native.adaptation_state.clone()
        for _ in range(10):
            native.step(0.5)
        torch.testing.assert_close(previous, saved, rtol=0, atol=0)
        torch.testing.assert_close(native.persistent_current, current, rtol=0, atol=0)
        self.assertFalse(torch.equal(first_adaptation, native.adaptation_state))

    def test_all_parameter_and_drive_gradients_use_reference(self):
        graph, model, state = self.make(17, 2, 80)
        model = replace(model, soma=Dynamics(model.soma.alpha.clone().requires_grad_(),
                                            model.soma.gain.clone().requires_grad_()),
                        dendrite_decay=model.dendrite_decay.clone().requires_grad_(),
                        adaptation_decay=model.adaptation_decay.clone().requires_grad_(),
                        coupling=model.coupling.clone().requires_grad_(),
                        adaptation_strength=model.adaptation_strength.clone().requires_grad_())
        for kind, kwargs in ((NativeDendriticMatrixRuntime, {}),
                             (NativeDendriticPELSRuntime, {"theta": 0.0})):
            starting = state.clone().requires_grad_()
            drive = torch.full_like(state, 0.3, requires_grad=True)
            dd = torch.full_like(state, 0.2, requires_grad=True)
            native = kind(graph, model, starting, extension=self.extension, **kwargs)
            reference = DendriticMatrixRuntime(graph, model, starting)
            for _ in range(10):
                native.step(drive, dendritic_drive=dd)
                reference.step(drive, dendritic_drive=dd)
            values = [*native._parameters(), drive, dd, starting]
            actual = torch.autograd.grad(native.state.sum(), values, retain_graph=True)
            expected = torch.autograd.grad(reference.state.sum(), values)
            for a, e in zip(actual, expected):
                torch.testing.assert_close(a, e, atol=2e-5, rtol=2e-4)
                self.assertGreater(float(a.abs().sum()), 0)
            self.assertEqual(native.last_backend, "reference-autograd")

    def test_synaptic_gradient_is_not_silently_dropped(self):
        graph, model, state = self.make(17, 2, 80)
        weights = graph.weight.detach().clone().requires_grad_()
        graph = SparseTopology.from_edges(17, graph.src, graph.dst, weights)
        native = NativeDendriticMatrixRuntime(graph, model, state,
                                              extension=self.extension)
        for _ in range(4):
            native.step(0.3)
        gradient = torch.autograd.grad(native.state.sum(), weights)[0]
        self.assertGreater(float(gradient.abs().sum()), 0)
        self.assertEqual(native.last_backend, "reference-autograd")

    def test_parameters_remain_live_and_no_grad_can_accelerate(self):
        graph, model, state = self.make()
        model = replace(model, coupling=model.coupling.clone().requires_grad_())
        native = NativeDendriticMatrixRuntime(graph, model, state.clone(), extension=self.extension)
        reference = DendriticMatrixRuntime(graph, model, state.clone())
        with torch.no_grad():
            for _ in range(6):
                model.coupling.mul_(0.9)
                native.step(0.2)
                reference.step(0.2)
        self.assert_states(native, reference)
        self.assertEqual(native.last_backend, "native")

    def test_checkpoint_resume_and_nondefault_stream(self):
        graph, model, state = self.make()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for kind, kwargs in ((NativeDendriticMatrixRuntime, {}),
                                 (NativeDendriticPELSRuntime, {"theta": 0.01})):
                native = kind(graph, model, state.clone(), extension=self.extension, **kwargs)
                for _ in range(8):
                    native.step(0.2)
                buffer = io.BytesIO()
                save_runtime_checkpoint(native, buffer)
                restored = kind(graph, model, state.clone(), extension=self.extension, **kwargs)
                buffer.seek(0)
                load_runtime_checkpoint(buffer, restored)
                for _ in range(8):
                    native.step(0.3)
                    restored.step(0.3)
                self.assert_states(native, restored)
                if hasattr(native, "_tick"):
                    self.assertEqual(native._tick, restored._tick)
        torch.cuda.current_stream().wait_stream(stream)

    def test_invalid_inputs_fail_before_state_mutation(self):
        graph, model, state = self.make()
        native = NativeDendriticMatrixRuntime(graph, model, state.clone(), extension=self.extension)
        for value in (torch.zeros(3, 3, 97, device="cuda"), torch.zeros(97),
                      torch.zeros(97, device="cuda", dtype=torch.float64)):
            with self.assertRaises(ValueError):
                native.step(value)
            torch.testing.assert_close(native.state, state)
        with self.assertRaisesRegex(ValueError, "threads_per_row"):
            NativeDendriticMatrixRuntime(graph, model, state, threads_per_row=7)
        with self.assertRaisesRegex(RuntimeError, "parameter"):
            self.extension.dendritic_step(state, state, state, [state],
                graph.in_rowptr, graph.in_col, graph.in_weight, None, 0., None, 0., 32)


if __name__ == "__main__":
    unittest.main()
