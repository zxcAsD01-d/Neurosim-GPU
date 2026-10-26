from __future__ import annotations

import io
import unittest

import torch

from neurosim_gpu import Dynamics, HybridPELSRuntime, MatrixRuntime, SparseTopology
from neurosim_gpu import (
    DopamineMatrixRuntime,
    DopaminePlasticityConfig,
    DopamineReceptors,
    RegionalDopamineField,
)
from neurosim_gpu.checkpoint import (
    capture_runtime,
    load_runtime_checkpoint,
    restore_runtime,
    save_runtime_checkpoint,
)
from neurosim_gpu.communication import DelayedMultiChannelCommunication


class CheckpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float64
        self.topology = SparseTopology.from_edges(
            4,
            torch.tensor([0, 0, 1, 2], device=self.device),
            torch.tensor([1, 2, 3, 3], device=self.device),
            torch.tensor([0.2, 0.1, -0.1, 0.15], device=self.device, dtype=self.dtype),
        )
        self.dynamics = Dynamics(
            alpha=torch.full((4,), 0.9, device=self.device, dtype=self.dtype),
            gain=torch.full((4,), 0.7, device=self.device, dtype=self.dtype),
        )
        self.initial = torch.full(
            (1, 4), 0.3, device=self.device, dtype=self.dtype
        )

    def make_comm(self) -> DelayedMultiChannelCommunication:
        return DelayedMultiChannelCommunication(
            4,
            torch.tensor([0, 1], device=self.device),
            torch.tensor([1, 2], device=self.device),
            torch.tensor([1.0, -0.5], device=self.device, dtype=self.dtype),
            torch.tensor([2, 0], device=self.device),
            torch.tensor([0, 0], device=self.device),
            torch.tensor([1.0], device=self.device, dtype=self.dtype),
        )

    def test_hybrid_resume_reproduces_next_state(self) -> None:
        runtime = HybridPELSRuntime(
            self.topology,
            self.dynamics,
            self.initial.clone(),
            theta=0.02,
            propagation="matrix",
        )
        runtime.step(0.15)
        payload = capture_runtime(runtime)
        expected = runtime.step(0.15).clone()

        restored = HybridPELSRuntime(
            self.topology,
            self.dynamics,
            self.initial.clone(),
            theta=0.02,
            propagation="matrix",
        )
        restore_runtime(restored, payload)
        actual = restored.step(0.15)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)

    def test_communication_history_roundtrip(self) -> None:
        communication = self.make_comm()
        for value in (1.0, 2.0, 4.0):
            communication.step(
                torch.tensor(
                    [[value, 0.0, 0.0, 0.0]],
                    device=self.device,
                    dtype=self.dtype,
                )
            )
        runtime = MatrixRuntime(self.topology, self.dynamics, self.initial.clone())
        payload = capture_runtime(runtime, communication=communication)
        expected = communication.step(
            torch.tensor(
                [[8.0, 0.0, 0.0, 0.0]],
                device=self.device,
                dtype=self.dtype,
            )
        )
        restored_comm = self.make_comm()
        restored_runtime = MatrixRuntime(
            self.topology, self.dynamics, self.initial.clone()
        )
        restore_runtime(restored_runtime, payload, communication=restored_comm)
        actual = restored_comm.step(
            torch.tensor(
                [[8.0, 0.0, 0.0, 0.0]],
                device=self.device,
                dtype=self.dtype,
            )
        )
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_file_and_buffer_roundtrip(self) -> None:
        runtime = MatrixRuntime(self.topology, self.dynamics, self.initial.clone())
        runtime.step(0.1)
        buffer = io.BytesIO()
        save_runtime_checkpoint(runtime, buffer)
        buffer.seek(0)
        restored = MatrixRuntime(
            self.topology, self.dynamics, self.initial.clone()
        )
        load_runtime_checkpoint(buffer, restored)
        torch.testing.assert_close(restored.state, runtime.state)

    def test_structure_mismatch_is_rejected(self) -> None:
        runtime = MatrixRuntime(self.topology, self.dynamics, self.initial.clone())
        payload = capture_runtime(runtime)
        other_topology = SparseTopology.from_edges(
            4,
            torch.tensor([0], device=self.device),
            torch.tensor([3], device=self.device),
            torch.tensor([0.2], device=self.device, dtype=self.dtype),
        )
        other = MatrixRuntime(other_topology, self.dynamics, self.initial.clone())
        with self.assertRaisesRegex(ValueError, "topology"):
            restore_runtime(other, payload)

    def test_parameter_mismatch_is_rejected(self) -> None:
        runtime = MatrixRuntime(self.topology, self.dynamics, self.initial.clone())
        payload = capture_runtime(runtime)
        altered = Dynamics(
            alpha=self.dynamics.alpha * 0.99,
            gain=self.dynamics.gain,
        )
        other = MatrixRuntime(self.topology, altered, self.initial.clone())
        with self.assertRaisesRegex(ValueError, "parameters"):
            restore_runtime(other, payload)

    def test_dopamine_cache_resume_reproduces_next_transition(self) -> None:
        field_kwargs = dict(
            tonic_baseline=torch.full((1,), 0.5, device=self.device, dtype=self.dtype),
            uptake_rate=torch.full((1,), 0.1, device=self.device, dtype=self.dtype),
            decay_rate=torch.zeros(1, device=self.device, dtype=self.dtype),
            release_projection=torch.ones(1, 1, device=self.device, dtype=self.dtype),
            edge_src=torch.empty(0, device=self.device, dtype=torch.int64),
            edge_dst=torch.empty(0, device=self.device, dtype=torch.int64),
            diffusion_rate=torch.empty(0, device=self.device, dtype=self.dtype),
            dt=0.1,
        )
        receptors = DopamineReceptors(
            d1_density=torch.ones(4, device=self.device, dtype=self.dtype),
            d2_density=torch.zeros(4, device=self.device, dtype=self.dtype),
            tonic_concentration=0.5,
        )
        config = DopaminePlasticityConfig(
            chemistry_interval=3,
            validate_inputs=False,
        )
        runtime = DopamineMatrixRuntime(
            self.topology,
            self.dynamics,
            self.initial.clone(),
            chemistry=RegionalDopamineField(**field_kwargs),
            receptors=receptors,
            neuron_region=torch.zeros(4, device=self.device, dtype=torch.int64),
            config=config,
        )
        for value in (0.2, 0.0):
            runtime.step(0.1, rpe=torch.tensor([[value]], device=self.device, dtype=self.dtype))
        payload = capture_runtime(runtime)
        expected = runtime.step(0.1, rpe=torch.tensor([[0.0]], device=self.device, dtype=self.dtype)).clone()
        restored = DopamineMatrixRuntime(
            self.topology,
            self.dynamics,
            self.initial.clone(),
            chemistry=RegionalDopamineField(**field_kwargs),
            receptors=receptors,
            neuron_region=torch.zeros(4, device=self.device, dtype=torch.int64),
            config=config,
        )
        restore_runtime(restored, payload)
        actual = restored.step(0.1, rpe=torch.tensor([[0.0]], device=self.device, dtype=self.dtype))
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
