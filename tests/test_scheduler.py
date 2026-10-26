from __future__ import annotations

import io
import unittest

import torch

from neurosim_gpu import Dynamics, MatrixRuntime, SparseTopology
from neurosim_gpu.scheduler import MultiRateScheduler, SchedulerConfig


class SchedulerTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float64
        self.topology = SparseTopology.from_edges(
            3,
            torch.tensor([0, 1], device=self.device),
            torch.tensor([1, 2], device=self.device),
            torch.tensor([0.1, 0.2], dtype=dtype, device=self.device),
        )
        self.dynamics = Dynamics(
            alpha=torch.full((3,), 0.9, dtype=dtype, device=self.device),
            gain=torch.full((3,), 0.5, dtype=dtype, device=self.device),
        )

    def runtime(self) -> MatrixRuntime:
        return MatrixRuntime(
            self.topology,
            self.dynamics,
            torch.full((1, 3), 0.2, dtype=self.dynamics.alpha.dtype, device=self.device),
        )

    def test_callbacks_run_at_exact_boundaries(self) -> None:
        migration_ticks: list[int] = []
        structural_ticks: list[int] = []
        scheduler = MultiRateScheduler(
            self.runtime(),
            config=SchedulerConfig(migration_interval=2, structural_interval=3),
            migration_callback=lambda s: migration_ticks.append(s.tick),
            structural_callback=lambda s: structural_ticks.append(s.tick) or None,
        )
        flags = []
        for _ in range(6):
            _, info = scheduler.step(0.1)
            flags.append((info.migration_applied, info.structural_applied))
        self.assertEqual(migration_ticks, [2, 4, 6])
        self.assertEqual(structural_ticks, [3, 6])
        self.assertEqual(flags, [(False, False), (True, False), (False, False), (True, False), (False, False), (True, False)])

    def test_structural_replacement_resets_edge_state(self) -> None:
        replacement = SparseTopology.from_edges(
            3,
            torch.tensor([0, 2], device=self.device),
            torch.tensor([2, 1], device=self.device),
            torch.tensor([0.3, 0.4], dtype=self.dynamics.alpha.dtype, device=self.device),
        )
        scheduler = MultiRateScheduler(
            self.runtime(),
            config=SchedulerConfig(structural_interval=1),
            structural_callback=lambda _: replacement,
        )
        scheduler.step(0.1)
        self.assertEqual(scheduler.runtime.topology.n_edges, 2)
        self.assertEqual(scheduler.structural_epochs, 1)

    def test_structural_boundary_can_rebuild_communication_transactionally(self) -> None:
        replacement = SparseTopology.from_edges(
            3,
            torch.tensor([0], device=self.device),
            torch.tensor([2], device=self.device),
            torch.tensor([0.3], dtype=self.dynamics.alpha.dtype, device=self.device),
        )
        seen = []
        scheduler = MultiRateScheduler(
            self.runtime(),
            config=SchedulerConfig(structural_interval=1),
            structural_callback=lambda _: replacement,
            communication_rebuilder=lambda _, topology: seen.append(topology.n_edges) or None,
        )
        scheduler.step(0.1)
        self.assertEqual(seen, [1])

    def test_structural_replacement_rejects_stale_communication_graph(self) -> None:
        from neurosim_gpu.communication import DelayedMultiChannelCommunication

        communication = DelayedMultiChannelCommunication(
            3,
            torch.tensor([0], device=self.device),
            torch.tensor([1], device=self.device),
            torch.tensor([1.0], device=self.device),
            torch.tensor([0], device=self.device),
            torch.tensor([0], device=self.device),
            torch.tensor([1.0], device=self.device),
        )
        replacement = SparseTopology.from_edges(
            3,
            torch.tensor([0], device=self.device),
            torch.tensor([2], device=self.device),
            torch.tensor([0.3], dtype=self.dynamics.alpha.dtype, device=self.device),
        )
        scheduler = MultiRateScheduler(
            self.runtime(),
            communication=communication,
        )
        with self.assertRaisesRegex(ValueError, "communication_rebuilder"):
            scheduler.replace_topology(replacement)

    def test_communication_current_is_added_before_neural_transition(self) -> None:
        from neurosim_gpu.communication import DelayedMultiChannelCommunication

        communication = DelayedMultiChannelCommunication(
            3,
            torch.tensor([0], device=self.device),
            torch.tensor([1], device=self.device),
            torch.tensor([1.0], dtype=self.dynamics.alpha.dtype, device=self.device),
            torch.tensor([0], device=self.device),
            torch.tensor([0], device=self.device),
            torch.tensor([1.0], dtype=self.dynamics.alpha.dtype, device=self.device),
        )
        runtime = self.runtime()
        scheduler = MultiRateScheduler(runtime, communication=communication)
        before = runtime.state.clone()
        scheduler.step(0.0)
        # The first communication tick has no delayed/current history effect for
        # this same-tick graph beyond the source state; subsequent calls remain
        # causal and stateful rather than being applied after the transition.
        self.assertFalse(torch.equal(runtime.state, before))

    def test_checkpoint_resume_preserves_scheduler_counters(self) -> None:
        scheduler = MultiRateScheduler(
            self.runtime(),
            config=SchedulerConfig(migration_interval=2),
        )
        scheduler.step(0.1)
        scheduler.step(0.1)
        payload = scheduler.state_dict()
        expected, _ = scheduler.step(0.1)

        resumed = MultiRateScheduler(
            self.runtime(),
            config=SchedulerConfig(migration_interval=2),
        )
        resumed.load_state_dict(payload)
        actual, info = resumed.step(0.1)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)
        self.assertEqual(info.tick, 3)


if __name__ == "__main__":
    unittest.main()
