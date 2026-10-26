from __future__ import annotations

import unittest

import torch

from neurosim_gpu import Dynamics, MatrixRuntime, SparseTopology
from neurosim_gpu.continual import (
    ContinualLearningConfig,
    ContinualLearningController,
    ReplayBuffer,
)
from neurosim_gpu.scheduler import MultiRateScheduler


class ContinualTests(unittest.TestCase):
    def setUp(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float64
        topology = SparseTopology.from_edges(
            3,
            torch.tensor([0, 1], device=self.device),
            torch.tensor([1, 2], device=self.device),
            torch.tensor([0.1, 0.2], device=self.device, dtype=dtype),
        )
        dynamics = Dynamics(
            alpha=torch.full((3,), 0.9, device=self.device, dtype=dtype),
            gain=torch.full((3,), 0.5, device=self.device, dtype=dtype),
        )
        runtime = MatrixRuntime(
            topology,
            dynamics,
            torch.full((1, 3), 0.2, device=self.device, dtype=dtype),
        )
        self.scheduler = MultiRateScheduler(runtime)

    def transition(self, value: float) -> tuple[torch.Tensor, ...]:
        dtype = torch.float64
        state = torch.full((1, 3), value, device=self.device, dtype=dtype)
        return (
            state,
            torch.tensor([[1.0]], device=self.device, dtype=dtype),
            torch.tensor([value], device=self.device, dtype=dtype),
            state + 0.1,
            torch.tensor([False], device=self.device),
        )

    def test_ring_buffer_and_sampling(self) -> None:
        buffer = ReplayBuffer(3, device=self.device, seed=7)
        for value in (1.0, 2.0, 3.0, 4.0):
            buffer.append(*self.transition(value))
        self.assertEqual(len(buffer), 3)
        batch = buffer.sample(8)
        self.assertEqual(batch.state.shape, (8, 3))
        self.assertTrue(bool(torch.all(batch.state[:, 0] >= 2.0)))

    def test_append_batch_larger_than_capacity_keeps_newest(self) -> None:
        buffer = ReplayBuffer(3, device=self.device)
        values = torch.arange(5, device=self.device, dtype=torch.float64).reshape(5, 1).expand(5, 3)
        action = torch.zeros((5, 1), device=self.device, dtype=torch.float64)
        reward = torch.arange(5, device=self.device, dtype=torch.float64)
        terminal = torch.zeros(5, device=self.device, dtype=torch.bool)
        buffer.append(values, action, reward, values, terminal)
        self.assertEqual(len(buffer), 3)
        self.assertTrue(bool(torch.all(buffer._state[:, 0] >= 2.0)))

    def test_invalid_device_append_does_not_initialize_buffer(self) -> None:
        buffer = ReplayBuffer(4, device=self.device)
        other = torch.device("cpu") if self.device.type == "cuda" else torch.device("meta")
        state = torch.zeros((1, 3), device=other)
        with self.assertRaisesRegex(ValueError, "buffer device"):
            buffer.append(state, state[:, :1], state[:, 0], state, state[:, 0].bool())
        self.assertEqual(len(buffer), 0)

    def test_controller_resume_is_deterministic(self) -> None:
        replay = ReplayBuffer(8, device=self.device, seed=11)
        controller = ContinualLearningController(
            self.scheduler,
            replay,
            config=ContinualLearningConfig(replay_every=2),
        )
        for value in (0.1, 0.2):
            controller.observe(*self.transition(value))
            controller.step(0.1)
        payload = controller.state_dict()
        expected = controller.step(0.1).clone()
        replay2 = ReplayBuffer(8, device=self.device, seed=11)
        scheduler2 = MultiRateScheduler(
            MatrixRuntime(
                self.scheduler.runtime.topology,
                self.scheduler.runtime.dynamics,
                self.scheduler.runtime.state.clone(),
            )
        )
        controller2 = ContinualLearningController(
            scheduler2,
            replay2,
            config=ContinualLearningConfig(replay_every=2),
        )
        controller2.load_state_dict(payload)
        actual = controller2.step(0.1)
        torch.testing.assert_close(actual, expected, rtol=0.0, atol=1e-12)
        torch.testing.assert_close(
            controller2.replay.sample(4).state,
            controller.replay.sample(4).state,
        )

    def test_replay_cadence_is_explicit(self) -> None:
        replay = ReplayBuffer(8, device=self.device)
        controller = ContinualLearningController(
            self.scheduler,
            replay,
            config=ContinualLearningConfig(replay_every=2),
        )
        controller.observe(*self.transition(0.1))
        controller.step(0.1)
        self.assertFalse(controller.replay_due)
        self.assertIsNone(controller.maybe_replay_batch(1))
        controller.observe(*self.transition(0.2))
        controller.step(0.1)
        self.assertTrue(controller.replay_due)
        self.assertIsNotNone(controller.maybe_replay_batch(1))


if __name__ == "__main__":
    unittest.main()
