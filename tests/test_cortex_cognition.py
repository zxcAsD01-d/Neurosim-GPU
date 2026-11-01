from __future__ import annotations

import unittest

import torch

from neurosim_gpu.cortex.cognition import (
    GlobalWorkspace,
    HippocampalMemory,
    SelfMonitor,
    WorkspaceConfig,
    expected_calibration_error,
    should_ask,
)


class CognitionTests(unittest.TestCase):
    def test_workspace_broadcast_and_ignition(self) -> None:
        workspace = GlobalWorkspace(32, WorkspaceConfig(slots=16, top_k=4))
        hidden = torch.randn(3, 32)
        broadcast, stats = workspace(hidden)
        self.assertEqual(broadcast.shape, (3, 32))
        self.assertEqual(stats["weights"].shape, (3, 4))
        self.assertTrue(
            torch.allclose(
                stats["weights"].sum(-1),
                torch.ones(3),
                atol=1e-5,
            )
        )
        self.assertTrue(bool((stats["ignition"] >= 0).all()))

    def test_hippocampal_one_shot_recall(self) -> None:
        memory = HippocampalMemory(32, slots=32, sparse_k=4)
        generator = torch.Generator().manual_seed(0)
        patterns = torch.randn(8, 32, generator=generator)
        for pattern in patterns:
            memory.write(pattern)
        noisy = patterns + 0.15 * torch.randn_like(patterns)
        recalled = memory.read(noisy)
        cosine = torch.nn.functional.cosine_similarity(
            recalled, patterns, dim=-1
        )
        self.assertGreater(float(cosine.mean()), 0.8)

    def test_calibration_and_ask_policy(self) -> None:
        confidence = torch.tensor([0.9, 0.8, 0.2, 0.1])
        correct = torch.tensor([1, 0, 0, 0])
        error = expected_calibration_error(confidence, correct, bins=4)
        self.assertGreaterEqual(error, 0.0)
        ask = should_ask(confidence, 0.5)
        self.assertEqual(ask.tolist(), [False, False, True, True])

    def test_self_monitor_shapes(self) -> None:
        monitor = SelfMonitor(16)
        output = monitor(torch.randn(4, 16))
        for value in output.values():
            self.assertEqual(value.shape, (4,))


if __name__ == "__main__":
    unittest.main()
