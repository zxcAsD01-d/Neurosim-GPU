from __future__ import annotations

import unittest

import torch

from neurosim_gpu.cortex.halting import (
    HaltingConfig,
    hard_ticks,
    soft_halting,
)


class HaltingTests(unittest.TestCase):
    def test_soft_weights_and_expected_ticks(self) -> None:
        readouts = torch.randn(2, 5, 8)
        logits = torch.zeros(2, 5)
        weighted, weights, expected = soft_halting(
            readouts, logits, HaltingConfig()
        )
        self.assertEqual(weighted.shape, (2, 8))
        self.assertTrue(
            torch.allclose(
                weights.sum(dim=1),
                torch.ones(2),
                atol=1e-5,
            )
        )
        self.assertTrue(bool((expected >= 1).all()))
        self.assertTrue(bool((expected <= 5).all()))

    def test_hard_ticks_stops_at_threshold(self) -> None:
        logits = torch.tensor([[5.0, 5.0, 5.0], [-5.0, -5.0, 5.0]])
        ticks = hard_ticks(logits, 0.5)
        self.assertTrue(bool((ticks >= 1).all()))
        self.assertTrue(bool((ticks <= 3).all()))


if __name__ == "__main__":
    unittest.main()
