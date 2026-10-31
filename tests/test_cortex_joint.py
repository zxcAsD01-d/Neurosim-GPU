from __future__ import annotations

import unittest

import torch

from neurosim_gpu.cortex.joint import JointConfig, JointObjective


class JointTests(unittest.TestCase):
    def test_joint_objective_returns_all_terms(self) -> None:
        heads = JointObjective(
            16,
            question_token_ids=(9,),
            config=JointConfig(
                time_weight=0.2,
                question_weight=0.2,
                compute_weight=0.1,
            ),
        )
        readout = torch.randn(2, 16)
        labels = torch.tensor([9, 3])
        lm = torch.tensor(2.0, requires_grad=True)
        loss, metrics = heads(readout, lm, labels, 0.5, 2, 4)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(float(loss), 2.0)
        self.assertEqual(set(metrics), {
            "lm", "time", "question", "question_accuracy", "compute", "time_mae"
        })
        self.assertGreater(float(metrics["time"]), 0.0)


if __name__ == "__main__":
    unittest.main()
