from __future__ import annotations

import sys
from pathlib import Path
import unittest

import torch


# The benchmark is intentionally a standalone script rather than a package
# module.  Import it through the local benchmark directory for focused tests.
_BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(_BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(_BENCHMARKS))

from gomoku_learning import (  # noqa: E402
    SparseRatePolicyValue,
    _teacher_action_and_value,
    _loss_and_metrics,
    make_gomoku_dataset,
    make_random_rollout_dataset,
)
from neurosim_gpu import GomokuEnv, random_topology  # noqa: E402


class SparseSensoryProjectionTests(unittest.TestCase):
    def test_each_input_feature_has_fixed_fanout(self) -> None:
        board_size = 5
        fanout = 4
        model = SparseRatePolicyValue(
            board_size,
            neurons=24,
            degree=3,
            recurrence_steps=2,
            sensory_fanout=fanout,
            seed=11,
            device=torch.device("cpu"),
        )
        expected_features = 3 * board_size * board_size
        self.assertEqual(model.sensory_src.numel(), expected_features * fanout)
        counts = torch.bincount(
            model.sensory_src,
            minlength=expected_features,
        )
        torch.testing.assert_close(
            counts,
            torch.full_like(counts, fanout),
        )

    def test_csr_values_follow_destination_permutation(self) -> None:
        neurons = 32
        degree = 5
        seed = 17
        model = SparseRatePolicyValue(
            board_size=5,
            neurons=neurons,
            degree=degree,
            recurrence_steps=1,
            sensory_fanout=4,
            seed=seed,
            device=torch.device("cpu"),
        )
        topology = random_topology(
            neurons,
            neurons * degree,
            device="cpu",
            dtype=torch.float32,
            seed=seed,
            weight_scale=0.18,
        )
        csr = model._csr()
        torch.testing.assert_close(csr.values(), topology.in_weight)
        torch.testing.assert_close(model.in_col, topology.in_col)

    def test_sparse_brain_forward_backward_has_edge_gradient(self) -> None:
        model = SparseRatePolicyValue(
            board_size=5,
            neurons=24,
            degree=4,
            recurrence_steps=2,
            sensory_fanout=4,
            seed=23,
            device=torch.device("cpu"),
        )
        dataset = make_gomoku_dataset(
            12,
            board_size=5,
            win_length=4,
            noise_stones=3,
            seed=29,
        )
        logits, value = model(dataset.boards)
        loss, _, _, _ = _loss_and_metrics(
            logits,
            value,
            dataset.policy_target,
            dataset.value_target,
        )
        loss.backward()
        self.assertIsNotNone(model.edge_weight.grad)
        assert model.edge_weight.grad is not None
        self.assertTrue(bool(torch.isfinite(model.edge_weight.grad).all()))
        self.assertGreater(float(model.edge_weight.grad.abs().sum()), 0.0)

    def test_sensory_weights_can_be_counted_as_trainable(self) -> None:
        model = SparseRatePolicyValue(
            board_size=5,
            neurons=24,
            degree=4,
            recurrence_steps=1,
            sensory_fanout=4,
            learn_sensory=True,
            seed=31,
            device=torch.device("cpu"),
        )
        self.assertTrue(model.sensory_weight.requires_grad)
        self.assertEqual(model.fixed_param_count, model.alpha.numel() + model.gain.numel())
        self.assertIn("sensory_weight", dict(model.named_parameters()))

    def test_random_rollout_dataset_is_deterministic_and_legal(self) -> None:
        kwargs = dict(
            samples=24,
            board_size=5,
            win_length=4,
            max_random_moves=8,
            seed=41,
        )
        first = make_random_rollout_dataset(**kwargs)
        second = make_random_rollout_dataset(**kwargs)
        torch.testing.assert_close(first.boards, second.boards)
        torch.testing.assert_close(first.policy_target, second.policy_target)
        torch.testing.assert_close(first.value_target, second.value_target)

        occupied = first.boards[:, :2].sum(dim=1).flatten(1) > 0
        selected = occupied.gather(1, first.policy_target[:, None]).squeeze(1)
        self.assertFalse(bool(selected.any()))
        self.assertTrue(bool(torch.isfinite(first.value_target).all()))
        self.assertTrue(bool(((first.value_target >= -1) & (first.value_target <= 1)).all()))

    def test_rollout_teacher_prefers_win_then_block(self) -> None:
        winning = GomokuEnv(5, win_length=4)
        winning.board[0, 0:3] = 1
        winning.board[2, 0:2] = -1
        winning.current_player = 1
        action, value = _teacher_action_and_value(winning)
        self.assertEqual(action, 3)
        self.assertEqual(value, 1.0)

        blocking = GomokuEnv(5, win_length=4)
        blocking.board[1, 0:3] = -1
        blocking.board[0, 0:2] = 1
        blocking.current_player = 1
        action, value = _teacher_action_and_value(blocking)
        self.assertEqual(action, 8)
        self.assertEqual(value, -1.0)


if __name__ == "__main__":
    unittest.main()
