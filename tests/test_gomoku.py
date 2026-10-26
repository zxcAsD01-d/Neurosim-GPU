from __future__ import annotations

import unittest

import torch

from neurosim_gpu.gomoku import (
    BLACK,
    EMPTY,
    WHITE,
    GomokuEnv,
    RandomOpponent,
    apply_move,
    canonical_board,
    canonical_encoding,
    canonical_move,
    canonical_symmetry_index,
    dihedral_symmetries,
    empty_board,
    encode_board,
    has_won,
    inverse_transform_move,
    is_draw,
    legal_moves,
    random_rollout,
    random_game,
    transform_move,
    winner,
)


class GomokuRulesTests(unittest.TestCase):
    def test_all_four_win_directions(self) -> None:
        cases = (
            [(0, c) for c in range(5)],
            [(r, 0) for r in range(5)],
            [(r, r) for r in range(5)],
            [(r, 4 - r) for r in range(5)],
        )
        for coordinates in cases:
            board = empty_board(9)
            for row, col in coordinates:
                board[row, col] = BLACK
            self.assertTrue(has_won(board, BLACK))
            self.assertEqual(winner(board), BLACK)

    def test_last_move_does_not_miss_diagonal(self) -> None:
        board = empty_board(9)
        moves = [(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)]
        for row, col in moves:
            board[row, col] = WHITE
        self.assertTrue(has_won(board, WHITE, last_move=moves[-1]))

    def test_illegal_move_is_rejected_without_mutation(self) -> None:
        env = GomokuEnv(9)
        env.play((2, 3))
        before = env.board.clone()
        with self.assertRaises(ValueError):
            env.play((2, 3))
        torch.testing.assert_close(env.board, before)
        with self.assertRaises(ValueError):
            env.play(81)

    def test_draw_detection_on_small_board(self) -> None:
        # Use a no-three-in-a-row pattern for both colors.
        board = torch.tensor(
            [[BLACK, WHITE, BLACK], [BLACK, WHITE, WHITE], [WHITE, BLACK, BLACK]],
            dtype=torch.int8,
        )
        self.assertFalse(has_won(board, BLACK, win_length=3))
        self.assertFalse(has_won(board, WHITE, win_length=3))
        self.assertTrue(is_draw(board, win_length=3))


class GomokuEncodingTests(unittest.TestCase):
    def test_canonical_encoding_is_player_relative(self) -> None:
        board = empty_board(9)
        board[1, 2] = BLACK
        board[4, 5] = WHITE
        encoded = encode_board(board, BLACK)
        self.assertEqual(tuple(encoded.shape), (3, 9, 9))
        self.assertEqual(float(encoded[0, 1, 2]), 1.0)
        self.assertEqual(float(encoded[1, 4, 5]), 1.0)
        torch.testing.assert_close(encode_board(board, WHITE)[0], encoded[1])

    def test_symmetries_share_canonical_representation(self) -> None:
        board = empty_board(9)
        board[0, 1] = BLACK
        board[2, 6] = WHITE
        expected = canonical_board(board, BLACK, symmetry=True)
        expected_encoding = canonical_encoding(board, BLACK)
        for transformed in dihedral_symmetries(board):
            torch.testing.assert_close(canonical_board(transformed, BLACK, symmetry=True), expected)
            torch.testing.assert_close(canonical_encoding(transformed, BLACK), expected_encoding)

    def test_action_transform_round_trip_and_canonical_alignment(self) -> None:
        board = empty_board(9)
        board[1, 2] = BLACK
        board[7, 4] = WHITE
        action = 5 * 9 + 6
        index = canonical_symmetry_index(board, BLACK)
        canonical_action = canonical_move(board, action, BLACK)
        self.assertEqual(canonical_action, transform_move(action, 9, index))
        self.assertEqual(inverse_transform_move(canonical_action, 9, index), action)

        # A marker at the transformed action lands in the same cell as the
        # corresponding transformed board.
        marked = board.clone()
        marked.reshape(-1)[action] = BLACK
        transformed = dihedral_symmetries(marked)[index]
        self.assertEqual(int(transformed.reshape(-1)[canonical_action]), BLACK)

    def test_batch_encoding_is_independent(self) -> None:
        first = empty_board(9)
        first[0, 0] = BLACK
        second = empty_board(9)
        second[8, 8] = BLACK
        batch = torch.stack((first, second))
        encoded = encode_board(batch, BLACK)
        self.assertEqual(tuple(encoded.shape), (2, 3, 9, 9))
        self.assertEqual(float(encoded[0, 0, 0, 0]), 1.0)
        self.assertEqual(float(encoded[1, 0, 8, 8]), 1.0)
        self.assertEqual(float(encoded[0, 0, 8, 8]), 0.0)

    def test_legal_moves_preserve_batch_independence(self) -> None:
        first = empty_board(5)
        second = empty_board(5)
        first[0, 0] = BLACK
        second[0, 1] = WHITE
        moves = legal_moves(torch.stack((first, second)))
        self.assertEqual(len(moves), 2)
        self.assertNotIn(0, moves[0].tolist())
        self.assertIn(0, moves[1].tolist())
        self.assertNotIn(1, moves[1].tolist())


class GomokuRandomTests(unittest.TestCase):
    def test_seeded_opponent_and_rollout_are_deterministic(self) -> None:
        first = GomokuEnv(5, win_length=4)
        second = GomokuEnv(5, win_length=4)
        generator_a = torch.Generator().manual_seed(1234)
        generator_b = torch.Generator().manual_seed(1234)
        result_a = random_rollout(first, generator=generator_a)
        result_b = random_rollout(second, generator=generator_b)
        self.assertEqual(result_a.moves, result_b.moves)
        torch.testing.assert_close(result_a.board, result_b.board)

        opponent_a = RandomOpponent(99)
        opponent_b = RandomOpponent(99)
        board = empty_board(5)
        self.assertEqual(opponent_a.choose(board), opponent_b.choose(board))
        game_a = random_game(5, win_length=4, seed=77)
        game_b = random_game(5, win_length=4, seed=77)
        self.assertEqual(game_a.moves, game_b.moves)
        torch.testing.assert_close(game_a.board, game_b.board)

    def test_environment_step_and_reset(self) -> None:
        env = GomokuEnv(5, win_length=3)
        observation, reward, done, info = env.step((0, 0))
        self.assertEqual(tuple(observation.shape), (5, 5))
        self.assertEqual(reward, 0.0)
        self.assertFalse(done)
        self.assertEqual(info["player"], BLACK)
        self.assertEqual(env.last_move, 0)
        reset = env.reset()
        self.assertEqual(int(reset.sum()), 0)
        self.assertEqual(env.current_player, BLACK)
        self.assertIsNone(env.last_move)


if __name__ == "__main__":
    unittest.main()
