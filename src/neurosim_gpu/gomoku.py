"""Small, deterministic Gomoku environment and tensor utilities.

The module is intentionally CPU-first and has no dependency on an RL or game
framework.  Boards are encoded with ``0`` for an empty intersection, ``1`` for
the first player and ``-1`` for the second player.  All functional helpers are
out-of-place and accept tensors on either CPU or CUDA; the stateful environment
keeps its board on the requested device.

The canonical representation is from the point of view of a player: that
player's stones are ``+1`` and the opponent's are ``-1``.  ``canonical_board``
can additionally quotient out the eight square-board symmetries.  This is
useful for replay storage and makes equivalent positions share one key without
requiring a hash-table implementation in the hot path.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, Iterator, Sequence

import torch


EMPTY = 0
BLACK = 1
WHITE = -1
PLAYERS = (BLACK, WHITE)


def _validate_dimensions(board_size: int, win_length: int) -> None:
    if isinstance(board_size, bool) or int(board_size) != board_size:
        raise ValueError("board_size must be an integer")
    if isinstance(win_length, bool) or int(win_length) != win_length:
        raise ValueError("win_length must be an integer")
    if board_size < 1:
        raise ValueError("board_size must be positive")
    if win_length < 1 or win_length > board_size:
        raise ValueError("win_length must be in [1, board_size]")


def _validate_player(player: int) -> int:
    value = int(player)
    if value not in PLAYERS:
        raise ValueError("player must be BLACK (1) or WHITE (-1)")
    return value


def _as_single_board(board: torch.Tensor) -> torch.Tensor:
    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    if board.ndim != 2 or board.shape[0] != board.shape[1]:
        raise ValueError("board must have shape [size, size]")
    if board.numel():
        valid = (board == EMPTY) | (board == BLACK) | (board == WHITE)
        if not bool(torch.all(valid)):
            raise ValueError("board values must be -1, 0, or 1")
    return board


def _as_batch(board: torch.Tensor) -> torch.Tensor:
    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    if board.ndim not in (2, 3):
        raise ValueError("board must have shape [size, size] or [batch, size, size]")
    if board.ndim == 2:
        board = board.unsqueeze(0)
    if board.shape[1] != board.shape[2]:
        raise ValueError("board must be square")
    if board.numel():
        valid = (board == EMPTY) | (board == BLACK) | (board == WHITE)
        if not bool(torch.all(valid)):
            raise ValueError("board values must be -1, 0, or 1")
    return board


def empty_board(
    board_size: int = 9,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.int8,
) -> torch.Tensor:
    """Return an empty square board."""

    if board_size < 1:
        raise ValueError("board_size must be positive")
    if dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
        raise ValueError("board dtype must be an integer torch dtype")
    return torch.zeros((board_size, board_size), dtype=dtype, device=device)


@dataclass(frozen=True)
class GomokuConfig:
    """Immutable game configuration."""

    board_size: int = 9
    win_length: int = 5
    first_player: int = BLACK

    def __post_init__(self) -> None:
        _validate_dimensions(self.board_size, self.win_length)
        _validate_player(self.first_player)


@dataclass(frozen=True)
class MoveResult:
    """Result of one legal move made by :class:`GomokuEnv`."""

    move: int
    row: int
    col: int
    player: int
    winner: int
    done: bool
    reward: float
    move_count: int


@dataclass(frozen=True)
class RolloutResult:
    """Summary returned by :func:`random_rollout`."""

    moves: tuple[int, ...]
    winner: int
    draw: bool
    board: torch.Tensor


def legal_move_mask(board: torch.Tensor) -> torch.Tensor:
    """Return a boolean mask with ``True`` at every empty intersection.

    Both ``[H, W]`` and ``[B, H, W]`` are accepted and the original rank is
    retained.  This rank-preserving behavior makes the helper safe to use in a
    batched policy without accidentally coupling samples.
    """

    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    _as_batch(board)
    return board == EMPTY


def legal_moves(board: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
    """Return legal flat move indices.

    For a single board this returns a one-dimensional tensor.  For a batch it
    returns one independent one-dimensional tensor per sample; games with
    different numbers of empty intersections therefore do not require padding.
    """

    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    _as_batch(board)
    single = board.ndim == 2
    batch = board.unsqueeze(0) if single else board
    result = [torch.nonzero(item.reshape(-1) == EMPTY, as_tuple=False).flatten() for item in batch]
    return result[0] if single else result


def _line_length(board: torch.Tensor, row: int, col: int, dr: int, dc: int, player: int) -> int:
    size = board.shape[0]
    length = 1
    r, c = row + dr, col + dc
    while 0 <= r < size and 0 <= c < size and int(board[r, c]) == player:
        length += 1
        r += dr
        c += dc
    r, c = row - dr, col - dc
    while 0 <= r < size and 0 <= c < size and int(board[r, c]) == player:
        length += 1
        r -= dr
        c -= dc
    return length


def has_won(
    board: torch.Tensor,
    player: int,
    win_length: int = 5,
    last_move: int | Sequence[int] | torch.Tensor | None = None,
) -> bool | torch.Tensor:
    """Whether ``player`` has a winning line.

    ``last_move`` is an optional flat index or ``(row, col)`` pair.  Supplying
    it reduces a move-time check to four line walks; without it all occupied
    cells are inspected.  Batched boards return one boolean per sample.
    """

    player = _validate_player(player)
    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    boards = _as_batch(board)
    _validate_dimensions(int(boards.shape[1]), int(win_length))

    def one(item: torch.Tensor, move: int | Sequence[int] | torch.Tensor | None) -> bool:
        size = item.shape[0]
        candidates: Iterable[tuple[int, int]]
        if move is not None:
            row, col = _move_to_row_col(move, size)
            candidates = ((row, col),)
        else:
            coords = torch.nonzero(item == player, as_tuple=False)
            candidates = ((int(rc[0]), int(rc[1])) for rc in coords)
        for row, col in candidates:
            if int(item[row, col]) != player:
                continue
            for dr, dc in ((1, 0), (0, 1), (1, 1), (1, -1)):
                if _line_length(item, row, col, dr, dc, player) >= win_length:
                    return True
        return False

    if board.ndim == 2:
        return one(boards[0], last_move)
    if last_move is None:
        moves = [None] * boards.shape[0]
    elif isinstance(last_move, torch.Tensor) and last_move.ndim > 1:
        moves = list(last_move)
    elif isinstance(last_move, (list, tuple)) and len(last_move) == boards.shape[0] and any(
        isinstance(value, (list, tuple, torch.Tensor)) for value in last_move
    ):
        moves = list(last_move)
    else:
        moves = [last_move] * boards.shape[0]
    return torch.tensor([one(item, move) for item, move in zip(boards, moves)], dtype=torch.bool, device=board.device)


def winner(
    board: torch.Tensor,
    win_length: int = 5,
    last_move: int | Sequence[int] | torch.Tensor | None = None,
) -> int | torch.Tensor:
    """Return ``1``, ``-1`` or ``0`` for a board, or a batch of such values."""

    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    boards = _as_batch(board)
    _validate_dimensions(int(boards.shape[1]), int(win_length))

    def one(item: torch.Tensor, move: object) -> int:
        # A legal game can only have one winner.  Checking both players also
        # makes this utility useful for externally loaded positions.
        if has_won(item, BLACK, win_length, move):
            return BLACK
        if has_won(item, WHITE, win_length, move):
            return WHITE
        return EMPTY

    if board.ndim == 2:
        return one(boards[0], last_move)
    if last_move is None:
        moves = [None] * boards.shape[0]
    elif isinstance(last_move, torch.Tensor) and last_move.ndim > 1:
        moves = list(last_move)
    elif isinstance(last_move, (list, tuple)) and len(last_move) == boards.shape[0] and any(
        isinstance(value, (list, tuple, torch.Tensor)) for value in last_move
    ):
        moves = list(last_move)
    else:
        moves = [last_move] * boards.shape[0]
    return torch.tensor([one(item, move) for item, move in zip(boards, moves)], dtype=torch.int8, device=board.device)


def is_draw(
    board: torch.Tensor,
    win_length: int = 5,
    last_move: int | Sequence[int] | torch.Tensor | None = None,
) -> bool | torch.Tensor:
    """Return true when the board is full and no player has won."""

    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    boards = _as_batch(board)
    wins = winner(board, win_length, last_move)
    if board.ndim == 2:
        return bool(torch.all(boards[0] != EMPTY)) and int(wins) == EMPTY
    return (torch.all(boards != EMPTY, dim=(1, 2))) & (wins == EMPTY)


def _move_to_row_col(move: int | Sequence[int] | torch.Tensor, board_size: int) -> tuple[int, int]:
    if isinstance(move, torch.Tensor):
        if move.numel() != 1 and move.numel() != 2:
            raise ValueError("move tensor must contain one flat index or (row, col)")
        values = [int(v) for v in move.detach().cpu().reshape(-1)]
    elif isinstance(move, (tuple, list)):
        values = [int(v) for v in move]
    else:
        values = [int(move)]
    if len(values) == 1:
        flat = values[0]
        if flat < 0 or flat >= board_size * board_size:
            raise ValueError("move index is outside the board")
        return divmod(flat, board_size)
    if len(values) == 2:
        row, col = values
        if not (0 <= row < board_size and 0 <= col < board_size):
            raise ValueError("move coordinate is outside the board")
        return row, col
    raise ValueError("move must be a flat index or (row, col)")


def apply_move(
    board: torch.Tensor,
    move: int | Sequence[int] | torch.Tensor,
    player: int,
    *,
    validate: bool = True,
) -> torch.Tensor:
    """Return a copy of ``board`` with one stone placed."""

    player = _validate_player(player)
    item = _as_single_board(board)
    row, col = _move_to_row_col(move, item.shape[0])
    if validate and int(item[row, col]) != EMPTY:
        raise ValueError("move is illegal: intersection is occupied")
    result = item.clone()
    result[row, col] = player
    return result


def dihedral_symmetries(board: torch.Tensor) -> torch.Tensor:
    """Return the eight rotations/reflections of a square board.

    A single board returns ``[8, H, H]`` and a batch returns ``[B, 8, H, H]``.
    The order is deterministic: rotations of the original followed by
    rotations of its left-right reflection.
    """

    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    boards = _as_batch(board)
    values = []
    for k in range(4):
        values.append(torch.rot90(boards, k, dims=(-2, -1)))
    reflected = torch.flip(boards, dims=(-1,))
    for k in range(4):
        values.append(torch.rot90(reflected, k, dims=(-2, -1)))
    result = torch.stack(values, dim=1)
    return result[0] if board.ndim == 2 else result


def transform_move(
    move: int | Sequence[int] | torch.Tensor,
    board_size: int,
    symmetry: int,
) -> int:
    """Map a move through one of :func:`dihedral_symmetries`.

    Symmetry indices ``0..3`` are rotations by ``0, 90, 180, 270`` degrees;
    ``4..7`` first reflect left/right and then apply those rotations.  This
    helper is important when a symmetry-canonical board is paired with a
    policy target: the action must be transformed by the same index as the
    board.  The returned value is always a flat index.
    """

    if not isinstance(board_size, int) or board_size < 1:
        raise ValueError("board_size must be a positive integer")
    if not isinstance(symmetry, int) or not 0 <= symmetry < 8:
        raise ValueError("symmetry must be an integer in [0, 7]")
    row, col = _move_to_row_col(move, board_size)
    rotations = symmetry % 4
    if symmetry >= 4:
        col = board_size - 1 - col
    for _ in range(rotations):
        row, col = board_size - 1 - col, row
    return row * board_size + col


def inverse_transform_move(
    move: int | Sequence[int] | torch.Tensor,
    board_size: int,
    symmetry: int,
) -> int:
    """Map a canonical-board move back to the original board coordinates."""

    if not isinstance(board_size, int) or board_size < 1:
        raise ValueError("board_size must be a positive integer")
    if not isinstance(symmetry, int) or not 0 <= symmetry < 8:
        raise ValueError("symmetry must be an integer in [0, 7]")
    row, col = _move_to_row_col(move, board_size)
    for _ in range((4 - symmetry % 4) % 4):
        row, col = board_size - 1 - col, row
    if symmetry >= 4:
        col = board_size - 1 - col
    return row * board_size + col


def canonical_symmetry_index(
    board: torch.Tensor,
    player: int = BLACK,
) -> int | torch.Tensor:
    """Return the deterministic dihedral index selected by ``canonical_board``."""

    player = _validate_player(player)
    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    boards = _as_batch(board)

    def choose(item: torch.Tensor) -> int:
        candidates = dihedral_symmetries(item)
        best_key = tuple(int(value) for value in candidates[0].detach().cpu().reshape(-1))
        best_index = 0
        for index, candidate in enumerate(candidates[1:], start=1):
            key = tuple(int(value) for value in candidate.detach().cpu().reshape(-1))
            if key < best_key:
                best_key = key
                best_index = index
        return best_index

    signed = boards * player
    if board.ndim == 2:
        return choose(signed[0])
    return torch.tensor(
        [choose(item) for item in signed], dtype=torch.int64, device=board.device
    )


def canonical_board(
    board: torch.Tensor,
    player: int = BLACK,
    *,
    symmetry: bool = False,
) -> torch.Tensor:
    """Return a player-relative board, optionally canonicalized by symmetry.

    With ``symmetry=True`` all dihedral-equivalent positions map to exactly the
    same tensor.  Batch samples are canonicalized independently.
    """

    player = _validate_player(player)
    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    boards = _as_batch(board)
    signed = boards * player
    if not symmetry:
        return signed[0] if board.ndim == 2 else signed

    def choose(item: torch.Tensor) -> torch.Tensor:
        candidates = dihedral_symmetries(item)
        # Lexicographic comparison is deterministic and independent of device.
        # The tiny Python loop is only used at replay/key boundaries, not in the
        # neural tick hot path.
        best = candidates[0]
        best_key = tuple(int(value) for value in best.detach().cpu().reshape(-1))
        for candidate in candidates[1:]:
            key = tuple(int(value) for value in candidate.detach().cpu().reshape(-1))
            if key < best_key:
                best, best_key = candidate, key
        return best

    result = torch.stack([choose(item) for item in signed], dim=0)
    return result[0] if board.ndim == 2 else result


def canonical_move(
    board: torch.Tensor,
    move: int | Sequence[int] | torch.Tensor,
    player: int = BLACK,
) -> int | torch.Tensor:
    """Transform an action to the canonical orientation of ``board``.

    For a batched board, ``move`` must contain one flat action per sample and a
    tensor of transformed flat actions is returned.  The function is intended
    for replay/policy-label preparation, not for a per-tick CUDA kernel.
    """

    if not isinstance(board, torch.Tensor):
        board = torch.as_tensor(board)
    boards = _as_batch(board)
    indices = canonical_symmetry_index(board, player)
    if board.ndim == 2:
        return transform_move(move, int(boards.shape[-1]), int(indices))
    if isinstance(move, torch.Tensor):
        moves = [int(value) for value in move.detach().cpu().reshape(-1)]
    elif isinstance(move, (list, tuple)):
        moves = [int(value) for value in move]
    else:
        raise ValueError("batched canonical_move requires one move per board")
    if len(moves) != boards.shape[0]:
        raise ValueError("move batch length must match board batch size")
    return torch.tensor(
        [transform_move(action, int(boards.shape[-1]), int(index)) for action, index in zip(moves, indices)],
        dtype=torch.int64,
        device=board.device,
    )


def encode_board(
    board: torch.Tensor,
    player: int = BLACK,
    *,
    symmetry: bool = False,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Encode a board as ``[own, opponent, empty]`` planes.

    The output is ``[3, H, H]`` for one board and ``[B, 3, H, H]`` for a
    batch.  ``symmetry=True`` applies the same canonical dihedral transform as
    :func:`canonical_board`.
    """

    signed = canonical_board(board, player, symmetry=symmetry)
    single = signed.ndim == 2
    values = signed.unsqueeze(0) if single else signed
    encoded = torch.stack(
        ((values == BLACK), (values == WHITE), (values == EMPTY)), dim=1
    ).to(dtype=dtype)
    return encoded[0] if single else encoded


def canonical_encoding(board: torch.Tensor, player: int = BLACK) -> torch.Tensor:
    """Symmetry-invariant three-plane encoding convenience wrapper."""

    return encode_board(board, player, symmetry=True)


def random_legal_move(
    board: torch.Tensor,
    generator: torch.Generator | None = None,
) -> int:
    """Select one legal move deterministically from a supplied generator."""

    moves = legal_moves(board)
    if isinstance(moves, list):
        raise ValueError("random_legal_move expects one board, not a batch")
    if moves.numel() == 0:
        raise ValueError("board has no legal moves")
    if generator is None:
        choice = torch.randint(moves.numel(), (1,), device=moves.device)
    else:
        # A CPU generator cannot directly drive a CUDA sampling operation (and
        # vice versa).  Sample the scalar on the generator's device, then use
        # it as a normal Python index; this is one scalar transfer per move and
        # keeps the board tensor untouched.
        generator_device = getattr(generator, "device", torch.device("cpu"))
        choice = torch.randint(
            moves.numel(),
            (1,),
            generator=generator,
            device=generator_device,
        )
    return int(moves[int(choice.item())].item())


class RandomOpponent:
    """Deterministic uniform random policy for a single Gomoku position."""

    def __init__(self, seed: int | None = None, *, device: torch.device | str = "cpu") -> None:
        self.generator = torch.Generator(device=device)
        if seed is not None:
            self.generator.manual_seed(int(seed))

    def choose(self, board: torch.Tensor) -> int:
        return random_legal_move(board, self.generator)

    def __call__(self, board: torch.Tensor) -> int:
        return self.choose(board)


class GomokuEnv:
    """Minimal stateful Gomoku environment suitable for policy benchmarks."""

    def __init__(
        self,
        board_size: int = 9,
        win_length: int = 5,
        *,
        first_player: int = BLACK,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.int8,
    ) -> None:
        self.config = GomokuConfig(board_size, win_length, first_player)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.dtype = dtype
        if dtype not in (torch.int8, torch.int16, torch.int32, torch.int64):
            raise ValueError("board dtype must be an integer torch dtype")
        self.board = empty_board(board_size, device=self.device, dtype=dtype)
        self.current_player = first_player
        self.winner = EMPTY
        self.done = False
        self.move_count = 0
        self.last_move: int | None = None

    @property
    def board_size(self) -> int:
        return self.config.board_size

    @property
    def win_length(self) -> int:
        return self.config.win_length

    def reset(self) -> torch.Tensor:
        self.board.zero_()
        self.current_player = self.config.first_player
        self.winner = EMPTY
        self.done = False
        self.move_count = 0
        self.last_move = None
        return self.board.clone()

    def legal_moves(self) -> torch.Tensor:
        return legal_moves(self.board)

    def legal_move_mask(self) -> torch.Tensor:
        return legal_move_mask(self.board)

    def encode(self, player: int | None = None, *, symmetry: bool = False) -> torch.Tensor:
        perspective = self.current_player if player is None else player
        return encode_board(self.board, perspective, symmetry=symmetry)

    def play(self, move: int | Sequence[int] | torch.Tensor) -> MoveResult:
        if self.done:
            raise ValueError("game is already over")
        row, col = _move_to_row_col(move, self.board_size)
        if int(self.board[row, col]) != EMPTY:
            raise ValueError("move is illegal: intersection is occupied")
        player = self.current_player
        self.board[row, col] = player
        self.move_count += 1
        self.last_move = row * self.board_size + col
        self.winner = int(winner(self.board, self.win_length, (row, col)))
        self.done = self.winner != EMPTY or self.move_count == self.board_size**2
        reward = 1.0 if self.winner == player else 0.0
        if not self.done:
            self.current_player = -player
        return MoveResult(
            move=row * self.board_size + col,
            row=row,
            col=col,
            player=player,
            winner=self.winner,
            done=self.done,
            reward=reward,
            move_count=self.move_count,
        )

    def step(self, move: int | Sequence[int] | torch.Tensor) -> tuple[torch.Tensor, float, bool, dict[str, int | bool]]:
        result = self.play(move)
        info: dict[str, int | bool] = {
            "winner": result.winner,
            "player": result.player,
            "move": result.move,
            "draw": bool(result.done and result.winner == EMPTY),
        }
        return self.board.clone(), result.reward, result.done, info

    def clone(self) -> "GomokuEnv":
        other = GomokuEnv(
            self.board_size,
            self.win_length,
            first_player=self.config.first_player,
            device=self.device,
            dtype=self.dtype,
        )
        other.board.copy_(self.board)
        other.current_player = self.current_player
        other.winner = self.winner
        other.done = self.done
        other.move_count = self.move_count
        other.last_move = self.last_move
        return other


def random_rollout(
    env: GomokuEnv,
    *,
    generator: torch.Generator | None = None,
    max_moves: int | None = None,
    policy: Callable[[torch.Tensor], int] | None = None,
) -> RolloutResult:
    """Play a deterministic random (or supplied) policy until termination.

    The environment is advanced in place.  A policy receives a board tensor and
    must return a flat move index; if omitted, uniform legal moves are sampled
    using ``generator``.  ``max_moves`` is useful for bounded benchmark traces.
    """

    moves: list[int] = []
    limit = env.board_size**2 if max_moves is None else max(0, int(max_moves))
    while not env.done and len(moves) < limit:
        move = random_legal_move(env.board, generator) if policy is None else int(policy(env.board))
        result = env.play(move)
        moves.append(result.move)
    return RolloutResult(tuple(moves), env.winner, bool(env.done and env.winner == EMPTY), env.board.clone())


# Friendly helpers used by small benchmark scripts.
Gomoku = GomokuEnv
canonical_encode = canonical_encoding


def random_game(
    board_size: int | GomokuEnv = 9,
    win_length: int = 5,
    *,
    seed: int | None = None,
    generator: torch.Generator | None = None,
    max_moves: int | None = None,
    device: torch.device | str | None = None,
) -> RolloutResult:
    """Create (or use) an environment and play a reproducible random game.

    Passing an existing :class:`GomokuEnv` advances it in place, matching
    :func:`random_rollout`.  Passing an integer creates a fresh environment;
    ``seed`` initializes a local ``torch.Generator`` and never changes global
    RNG state.
    """

    if isinstance(board_size, GomokuEnv):
        if seed is not None and generator is not None:
            raise ValueError("provide either seed or generator, not both")
        if generator is None:
            generator = torch.Generator(device=board_size.device)
            if seed is not None:
                generator.manual_seed(int(seed))
        return random_rollout(board_size, generator=generator, max_moves=max_moves)
    env = GomokuEnv(int(board_size), int(win_length), device=device)
    if seed is not None and generator is not None:
        raise ValueError("provide either seed or generator, not both")
    if generator is None:
        generator = torch.Generator(device=env.device)
        if seed is not None:
            generator.manual_seed(int(seed))
    return random_rollout(env, generator=generator, max_moves=max_moves)


__all__ = [
    "BLACK",
    "EMPTY",
    "WHITE",
    "Gomoku",
    "GomokuConfig",
    "GomokuEnv",
    "MoveResult",
    "RandomOpponent",
    "RolloutResult",
    "apply_move",
    "canonical_board",
    "canonical_encode",
    "canonical_encoding",
    "canonical_move",
    "canonical_symmetry_index",
    "dihedral_symmetries",
    "empty_board",
    "encode_board",
    "has_won",
    "is_draw",
    "inverse_transform_move",
    "legal_move_mask",
    "legal_moves",
    "random_game",
    "random_legal_move",
    "random_rollout",
    "transform_move",
    "winner",
]
