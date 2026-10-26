"""Small, reproducible Gomoku learning comparison.

This benchmark compares a compact CNN policy/value model with a sparse rate
model built from the public ``Dynamics`` and ``SparseTopology`` APIs.  It is a
functional efficiency probe, not a claim that the rate model is biologically
complete.  Both models see exactly the same deterministic generated positions,
use the same AdamW optimizer and loss, and are evaluated with the same timing
protocol.

The default workload is deliberately small (a 9x9 board and a few hundred
positions) so it can run on a workstation CPU or a lightly loaded CUDA device.
Use ``--device cuda`` for a small GPU run; this script never allocates a
large/full-brain topology.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import statistics
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import torch
from torch import nn
from torch.nn import functional as F


# Keep the script runnable directly from a source checkout as well as from an
# editable install.  Existing benchmarks normally rely on the package being
# installed, but this fallback makes the experiment easier to reproduce.
try:
    from neurosim_gpu import (
        Dynamics,
        GomokuEnv,
        SparseTopology,
        encode_board,
        has_won,
        random_topology,
        random_legal_move,
    )
except ModuleNotFoundError:  # pragma: no cover - exercised only in checkout use
    _ROOT = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(_ROOT / "src"))
    from neurosim_gpu import (
        Dynamics,
        GomokuEnv,
        SparseTopology,
        encode_board,
        has_won,
        random_topology,
        random_legal_move,
    )


@dataclass(frozen=True)
class GomokuDataset:
    boards: torch.Tensor
    policy_target: torch.Tensor
    value_target: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.policy_target.numel())


@dataclass(frozen=True)
class TrainResult:
    """Training measurements kept separate from the model summary."""

    train_wall_ms: float
    samples_per_sec: float
    optimizer_state_bytes: int
    train_loss: float
    train_policy_accuracy: float
    train_value_accuracy: float
    train_value_mse: float
    val_loss: float
    val_policy_accuracy: float
    val_value_accuracy: float
    val_value_mse: float
    samples_to_policy_target: int | None
    efficiency_curve: tuple[dict[str, float | int], ...]


@dataclass(frozen=True)
class ModelMetrics:
    name: str
    trainable_params: int
    fixed_params: int
    total_effective_params: int
    fixed_storage_bytes: int
    train_wall_ms: float
    samples_per_sec: float
    optimizer_state_bytes: int
    train_loss: float
    train_policy_accuracy: float
    train_value_accuracy: float
    train_value_mse: float
    val_loss: float
    val_policy_accuracy: float
    val_value_accuracy: float
    val_value_mse: float
    inference_p50_ms: float
    inference_p95_ms: float
    inference_p99_ms: float
    inference_batch: int
    samples_to_policy_target: int | None
    efficiency_curve: tuple[dict[str, float | int], ...]
    model_win_rate_vs_random: float | None
    model_draw_rate_vs_random: float | None
    peak_cuda_mib: float | None


def _segments(board_size: int, win_length: int) -> list[tuple[tuple[int, int], ...]]:
    """Return all contiguous winning segments of exactly ``win_length``."""

    result: list[tuple[tuple[int, int], ...]] = []
    directions = ((1, 0), (0, 1), (1, 1), (1, -1))
    for row in range(board_size):
        for col in range(board_size):
            for drow, dcol in directions:
                cells = tuple(
                    (row + step * drow, col + step * dcol)
                    for step in range(win_length)
                )
                if all(
                    0 <= cell_row < board_size and 0 <= cell_col < board_size
                    for cell_row, cell_col in cells
                ):
                    result.append(cells)
    if not result:
        raise ValueError("board_size must be at least win_length")
    return result


def make_gomoku_dataset(
    samples: int,
    *,
    board_size: int = 9,
    win_length: int = 5,
    noise_stones: int = 8,
    seed: int = 123,
) -> GomokuDataset:
    """Generate deterministic tactical positions.

    Even-indexed samples contain a four-in-a-row for the side to move and the
    target is the missing stone (offensive move).  Odd-indexed samples contain
    the same pattern in the opponent channel and the target is the block
    (defensive move).  Random, non-overlapping distractor stones prevent a
    lookup-table solution while keeping labels unambiguous.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    if board_size < win_length or win_length < 2:
        raise ValueError("board_size must be >= win_length >= 2")
    if noise_stones < 0:
        raise ValueError("noise_stones must be non-negative")

    rng = random.Random(seed)
    segments = _segments(board_size, win_length)
    cells = [(row, col) for row in range(board_size) for col in range(board_size)]
    # Keep the same three-plane contract as ``encode_board``: own stones,
    # opponent stones, and empty intersections.  The explicit empty plane is
    # useful for the brain adapter's sensory projection and makes the dataset
    # directly usable by the environment evaluation below.
    boards = torch.zeros(samples, 3, board_size, board_size, dtype=torch.float32)
    policy = torch.empty(samples, dtype=torch.long)
    value = torch.empty(samples, dtype=torch.float32)

    for sample_id in range(samples):
        offensive = sample_id % 2 == 0
        line = segments[rng.randrange(len(segments))]
        target_index = rng.randrange(win_length)
        target = line[target_index]
        channel = 0 if offensive else 1

        occupied: set[tuple[int, int]] = set()
        for row, col in line:
            if (row, col) != target:
                boards[sample_id, channel, row, col] = 1.0
                occupied.add((row, col))

        candidates = [cell for cell in cells if cell not in occupied and cell != target]
        rng.shuffle(candidates)
        distractors = min(noise_stones, len(candidates))
        for index in range(distractors):
            row, col = candidates[index]
            # Alternate channels to expose both excitatory patterns without
            # changing the target rule.
            distractor_channel = (index + (0 if offensive else 1)) % 2
            boards[sample_id, distractor_channel, row, col] = 1.0
            occupied.add((row, col))

        policy[sample_id] = target[0] * board_size + target[1]
        value[sample_id] = 1.0 if offensive else -1.0

        occupied_mask = boards[sample_id, :2].sum(dim=0) > 0
        boards[sample_id, 2] = (~occupied_mask).to(torch.float32)

    return GomokuDataset(boards, policy, value)


def _teacher_action_and_value(env: GomokuEnv) -> tuple[int, float]:
    """Choose a legal tactical action and a bounded heuristic value.

    The teacher checks an immediate win first, then an immediate opponent
    threat.  If neither exists it prefers central moves with a small local
    occupancy bonus.  All checks happen on the CPU game board and are outside
    the neural timing window.
    """

    legal = [int(move) for move in env.legal_moves().tolist()]
    if not legal:
        raise ValueError("teacher requires at least one legal move")
    board = env.board
    size = env.board_size
    player = env.current_player
    opponent = -player

    def winning_moves(candidate_player: int) -> list[int]:
        result: list[int] = []
        for move in legal:
            row, col = divmod(move, size)
            candidate = board.clone()
            candidate[row, col] = candidate_player
            if bool(has_won(candidate, candidate_player, env.win_length, move)):
                result.append(move)
        return result

    own_wins = winning_moves(player)
    if own_wins:
        candidates = own_wins
        value = 1.0
    else:
        opponent_wins = winning_moves(opponent)
        candidates = opponent_wins if opponent_wins else legal
        value = -1.0 if opponent_wins else 0.0

    center = (size - 1) / 2.0

    def score(move: int) -> tuple[float, int]:
        row, col = divmod(move, size)
        distance = abs(row - center) + abs(col - center)
        local_own = local_opponent = 0
        for drow in (-1, 0, 1):
            for dcol in (-1, 0, 1):
                if drow == 0 and dcol == 0:
                    continue
                near_row, near_col = row + drow, col + dcol
                if 0 <= near_row < size and 0 <= near_col < size:
                    value_at = int(board[near_row, near_col])
                    local_own += value_at == player
                    local_opponent += value_at == opponent
        # The flat-index tie-break keeps generation deterministic across
        # Python versions and devices.
        heuristic = -float(distance) + 0.25 * local_own + 0.10 * local_opponent
        return heuristic, -move

    action = max(candidates, key=score)
    return int(action), value


def make_random_rollout_dataset(
    samples: int,
    *,
    board_size: int = 9,
    win_length: int = 5,
    max_random_moves: int = 20,
    seed: int = 123,
) -> GomokuDataset:
    """Generate legal random-rollout positions with a tactical teacher.

    Each sample starts from an empty :class:`GomokuEnv`, plays a deterministic
    number of uniformly random legal moves, and labels the resulting position
    with :func:`_teacher_action_and_value`. Terminal positions are discarded
    and re-sampled, so every returned policy target is legal. This is a
    distribution-shift probe, not a self-play or Elo benchmark.
    """

    if samples <= 0:
        raise ValueError("samples must be positive")
    if board_size < win_length or win_length < 2:
        raise ValueError("board_size must be >= win_length >= 2")
    if max_random_moves < 0:
        raise ValueError("max_random_moves must be non-negative")

    generator = torch.Generator(device="cpu").manual_seed(seed)
    env = GomokuEnv(board_size, win_length, device="cpu")
    boards = torch.empty(samples, 3, board_size, board_size, dtype=torch.float32)
    policy = torch.empty(samples, dtype=torch.long)
    value = torch.empty(samples, dtype=torch.float32)
    move_limit = min(max_random_moves, board_size * board_size - 1)

    for sample_id in range(samples):
        # A terminal random rollout is rare at the default limit, but bounded
        # retries keep generation valid for larger requested limits.
        for _attempt in range(128):
            env.reset()
            random_steps = int(
                torch.randint(
                    move_limit + 1,
                    (1,),
                    generator=generator,
                    device="cpu",
                ).item()
            )
            for _ in range(random_steps):
                if env.done:
                    break
                env.play(random_legal_move(env.board, generator))
            if not env.done:
                break
        else:
            raise RuntimeError("could not sample a non-terminal random position")

        boards[sample_id].copy_(env.encode())
        action, target_value = _teacher_action_and_value(env)
        policy[sample_id] = action
        value[sample_id] = target_value

    return GomokuDataset(boards, policy, value)


def _masked_logits(logits: torch.Tensor, boards: torch.Tensor) -> torch.Tensor:
    occupied = boards[:, :2].sum(dim=1).flatten(1) > 0
    return logits.masked_fill(occupied, -1.0e4)


def _parameter_count(module: nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)


class CNNPolicyValue(nn.Module):
    """Compact convolutional policy/value baseline."""

    def __init__(self, board_size: int, channels: int) -> None:
        super().__init__()
        self.board_size = board_size
        self.conv1 = nn.Conv2d(3, channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.policy_head = nn.Conv2d(channels, 1, kernel_size=1)
        self.value_head = nn.Linear(channels, 1)

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = F.gelu(self.conv1(boards))
        hidden = F.gelu(self.conv2(hidden))
        logits = self.policy_head(hidden).flatten(1)
        pooled = hidden.mean(dim=(2, 3))
        value = torch.tanh(self.value_head(pooled).squeeze(-1))
        return _masked_logits(logits, boards), value


class SparseRatePolicyValue(nn.Module):
    """Brain-style sparse recurrent adapter.

    The recurrent graph and sensory projection are fixed for this benchmark;
    recurrent edge amplitudes and policy/value readouts are learned.  The
    sensory projection uses a fixed number of outgoing edges per input feature
    so its memory and work do not grow as ``input_features * neurons``.
    """

    def __init__(
        self,
        board_size: int,
        neurons: int,
        degree: int,
        *,
        recurrence_steps: int = 4,
        sensory_fanout: int = 4,
        learn_sensory: bool = False,
        seed: int = 123,
        device: torch.device,
    ) -> None:
        super().__init__()
        if neurons <= 0 or degree <= 0 or recurrence_steps <= 0:
            raise ValueError("neurons, degree and recurrence_steps must be positive")
        if (
            isinstance(sensory_fanout, bool)
            or int(sensory_fanout) != sensory_fanout
            or sensory_fanout <= 0
        ):
            raise ValueError("sensory_fanout must be a positive integer")
        sensory_fanout = int(sensory_fanout)
        self.board_size = board_size
        self.neurons = neurons
        self.recurrence_steps = recurrence_steps
        self.input_features = 3 * board_size * board_size
        self.sensory_fanout = sensory_fanout
        self.learn_sensory = bool(learn_sensory)

        topology = random_topology(
            neurons,
            neurons * degree,
            device=device,
            dtype=torch.float32,
            seed=seed,
            weight_scale=0.18,
        )
        self.register_buffer("in_rowptr", topology.in_rowptr.detach().clone())
        self.register_buffer("in_col", topology.in_col.detach().clone())
        self.register_buffer("src", topology.src.detach().clone())
        self.register_buffer("dst", topology.dst.detach().clone())
        # ``out_weight_index`` maps canonical source-major edge IDs to their
        # destination-CSR positions.  The sparse tensor needs the inverse
        # permutation: CSR position -> canonical edge ID.
        self.register_buffer(
            "in_weight_index",
            torch.argsort(topology.out_weight_index.to(torch.int64)).to(torch.int32),
        )
        self.edge_weight = nn.Parameter(topology.weight.detach().clone())

        generator = torch.Generator(device=device).manual_seed(seed + 17)
        # Each input feature has exactly ``sensory_fanout`` fixed outgoing
        # edges.  The edge list is source-major, which makes the gather and
        # scatter operation deterministic and auditable without materializing
        # a dense input-by-neuron matrix.
        sensory_src = torch.arange(
            self.input_features,
            device=device,
            dtype=torch.int64,
        ).repeat_interleave(self.sensory_fanout)
        sensory_dst = torch.randint(
            neurons,
            (sensory_src.numel(),),
            device=device,
            dtype=torch.int64,
            generator=generator,
        )
        # Match the order of magnitude of a dense N(0, 1/F) projection.  A
        # uniform fanout gives roughly F*k/N terms per destination, hence the
        # sqrt(N/(F*k)) scale keeps the aggregate variance O(1).
        sensory_weight = torch.randn(
            sensory_src.numel(),
            device=device,
            dtype=torch.float32,
            generator=generator,
        ) * math.sqrt(
            float(neurons) / float(self.input_features * self.sensory_fanout)
        )
        self.register_buffer("sensory_src", sensory_src)
        self.register_buffer("sensory_dst", sensory_dst)
        if self.learn_sensory:
            self.sensory_weight = nn.Parameter(sensory_weight)
        else:
            self.register_buffer("sensory_weight", sensory_weight)
        self.register_buffer("alpha", torch.full((neurons,), 0.72, device=device))
        self.register_buffer("gain", torch.full((neurons,), 1.05, device=device))
        self.dynamics = Dynamics(self.alpha, self.gain)

        self.policy_head = nn.Linear(neurons, board_size * board_size)
        self.value_head = nn.Linear(neurons, 1)

    @property
    def edge_count(self) -> int:
        return int(self.edge_weight.numel())

    @property
    def sensory_edge_count(self) -> int:
        return int(self.sensory_weight.numel())

    @property
    def total_edge_count(self) -> int:
        return self.edge_count + self.sensory_edge_count

    @property
    def fixed_param_count(self) -> int:
        """Frozen scalar state used by the rate model (excluding indices)."""

        frozen_sensory = 0 if self.learn_sensory else self.sensory_weight.numel()
        return int(frozen_sensory + self.alpha.numel() + self.gain.numel())

    @property
    def fixed_storage_bytes(self) -> int:
        """Bytes for the frozen sensory fanout and graph metadata.

        Learned edge amplitudes are deliberately excluded here; they are
        already represented by ``trainable_params``.  The value is useful for
        interpreting the parameter-matched comparison on memory-bound GPUs.
        """

        tensors = (
            self.sensory_src,
            self.sensory_dst,
            self.sensory_weight,
            self.alpha,
            self.gain,
            self.in_rowptr,
            self.in_col,
            self.in_weight_index,
            self.src,
            self.dst,
        )
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def _csr(self) -> torch.Tensor:
        values = self.edge_weight[self.in_weight_index.to(torch.int64)]
        return torch.sparse_csr_tensor(
            self.in_rowptr,
            self.in_col,
            values,
            size=(self.neurons, self.neurons),
            device=self.edge_weight.device,
            dtype=self.edge_weight.dtype,
            check_invariants=False,
        )

    def forward(self, boards: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        flat = boards.flatten(1)
        selected = flat.index_select(1, self.sensory_src)
        contributions = selected * self.sensory_weight
        sensory = flat.new_zeros((flat.shape[0], self.neurons))
        sensory.scatter_add_(
            1,
            self.sensory_dst.expand(flat.shape[0], -1),
            contributions,
        )
        state = torch.sigmoid(sensory)
        drive = 0.75 * torch.tanh(sensory)
        weight = self._csr()
        for _ in range(self.recurrence_steps):
            current = torch.sparse.mm(weight, state.T).T
            state = self.dynamics.advance(state, current, drive)
        logits = self.policy_head(state)
        value = torch.tanh(self.value_head(state).squeeze(-1))
        return _masked_logits(logits, boards), value


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _loss_and_metrics(
    logits: torch.Tensor,
    value: torch.Tensor,
    policy_target: torch.Tensor,
    value_target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    policy_loss = F.cross_entropy(logits, policy_target)
    value_loss = F.mse_loss(value, value_target)
    loss = policy_loss + 0.25 * value_loss
    policy_correct = (logits.argmax(dim=1) == policy_target).sum()
    # Rollout positions can have a neutral teacher value (0). Treat a small
    # prediction around zero as the neutral class instead of penalizing every
    # neutral sample as a sign error.
    predicted_sign = torch.where(
        value.abs() < 0.1,
        torch.zeros_like(value),
        value.sign(),
    )
    value_correct = (predicted_sign == value_target.sign()).sum()
    value_squared_error = (value - value_target).square().sum()
    return loss, policy_correct, value_correct, value_squared_error


def _evaluate(
    model: nn.Module,
    dataset: GomokuDataset,
    indices: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> tuple[float, float, float, float]:
    model.eval()
    loss_sum = torch.zeros((), device=device)
    policy_correct = torch.zeros((), dtype=torch.float32, device=device)
    value_correct = torch.zeros((), dtype=torch.float32, device=device)
    value_squared_error = torch.zeros((), dtype=torch.float32, device=device)
    count = 0
    with torch.inference_mode():
        for start in range(0, int(indices.numel()), batch_size):
            batch_indices = indices[start : start + batch_size]
            boards = dataset.boards[batch_indices].to(device, non_blocking=True)
            policy_target = dataset.policy_target[batch_indices].to(device, non_blocking=True)
            value_target = dataset.value_target[batch_indices].to(device, non_blocking=True)
            logits, value = model(boards)
            loss, pcorrect, vcorrect, vmse = _loss_and_metrics(
                logits, value, policy_target, value_target
            )
            batch_count = int(batch_indices.numel())
            loss_sum = loss_sum + loss.detach() * batch_count
            policy_correct = policy_correct + pcorrect.detach().float()
            value_correct = value_correct + vcorrect.detach().float()
            value_squared_error = value_squared_error + vmse.detach().float()
            count += batch_count
    denominator = max(count, 1)
    return (
        float((loss_sum / denominator).cpu()),
        float((policy_correct / denominator).cpu()),
        float((value_correct / denominator).cpu()),
        float((value_squared_error / denominator).cpu()),
    )


def _train(
    model: nn.Module,
    dataset: GomokuDataset,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    seed: int,
    device: torch.device,
) -> TrainResult:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    _sync(device)
    start_time = time.perf_counter()
    training_seconds = 0.0
    curve: list[dict[str, float | int]] = []
    generator = torch.Generator(device="cpu").manual_seed(seed + 101)
    for epoch in range(epochs):
        order = train_indices[torch.randperm(train_indices.numel(), generator=generator)]
        _sync(device)
        epoch_start = time.perf_counter()
        for start in range(0, int(order.numel()), batch_size):
            batch_indices = order[start : start + batch_size]
            boards = dataset.boards[batch_indices].to(device, non_blocking=True)
            policy_target = dataset.policy_target[batch_indices].to(device, non_blocking=True)
            value_target = dataset.value_target[batch_indices].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits, value = model(boards)
            loss, _, _, _ = _loss_and_metrics(
                logits, value, policy_target, value_target
            )
            loss.backward()
            optimizer.step()
        _sync(device)
        training_seconds += time.perf_counter() - epoch_start
        # Evaluation is deliberately outside the timed interval.  This gives a
        # real sample-efficiency curve without charging validation work to the
        # training throughput number.
        train_loss, train_policy, train_value, train_mse = _evaluate(
            model, dataset, train_indices, batch_size, device
        )
        val_loss, val_policy, val_value, val_mse = _evaluate(
            model, dataset, val_indices, batch_size, device
        )
        curve.append(
            {
                "samples_seen": int(train_indices.numel()) * (epoch + 1),
                "train_policy_accuracy": train_policy,
                "val_policy_accuracy": val_policy,
                "train_value_mse": train_mse,
                "val_value_mse": val_mse,
            }
        )
        model.train()
    wall_seconds = training_seconds
    train_samples = int(train_indices.numel()) * epochs
    final_train = _evaluate(model, dataset, train_indices, batch_size, device)
    final_val = _evaluate(model, dataset, val_indices, batch_size, device)
    optimizer_state_bytes = sum(
        value.numel() * value.element_size()
        for state in optimizer.state.values()
        for value in state.values()
        if isinstance(value, torch.Tensor)
    )
    target = getattr(model, "_efficiency_target", 0.5)
    samples_to_target = next(
        (
            int(point["samples_seen"])
            for point in curve
            if float(point["val_policy_accuracy"]) >= target
        ),
        None,
    )
    return TrainResult(
        train_wall_ms=wall_seconds * 1000.0,
        samples_per_sec=train_samples / max(wall_seconds, 1.0e-12),
        optimizer_state_bytes=optimizer_state_bytes,
        train_loss=final_train[0],
        train_policy_accuracy=final_train[1],
        train_value_accuracy=final_train[2],
        train_value_mse=final_train[3],
        val_loss=final_val[0],
        val_policy_accuracy=final_val[1],
        val_value_accuracy=final_val[2],
        val_value_mse=final_val[3],
        samples_to_policy_target=samples_to_target,
        efficiency_curve=tuple(curve),
    )


def _percentile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


@torch.inference_mode()
def _inference_latency(
    model: nn.Module,
    boards: torch.Tensor,
    *,
    warmup: int,
    repeats: int,
    device: torch.device,
) -> tuple[float, float, float]:
    model.eval()
    inputs = boards.to(device, non_blocking=True)
    for _ in range(warmup):
        model(inputs)
    _sync(device)
    timings: list[float] = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        model(inputs)
        _sync(device)
        timings.append((time.perf_counter() - start) * 1000.0)
    return (
        _percentile(timings, 0.50),
        _percentile(timings, 0.95),
        _percentile(timings, 0.99),
    )


def _nearest_cnn_channels(target_params: int, board_size: int) -> int:
    candidates = range(4, 65)
    best = 4
    best_distance = float("inf")
    for channels in candidates:
        model = CNNPolicyValue(board_size, channels)
        distance = abs(_parameter_count(model) - target_params)
        if distance < best_distance:
            best = channels
            best_distance = distance
    return best


@torch.inference_mode()
def _evaluate_vs_random(
    model: nn.Module,
    *,
    board_size: int,
    win_length: int,
    games: int,
    seed: int,
    device: torch.device,
) -> tuple[float | None, float | None]:
    """Measure a trained policy against a fixed-seed uniform opponent.

    This is intentionally a small behavioral probe, not a self-play Elo
    measurement.  Environment rule checks stay on CPU and are excluded from
    neural inference timing.
    """

    if games <= 0:
        return None, None
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    wins = draws = 0
    for game_id in range(games):
        env = GomokuEnv(board_size, win_length)
        model_player = 1 if game_id % 2 == 0 else -1
        while not env.done:
            if env.current_player == model_player:
                encoded = encode_board(
                    env.board,
                    env.current_player,
                    dtype=torch.float32,
                ).unsqueeze(0).to(device)
                logits, _ = model(encoded)
                legal = env.legal_move_mask().reshape(-1).to(device)
                move = int(
                    logits[0].masked_fill(~legal, -1.0e4).argmax().item()
                )
            else:
                move = random_legal_move(env.board, generator)
            env.play(move)
        if env.winner == model_player:
            wins += 1
        elif env.winner == 0:
            draws += 1
    return wins / games, draws / games


def _build_metrics(
    name: str,
    model: nn.Module,
    train_result: TrainResult,
    latency: tuple[float, float, float],
    fixed_params: int,
    fixed_storage_bytes: int,
    inference_batch: int,
    model_win_rate_vs_random: float | None,
    model_draw_rate_vs_random: float | None,
    peak_cuda_mib: float | None,
) -> ModelMetrics:
    trainable = _parameter_count(model)
    return ModelMetrics(
        name=name,
        trainable_params=trainable,
        fixed_params=fixed_params,
        total_effective_params=trainable + fixed_params,
        fixed_storage_bytes=fixed_storage_bytes,
        train_wall_ms=train_result.train_wall_ms,
        samples_per_sec=train_result.samples_per_sec,
        optimizer_state_bytes=train_result.optimizer_state_bytes,
        train_loss=train_result.train_loss,
        train_policy_accuracy=train_result.train_policy_accuracy,
        train_value_accuracy=train_result.train_value_accuracy,
        train_value_mse=train_result.train_value_mse,
        val_loss=train_result.val_loss,
        val_policy_accuracy=train_result.val_policy_accuracy,
        val_value_accuracy=train_result.val_value_accuracy,
        val_value_mse=train_result.val_value_mse,
        inference_p50_ms=latency[0],
        inference_p95_ms=latency[1],
        inference_p99_ms=latency[2],
        inference_batch=inference_batch,
        samples_to_policy_target=train_result.samples_to_policy_target,
        efficiency_curve=train_result.efficiency_curve,
        model_win_rate_vs_random=model_win_rate_vs_random,
        model_draw_rate_vs_random=model_draw_rate_vs_random,
        peak_cuda_mib=peak_cuda_mib,
    )


def _run_model(
    name: str,
    model: nn.Module,
    dataset: GomokuDataset,
    train_indices: torch.Tensor,
    val_indices: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
    fixed_params: int,
    fixed_storage_bytes: int,
) -> ModelMetrics:
    model.to(device)
    # Prime convolution/sparse kernels before the timed optimizer loop.  CUDA
    # module loading and the first sparse-CSR dispatch can otherwise dominate
    # this intentionally small task.
    warmup_batch = dataset.boards[train_indices[: min(args.batch, train_indices.numel())]]
    with torch.inference_mode():
        for _ in range(2 if device.type == "cuda" else 1):
            model(warmup_batch.to(device, non_blocking=True))
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    train_result = _train(
        model,
        dataset,
        train_indices,
        val_indices,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
        weight_decay=args.weight_decay,
        seed=args.seed,
        device=device,
    )
    probe_count = min(args.inference_batch, val_indices.numel())
    probe = dataset.boards[val_indices[:probe_count]]
    latency = _inference_latency(
        model,
        probe,
        warmup=args.inference_warmup,
        repeats=args.inference_repeats,
        device=device,
    )
    peak = None
    if device.type == "cuda":
        _sync(device)
        peak = torch.cuda.max_memory_allocated(device) / 2**20
    win_rate, draw_rate = _evaluate_vs_random(
        model,
        board_size=args.board_size,
        win_length=args.win_length,
        games=args.eval_games,
        seed=args.seed + 4001,
        device=device,
    )
    metrics = _build_metrics(
        name,
        model,
        train_result,
        latency,
        fixed_params,
        fixed_storage_bytes,
        probe_count,
        win_rate,
        draw_rate,
        peak,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    gc.collect()
    return metrics


def _format_table(metrics: list[ModelMetrics]) -> str:
    columns = (
        "model",
        "trainable",
        "opt_KiB",
        "wall_ms",
        "samples/s",
        "val_loss",
        "val_policy",
        "val_value",
        "infer_p50_ms",
        "infer_p95_ms",
        "infer_p99_ms",
        "win_vs_rand",
    )
    lines = [" ".join(f"{column:>14}" for column in columns)]
    for item in metrics:
        lines.append(
            " ".join(
                (
                    f"{item.name:>14}",
                    f"{item.trainable_params:14,d}",
                    f"{item.optimizer_state_bytes / 1024.0:14.1f}",
                    f"{item.train_wall_ms:14.2f}",
                    f"{item.samples_per_sec:14.1f}",
                    f"{item.val_loss:14.5f}",
                    f"{item.val_policy_accuracy:14.4f}",
                    f"{item.val_value_accuracy:14.4f}",
                    f"{item.inference_p50_ms:14.4f}",
                    f"{item.inference_p95_ms:14.4f}",
                    f"{item.inference_p99_ms:14.4f}",
                    (
                        f"{item.model_win_rate_vs_random:14.4f}"
                        if item.model_win_rate_vs_random is not None
                        else f"{'n/a':>14}"
                    ),
                )
            )
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--win-length", type=int, default=5)
    parser.add_argument(
        "--dataset-source",
        choices=("tactical-lines", "random-rollouts"),
        default="tactical-lines",
        help="deterministic tactical templates or legal random-rollout positions",
    )
    parser.add_argument("--samples", type=int, default=512)
    parser.add_argument("--noise-stones", type=int, default=8)
    parser.add_argument(
        "--rollout-max-moves",
        type=int,
        default=20,
        help="maximum random moves before the rollout teacher labels a position",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2.0e-3)
    parser.add_argument("--weight-decay", type=float, default=1.0e-4)
    parser.add_argument("--brain-neurons", type=int, default=24)
    parser.add_argument("--brain-degree", type=int, default=4)
    parser.add_argument("--brain-steps", type=int, default=4)
    parser.add_argument(
        "--brain-sensory-fanout",
        type=int,
        default=4,
        help="fixed sparse outgoing edges per input feature",
    )
    parser.add_argument(
        "--brain-learn-sensory",
        action="store_true",
        help="learn sparse sensory weights; default keeps a fixed reservoir projection",
    )
    parser.add_argument("--cnn-channels", type=int, default=0, help="0 selects a trainable-param-matched width")
    parser.add_argument(
        "--parameter-match",
        choices=("trainable", "effective"),
        default="trainable",
        help="match CNN width to learned scalars or learned+frozen brain scalars",
    )
    parser.add_argument("--inference-warmup", type=int, default=5)
    parser.add_argument("--inference-repeats", type=int, default=30)
    parser.add_argument(
        "--inference-batch",
        type=int,
        default=1,
        help="positions per timed decision batch (1 is per-move latency)",
    )
    parser.add_argument(
        "--target-policy-accuracy",
        type=float,
        default=0.5,
        help="validation accuracy used for samples-to-target",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=0,
        help="optional model-vs-random games; 0 keeps the smoke run short",
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    if args.inference_batch <= 0 or args.inference_repeats <= 0:
        raise SystemExit("inference-batch and inference-repeats must be positive")
    if not 0.0 <= args.target_policy_accuracy <= 1.0:
        raise SystemExit("target-policy-accuracy must lie in [0, 1]")

    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("--device cuda requested but CUDA is unavailable")
    device = torch.device(
        "cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu"
    )
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    if args.samples < 2:
        raise SystemExit("samples must be at least 2 to create a validation split")
    train_count = max(1, int(args.samples * 0.8))
    val_count = args.samples - train_count
    if val_count <= 0:
        raise SystemExit("samples must be at least 2 to create a validation split")

    def build_dataset(count: int, seed: int) -> GomokuDataset:
        if args.dataset_source == "tactical-lines":
            return make_gomoku_dataset(
                count,
                board_size=args.board_size,
                win_length=args.win_length,
                noise_stones=args.noise_stones,
                seed=seed,
            )
        return make_random_rollout_dataset(
            count,
            board_size=args.board_size,
            win_length=args.win_length,
            max_random_moves=args.rollout_max_moves,
            seed=seed,
        )

    # Generate the partitions independently so validation does not contain
    # neighboring states or the same deterministic template instance.
    train_data = build_dataset(train_count, args.seed)
    val_data = build_dataset(val_count, args.seed + 1_000_003)
    dataset = GomokuDataset(
        boards=torch.cat((train_data.boards, val_data.boards), dim=0),
        policy_target=torch.cat((train_data.policy_target, val_data.policy_target), dim=0),
        value_target=torch.cat((train_data.value_target, val_data.value_target), dim=0),
    )
    train_indices = torch.arange(train_count, dtype=torch.long)
    val_indices = torch.arange(train_count, dataset.size, dtype=torch.long)

    # Construct the brain first so the CNN width can be selected to match its
    # trainable parameter count.  Both models are then freshly initialized
    # before their respective training runs.
    brain = SparseRatePolicyValue(
        args.board_size,
        args.brain_neurons,
        args.brain_degree,
        recurrence_steps=args.brain_steps,
        sensory_fanout=args.brain_sensory_fanout,
        learn_sensory=args.brain_learn_sensory,
        seed=args.seed + 7,
        device=device,
    )
    brain_trainable = _parameter_count(brain)
    fixed_brain = brain.fixed_param_count
    fixed_brain_bytes = brain.fixed_storage_bytes
    brain_match_params = (
        brain_trainable + fixed_brain
        if args.parameter_match == "effective"
        else brain_trainable
    )
    cnn_channels = args.cnn_channels or _nearest_cnn_channels(
        brain_match_params, args.board_size
    )
    cnn = CNNPolicyValue(args.board_size, cnn_channels)
    fixed_cnn = 0
    # The target is stored on each freshly constructed model so _train can
    # compute a comparable first-hit sample budget from its validation curve.
    brain._efficiency_target = args.target_policy_accuracy

    # Re-seed immediately before each run so optimizer/data order is the only
    # intentional difference.  Topology and sensory projection stay fixed in
    # the brain adapter as part of its reservoir definition.
    torch.manual_seed(args.seed + 1001)
    cnn = CNNPolicyValue(args.board_size, cnn_channels)
    cnn._efficiency_target = args.target_policy_accuracy
    cnn_metrics = _run_model(
        "cnn",
        cnn,
        dataset,
        train_indices,
        val_indices,
        args,
        device,
        fixed_cnn,
        0,
    )
    brain_metrics = _run_model(
        "sparse-brain",
        brain,
        dataset,
        train_indices,
        val_indices,
        args,
        device,
        fixed_brain,
        fixed_brain_bytes,
    )
    metrics = [cnn_metrics, brain_metrics]

    metadata = {
        "device": str(device),
        "board_size": args.board_size,
        "win_length": args.win_length,
        "dataset_source": args.dataset_source,
        "samples": dataset.size,
        "train_samples": int(train_indices.numel()),
        "validation_samples": int(val_indices.numel()),
        "train_seed": args.seed,
        "validation_seed": args.seed + 1_000_003,
        "epochs": args.epochs,
        "batch": args.batch,
        "optimizer": "AdamW",
        "loss": "cross_entropy(policy) + 0.25*mse(value)",
        "cnn_channels": cnn_channels,
        "parameter_match": args.parameter_match,
        "brain_match_params": brain_match_params,
        "brain_neurons": args.brain_neurons,
        "brain_degree": args.brain_degree,
        "brain_steps": args.brain_steps,
        "brain_sensory_fanout": args.brain_sensory_fanout,
        "brain_recurrent_edges": brain.edge_count,
        "brain_sensory_edges": brain.sensory_edge_count,
        "brain_total_edges": brain.total_edge_count,
        "inference_batch": args.inference_batch,
        "target_policy_accuracy": args.target_policy_accuracy,
        "eval_games": args.eval_games,
        "rollout_max_moves": args.rollout_max_moves,
        "brain_fixed_storage_bytes": fixed_brain_bytes,
        "brain_learn_sensory": args.brain_learn_sensory,
        "note": "Supervised tactical teacher task (template or legal random-rollout positions); sparse-brain is a functional rate-model BPTT upper bound, not a biological equivalence claim or self-play Elo result.",
    }
    if args.json:
        print(json.dumps({"config": metadata, "results": [asdict(item) for item in metrics]}, indent=2))
    else:
        print(
            f"device={device} board={args.board_size}x{args.board_size} "
            f"samples={dataset.size} train={train_indices.numel()} val={val_indices.numel()}"
        )
        print(
            f"cnn_channels={cnn_channels} brain_neurons={args.brain_neurons} "
            f"brain_edges={brain.total_edge_count} fixed_brain_params={fixed_brain:,} "
            f"fixed_brain_bytes={fixed_brain_bytes:,} "
            f"sensory_fanout={args.brain_sensory_fanout} "
            f"parameter_match={args.parameter_match}"
        )
        print(_format_table(metrics))
        print(
            "parameter_ratio(trainable brain/cnn)="
            f"{brain_metrics.trainable_params / max(cnn_metrics.trainable_params, 1):.4f}"
        )
        print(
            "efficiency_ratio(samples_per_sec brain/cnn)="
            f"{brain_metrics.samples_per_sec / max(cnn_metrics.samples_per_sec, 1.0e-12):.4f}"
        )
        print(
            "effective_parameter_ratio(brain/cnn)="
            f"{brain_metrics.total_effective_params / max(cnn_metrics.total_effective_params, 1):.4f}"
        )


if __name__ == "__main__":
    main()
