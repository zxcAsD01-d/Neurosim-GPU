"""Replay and episode-boundary primitives for continual embodied learning."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .scheduler import MultiRateScheduler


@dataclass(frozen=True)
class ReplayBatch:
    state: torch.Tensor
    action: torch.Tensor
    reward: torch.Tensor
    next_state: torch.Tensor
    terminal: torch.Tensor


class ReplayBuffer:
    """Fixed-capacity ring buffer with deterministic checkpointable sampling."""

    def __init__(
        self,
        capacity: int,
        *,
        device: torch.device | str = "cpu",
        seed: int = 0,
    ) -> None:
        if isinstance(capacity, bool) or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity = int(capacity)
        self.device = torch.device(device)
        if self.device.type == "cuda" and self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.generator = torch.Generator(device=self.device)
        self.generator.manual_seed(seed)
        self._size = 0
        self._cursor = 0
        self._state: torch.Tensor | None = None
        self._action: torch.Tensor | None = None
        self._reward: torch.Tensor | None = None
        self._next_state: torch.Tensor | None = None
        self._terminal: torch.Tensor | None = None

    @property
    def size(self) -> int:
        return self._size

    def __len__(self) -> int:
        return self._size

    def _initialize(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        terminal: torch.Tensor,
    ) -> None:
        if state.ndim < 1 or action.ndim < 1 or reward.ndim < 1:
            raise ValueError("transition tensors must have a batch dimension")
        if not (
            state.shape[0]
            == action.shape[0]
            == reward.shape[0]
            == next_state.shape[0]
            == terminal.shape[0]
        ):
            raise ValueError("transition batch dimensions must match")
        if state.device != self.device:
            raise ValueError("transition tensors must use buffer device")
        self._state = torch.empty(
            (self.capacity, *state.shape[1:]), dtype=state.dtype, device=self.device
        )
        self._action = torch.empty(
            (self.capacity, *action.shape[1:]), dtype=action.dtype, device=self.device
        )
        self._reward = torch.empty(
            (self.capacity, *reward.shape[1:]), dtype=reward.dtype, device=self.device
        )
        self._next_state = torch.empty_like(self._state)
        self._terminal = torch.empty(
            (self.capacity, *terminal.shape[1:]),
            dtype=terminal.dtype,
            device=self.device,
        )

    def append(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        terminal: torch.Tensor,
    ) -> None:
        tensors = (state, action, reward, next_state, terminal)
        if any(t.device != self.device for t in tensors):
            raise ValueError("all transition tensors must use buffer device")
        if state.shape[0] > self.capacity:
            # Keep the newest capacity transitions; modulo indices with a
            # larger batch would repeat destinations in index_copy_ and make
            # the result order-dependent on CUDA.
            start = state.shape[0] - self.capacity
            state = state[start:]
            action = action[start:]
            reward = reward[start:]
            next_state = next_state[start:]
            terminal = terminal[start:]
        if self._state is None:
            self._initialize(*tensors)
        assert self._state is not None
        if any(t.device != self.device for t in tensors):
            raise ValueError("all transition tensors must use buffer device")
        batch = state.shape[0]
        if batch == 0:
            return
        if tuple(state.shape[1:]) != tuple(self._state.shape[1:]):
            raise ValueError("state shape changed after buffer initialization")
        if tuple(next_state.shape[1:]) != tuple(self._next_state.shape[1:]):
            raise ValueError("next_state shape changed after buffer initialization")
        if tuple(action.shape[1:]) != tuple(self._action.shape[1:]):
            raise ValueError("action shape changed after buffer initialization")
        if tuple(reward.shape[1:]) != tuple(self._reward.shape[1:]):
            raise ValueError("reward shape changed after buffer initialization")
        if tuple(terminal.shape[1:]) != tuple(self._terminal.shape[1:]):
            raise ValueError("terminal shape changed after buffer initialization")

        indices = (torch.arange(batch, device=self.device) + self._cursor) % self.capacity
        self._state.index_copy_(0, indices, state)
        self._action.index_copy_(0, indices, action)
        self._reward.index_copy_(0, indices, reward)
        self._next_state.index_copy_(0, indices, next_state)
        self._terminal.index_copy_(0, indices, terminal)
        self._cursor = (self._cursor + batch) % self.capacity
        self._size = min(self.capacity, self._size + batch)

    def sample(self, batch_size: int) -> ReplayBatch:
        if self._size == 0:
            raise ValueError("cannot sample an empty replay buffer")
        if isinstance(batch_size, bool) or batch_size <= 0:
            raise ValueError("batch_size must be positive")
        indices = torch.randint(
            self._size,
            (batch_size,),
            device=self.device,
            generator=self.generator,
        )
        assert self._state is not None
        return ReplayBatch(
            state=self._state.index_select(0, indices),
            action=self._action.index_select(0, indices),
            reward=self._reward.index_select(0, indices),
            next_state=self._next_state.index_select(0, indices),
            terminal=self._terminal.index_select(0, indices),
        )

    def state_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "capacity": self.capacity,
            "size": self._size,
            "cursor": self._cursor,
            "generator_state": self.generator.get_state().cpu(),
        }
        for name in ("_state", "_action", "_reward", "_next_state", "_terminal"):
            value = getattr(self, name)
            payload[name] = None if value is None else value.detach().cpu().clone()
        return payload

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if int(payload.get("capacity", -1)) != self.capacity:
            raise ValueError("replay capacity differs from checkpoint")
        self._size = int(payload["size"])
        self._cursor = int(payload["cursor"])
        # PyTorch expects RNG state as a CPU ByteTensor even for a CUDA
        # generator; the generator handles device-specific state internally.
        generator_state = payload["generator_state"].to(
            device="cpu", dtype=torch.uint8
        )
        self.generator.set_state(generator_state)
        if payload["_state"] is None:
            self._state = self._action = self._reward = None
            self._next_state = self._terminal = None
            return
        for name in ("_state", "_action", "_reward", "_next_state", "_terminal"):
            setattr(self, name, payload[name].to(self.device).clone())


@dataclass(frozen=True)
class ContinualLearningConfig:
    replay_every: int = 1
    checkpoint_every_episode: int = 0

    def __post_init__(self) -> None:
        if self.replay_every <= 0:
            raise ValueError("replay_every must be positive")
        if self.checkpoint_every_episode < 0:
            raise ValueError("checkpoint_every_episode must be non-negative")


class ContinualLearningController:
    """Own episode counters, replay, and scheduler checkpoints.

    The controller intentionally leaves optimizer/loss choice to the caller.
    ``replay_batch`` supplies data; ``step`` advances the embodied simulator.
    """

    def __init__(
        self,
        scheduler: MultiRateScheduler,
        replay: ReplayBuffer,
        *,
        config: ContinualLearningConfig | None = None,
    ) -> None:
        self.scheduler = scheduler
        self.replay = replay
        self.config = config or ContinualLearningConfig()
        self.episode = 0
        self.environment_steps = 0

    def step(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        state, _ = self.scheduler.step(*args, **kwargs)
        self.environment_steps += 1
        return state

    def observe(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
        reward: torch.Tensor,
        next_state: torch.Tensor,
        terminal: torch.Tensor,
    ) -> None:
        self.replay.append(state, action, reward, next_state, terminal)

    def replay_batch(self, batch_size: int) -> ReplayBatch:
        return self.replay.sample(batch_size)

    @property
    def replay_due(self) -> bool:
        return (
            self.environment_steps > 0
            and self.environment_steps % self.config.replay_every == 0
        )

    def maybe_replay_batch(self, batch_size: int) -> ReplayBatch | None:
        """Return a replay batch only on the configured environment cadence."""

        if not self.replay_due or len(self.replay) == 0:
            return None
        return self.replay.sample(batch_size)

    def end_episode(self) -> int:
        self.episode += 1
        return self.episode

    def state_dict(self) -> dict[str, Any]:
        return {
            "episode": self.episode,
            "environment_steps": self.environment_steps,
            "scheduler": self.scheduler.state_dict(),
            "replay": self.replay.state_dict(),
            "replay_every": self.config.replay_every,
            "checkpoint_every_episode": self.config.checkpoint_every_episode,
        }

    def load_state_dict(self, payload: dict[str, Any]) -> None:
        if payload.get("replay_every") != self.config.replay_every:
            raise ValueError("replay cadence differs from checkpoint")
        if payload.get("checkpoint_every_episode") != self.config.checkpoint_every_episode:
            raise ValueError("episode checkpoint cadence differs from checkpoint")
        self.scheduler.load_state_dict(payload["scheduler"])
        self.replay.load_state_dict(payload["replay"])
        self.episode = int(payload["episode"])
        self.environment_steps = int(payload["environment_steps"])


__all__ = [
    "ContinualLearningConfig",
    "ContinualLearningController",
    "ReplayBatch",
    "ReplayBuffer",
]
