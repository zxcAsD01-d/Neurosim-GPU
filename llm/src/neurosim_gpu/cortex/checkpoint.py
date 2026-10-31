"""Checkpoint format for Cortex-LM.

Only trainable tensors are stored. Frozen backbone weights are rebuilt from the
model id (or the tiny config), and the cortical graph is rebuilt from its seed,
so a checkpoint stays small enough to copy around. The checkpoint records the
full experiment config and refuses to load into a differently shaped model.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import torch

from .config import ExperimentConfig
from .model import CortexSandwichLM

FORMAT = "cortex-lm/1"


def save_checkpoint(
    path: str | Path,
    model: CortexSandwichLM,
    optimizer: torch.optim.Optimizer | None,
    scheduler: Any,
    step: int,
    config: ExperimentConfig,
    *,
    metrics: dict[str, float] | None = None,
    best_val: float | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": FORMAT,
        "step": int(step),
        "config": config.to_dict(),
        "model": model.trainable_state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "metrics": metrics or {},
        "best_val": best_val,
        "rng": {
            "torch": torch.get_rng_state(),
            "python": random.getstate(),
            "cuda": (
                torch.cuda.get_rng_state_all()
                if torch.cuda.is_available()
                else None
            ),
        },
    }
    temporary = path.with_name(path.name + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(
    path: str | Path,
    model: CortexSandwichLM,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler: Any | None = None,
    *,
    map_location: str | torch.device = "cpu",
    restore_rng: bool = False,
) -> dict[str, Any]:
    payload = torch.load(
        Path(path), map_location=map_location, weights_only=False
    )
    if payload.get("format") != FORMAT:
        raise ValueError(
            f"unsupported checkpoint format: {payload.get('format')!r}"
        )
    model.load_trainable_state_dict(payload["model"])
    if optimizer is not None and payload.get("optimizer") is not None:
        optimizer.load_state_dict(payload["optimizer"])
    if scheduler is not None and payload.get("scheduler") is not None:
        scheduler.load_state_dict(payload["scheduler"])
    if restore_rng and payload.get("rng"):
        torch.set_rng_state(payload["rng"]["torch"].cpu())
        random.setstate(payload["rng"]["python"])
        cuda_state = payload["rng"].get("cuda")
        if cuda_state is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in cuda_state]
            )
    return payload


def checkpoint_config(payload: dict[str, Any]) -> ExperimentConfig:
    return ExperimentConfig.from_dict(payload["config"])
