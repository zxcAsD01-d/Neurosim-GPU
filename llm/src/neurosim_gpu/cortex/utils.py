"""Small shared helpers for the Cortex-LM commands."""

from __future__ import annotations

import torch


def resolve_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def autocast_dtype(name: str) -> torch.dtype | None:
    return {
        "none": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[name]


def format_count(value: int) -> str:
    if value >= 1_000_000:
        return f"{value / 1_000_000:.2f}M"
    if value >= 1_000:
        return f"{value / 1_000:.1f}k"
    return str(value)
