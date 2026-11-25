"""GPU-first device selection for the NeuroBody demo.

Training, closed-loop generation and the server all run on CUDA when it is
available.  CPU fallback is explicit (``--device cpu`` / ``--allow-cpu``) so a
broken CUDA setup cannot silently turn into a slow CPU run.
"""
from __future__ import annotations

import os
from typing import Any

import torch


def configure_cuda(device: torch.device) -> None:
    """Apply the GPU knobs that matter for the sparse/GEMM demo path."""

    if device.type != "cuda":
        return
    torch.backends.cudnn.benchmark = True
    if hasattr(torch.backends.cuda.matmul, "allow_tf32"):
        torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = True


def resolve_device(
    requested: str | None = None,
    *,
    require_cuda: bool = True,
) -> torch.device:
    """Return a device, preferring CUDA and refusing a silent CPU fallback.

    ``requested`` may be ``cuda``, ``cuda:1``, ``cpu`` or ``None``.  When it is
    ``None`` the environment variable ``NEUROBODY_DEVICE`` is consulted; if that
    is empty CUDA is selected when present.  ``require_cuda=True`` makes an
    unavailable/misconfigured CUDA setup an error, with ``--allow-cpu`` as the
    explicit escape hatch in the entry points.
    """

    if os.environ.get("NEUROBODY_ALLOW_CPU") == "1":
        require_cuda = False
    requested = (requested or os.environ.get("NEUROBODY_DEVICE") or "").strip()
    if not requested:
        if torch.cuda.is_available():
            requested = "cuda"
        elif require_cuda:
            raise RuntimeError(
                "NeuroBody is GPU-first, but CUDA is not available. "
                "Pass --allow-cpu (and --device cpu) only if you accept the slow path."
            )
        else:
            requested = "cpu"

    device = torch.device(requested)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but torch.cuda.is_available() is False. "
                "Install a CUDA build of PyTorch or pass --allow-cpu."
            )
        index = 0 if device.index is None else int(device.index)
        if index >= torch.cuda.device_count():
            raise RuntimeError(
                f"CUDA device {index} does not exist; {torch.cuda.device_count()} device(s) visible."
            )
        torch.cuda.set_device(index)
        device = torch.device("cuda", index)
        configure_cuda(device)
        return device

    if require_cuda:
        raise RuntimeError(
            "CPU fallback is disabled for the GPU-first demo; "
            "pass --allow-cpu to opt in explicitly."
        )
    return device


def device_info(device: torch.device) -> dict[str, Any]:
    """Serializable device description for ``/api/meta`` and the UI."""

    info: dict[str, Any] = {
        "type": device.type,
        "index": device.index,
        "name": device.type.upper(),
    }
    if device.type != "cuda":
        info["name"] = "CPU fallback"
        return info
    index = torch.cuda.current_device() if device.index is None else int(device.index)
    properties = torch.cuda.get_device_properties(index)
    info.update(
        {
            "index": index,
            "name": torch.cuda.get_device_name(index),
            "capability": f"{properties.major}.{properties.minor}",
            "totalMemory": int(properties.total_memory),
            "multiProcessorCount": int(properties.multi_processor_count),
        }
    )
    return info
