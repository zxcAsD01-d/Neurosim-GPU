"""Shared experiment harness for the neurosim-gpu task-level probes.

Design rules for this directory:

* pure PyTorch + the standard library (the project venv has no numpy/pip);
* every run is deterministic from ``(config, seed)``;
* every run prints exactly one machine-readable line ``RESULT {json}`` so that
  ``run_suite.py`` can aggregate subprocess output without parsing logs;
* offline data is cached under ``data/`` and is never required at import time.
"""

from __future__ import annotations

import gzip
import json
import math
import os
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    # The editable install in this workspace points at another checkout; make
    # every experiment import the sources next to this file instead.
    sys.path.insert(0, str(SRC))

import torch  # noqa: E402

RESULTS = ROOT / "results"
RAW = RESULTS / "raw"
DATA = ROOT / "data"


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def seed_everything(seed: int) -> None:
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(False)


def percentile(values: Sequence[float], probability: float) -> float:
    """Nearest-rank percentile, avoiding a numpy dependency."""

    if not values:
        return float("nan")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    ordered = sorted(float(v) for v in values)
    index = min(len(ordered) - 1, max(0, math.ceil(probability * len(ordered)) - 1))
    return ordered[index]


class Timer:
    def __init__(self) -> None:
        self._start = time.perf_counter()

    @property
    def seconds(self) -> float:
        return time.perf_counter() - self._start

    def reset(self) -> None:
        self._start = time.perf_counter()


def measure_latency(
    fn: Callable[[], Any],
    *,
    warmup: int = 5,
    iterations: int = 25,
) -> dict[str, float]:
    """Wall-clock latency percentiles in milliseconds for one call."""

    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000.0)
    return {
        "p50_ms": percentile(samples, 0.50),
        "p95_ms": percentile(samples, 0.95),
        "p99_ms": percentile(samples, 0.99),
        "mean_ms": sum(samples) / len(samples),
    }


def trainable_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == target).float().mean().item())


def emit_result(payload: dict[str, Any]) -> None:
    """Print the single machine-readable result line consumed by run_suite."""

    print("RESULT " + json.dumps(payload, sort_keys=True, default=_json_default))
    sys.stdout.flush()


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value)
    if isinstance(value, torch.device):
        return str(value)
    if isinstance(value, (set, tuple)):
        return list(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def slug(config: dict[str, Any]) -> str:
    """Filesystem-safe run id: readable prefix plus a content hash.

    Windows refuses ``[ ] , :`` in file names and the 260-character path limit
    is easy to hit with a long config dump, hence the sanitizing and truncation.
    """

    import hashlib
    import re

    digest = hashlib.sha1(
        json.dumps(config, sort_keys=True, default=str).encode("utf-8")
    ).hexdigest()[:10]
    parts = []
    for key in sorted(config):
        value = config[key]
        if isinstance(value, (list, tuple, dict)):
            continue
        if isinstance(value, float):
            text = f"{value:g}".replace(".", "p")
        else:
            text = str(value)
        parts.append(f"{key}={text}")
    readable = re.sub(r"[^0-9A-Za-z=_.-]+", "-", "_".join(parts))[:90]
    return f"{readable}_{digest}"


# ---------------------------------------------------------------------------
# MNIST (offline cache, no torchvision dependency)
# ---------------------------------------------------------------------------

_MNIST_MIRRORS = (
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://raw.githubusercontent.com/fgnt/mnist/master/",
)
_MNIST_FILES = {
    "train_images": ("train-images-idx3-ubyte.gz", 9_912_422),
    "train_labels": ("train-labels-idx1-ubyte.gz", 28_881),
    "test_images": ("t10k-images-idx3-ubyte.gz", 1_648_877),
    "test_labels": ("t10k-labels-idx1-ubyte.gz", 4_541),
}


def _download(url: str, destination: Path, timeout: int = 60) -> None:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        destination.write_bytes(response.read())


def _mnist_path(name: str, root: Path, download: bool) -> Path:
    filename, _ = _MNIST_FILES[name]
    path = root / filename
    if path.exists():
        return path
    if not download:
        raise FileNotFoundError(f"{path} is missing and download=False")
    root.mkdir(parents=True, exist_ok=True)
    errors = []
    for mirror in _MNIST_MIRRORS:
        url = mirror + filename
        try:
            _download(url, path)
            if path.stat().st_size > 0:
                return path
        except Exception as exc:  # pragma: no cover - network dependent
            errors.append(f"{url}: {exc}")
    raise RuntimeError("could not download MNIST:\n" + "\n".join(errors))


def _read_idx_images(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        buffer = bytearray(handle.read())
    if len(buffer) < 16:
        raise ValueError(f"{path} is not a MNIST image file")
    magic = int.from_bytes(buffer[:4], "big")
    if magic != 2051:
        raise ValueError(f"{path} has unexpected magic {magic}")
    count = int.from_bytes(buffer[4:8], "big")
    rows = int.from_bytes(buffer[8:12], "big")
    cols = int.from_bytes(buffer[12:16], "big")
    flat = torch.frombuffer(buffer, dtype=torch.uint8, offset=16)
    expected = count * rows * cols
    if flat.numel() != expected:
        raise ValueError(f"{path} payload has {flat.numel()} != {expected} bytes")
    return flat.view(count, rows * cols).to(torch.float32) / 255.0


def _read_idx_labels(path: Path) -> torch.Tensor:
    with gzip.open(path, "rb") as handle:
        buffer = bytearray(handle.read())
    magic = int.from_bytes(buffer[:4], "big")
    if magic != 2049:
        raise ValueError(f"{path} has unexpected magic {magic}")
    count = int.from_bytes(buffer[4:8], "big")
    flat = torch.frombuffer(buffer, dtype=torch.uint8, offset=8)
    if flat.numel() != count:
        raise ValueError(f"{path} label count mismatch")
    return flat.to(torch.int64)


def load_mnist(
    root: Path | None = None,
    *,
    download: bool = True,
) -> dict[str, torch.Tensor]:
    """Return flattened ``[N, 784]`` float images in [0, 1] and int64 labels."""

    root = root or (DATA / "mnist")
    return {
        "train_images": _read_idx_images(_mnist_path("train_images", root, download)),
        "train_labels": _read_idx_labels(_mnist_path("train_labels", root, download)),
        "test_images": _read_idx_images(_mnist_path("test_images", root, download)),
        "test_labels": _read_idx_labels(_mnist_path("test_labels", root, download)),
    }


def masked_accuracy(
    logits: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> float:
    keep = target >= 0
    if not bool(keep.any()):
        return float("nan")
    return float((logits.argmax(dim=1) == target)[keep].float().mean())


def moving_average(values: Iterable[float], window: int) -> list[float]:
    ordered = list(values)
    out = []
    for index in range(len(ordered)):
        start = max(0, index - window + 1)
        chunk = ordered[start : index + 1]
        out.append(sum(chunk) / len(chunk))
    return out


def cpu_threads() -> int:
    return int(os.environ.get("NEUROSIM_CPU_THREADS", "0")) or torch.get_num_threads()
