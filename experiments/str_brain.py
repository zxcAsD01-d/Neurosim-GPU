"""Pure spiking-free "brain" on scene text recognition (Union14M-L).

Architecture (no convolution anywhere, engine untouched):

    image -> retina (fixed positions, brightness only, NOT fixed wiring)
          -> learnable sparse projection (retina -> neurons)
          -> sparse recurrent core (Dynamics, rate neurons)
          -> linear readout per scan step
          -> CTC

The retina is a topographic identity map: rod cell ``(row, col)`` receives the
luminance of pixel ``(row, col)``.  It holds no learnable parameters and no
feature engineering; every connection after it is plastic and learned.

The scan slides a window of ``scan_win`` columns with ``scan_stride`` across the
image; the recurrent state persists across steps, so the core has to integrate
context over the whole word.

Usage::

    python experiments/str_brain.py --config-json '{"epochs":10,"train_limit":20000}'
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import random
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image

from common import DATA, Timer, emit_result, seed_everything, trainable_parameters

from neurosim_gpu import Dynamics, random_topology

STR_DIR = DATA / "union14m"

DEFAULTS: dict[str, Any] = {
    "img_h": 32,
    "img_w": 128,
    "scan_win": 10,
    "scan_stride": 5,
    "neurons": 512,
    "fanout": 4,
    "degree": 8,
    "inner_steps": 2,
    "alpha": 0.72,
    "gain": 1.05,
    "weight_scale": 0.18,
    "sensory_gain": 2.0,
    "learn_sensory": True,
    "dropout": 0.1,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "epochs": 10,
    "batch": 32,
    "train_limit": 20000,
    "charset_limit": 64,
    "warmup_frac": 0.05,
    "label_smoothing": 0.0,
    "max_text_len": 25,
    "blank_bias": -6.0,
    "seed": 0,
    "device": "auto",
}

BENCH_SUBSETS = [
    "benchmark_curve",
    "benchmark_multi_oriented",
    "benchmark_artistic",
    "benchmark_contextless",
    "benchmark_salient",
    "benchmark_multi_words",
    "benchmark_incomplete",
    "benchmark_incomplete_ori",
]


# ---------------------------------------------------------------------------
# Charset
# ---------------------------------------------------------------------------


class Charset:
    """Blank at index 0, then the characters observed in training text."""

    def __init__(self, chars: list[str]) -> None:
        self.chars = ["<blank>"] + chars
        self.index = {c: i for i, c in enumerate(self.chars)}

    def __len__(self) -> int:
        return len(self.chars)

    def encode(self, text: str) -> list[int] | None:
        out = []
        for ch in text:
            idx = self.index.get(ch)
            if idx is None:
                return None
            out.append(idx)
        return out

    @classmethod
    def from_texts(cls, texts: list[str], limit: int = 64) -> "Charset":
        counter: dict[str, int] = {}
        for t in texts:
            for ch in t:
                counter[ch] = counter.get(ch, 0) + 1
        ordered = sorted(counter, key=lambda c: (-counter[c], c))[: max(1, limit - 1)]
        return cls(sorted(ordered))

    def to_json(self) -> dict[str, Any]:
        return {"chars": self.chars}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def read_parquet_rows(path: Path, limit: int | None = None) -> list[tuple[bytes, str]]:
    import pandas as pd

    df = pd.read_parquet(path, columns=["image", "text"])
    if limit is not None:
        df = df.head(limit)
    rows: list[tuple[bytes, str]] = []
    for rec in df.itertuples(index=False):
        payload = rec.image
        data = payload["bytes"] if isinstance(payload, dict) else payload
        rows.append((data, str(rec.text)))
    return rows


def decode_image(raw: bytes, height: int, width: int) -> torch.Tensor:
    img = Image.open(io.BytesIO(raw)).convert("L").resize((width, height), Image.BILINEAR)
    return torch.frombuffer(bytearray(img.tobytes()), dtype=torch.uint8).view(height, width)


def build_cache(
    rows: list[tuple[bytes, str]],
    charset: Charset,
    height: int,
    width: int,
    *,
    max_text_len: int,
) -> tuple[torch.Tensor, list[list[int]]]:
    images: list[torch.Tensor] = []
    targets: list[list[int]] = []
    for raw, text in rows:
        if not text or len(text) > max_text_len:
            continue
        encoded = charset.encode(text)
        if encoded is None:
            continue
        images.append(decode_image(raw, height, width))
        targets.append(encoded)
    if not images:
        return torch.zeros((0, height, width), dtype=torch.uint8), []
    return torch.stack(images), targets


def load_split(name: str, limit: int | None) -> list[tuple[bytes, str]]:
    path = STR_DIR / f"{name}-00000-of-00001.parquet"
    if not path.exists():
        return []
    return read_parquet_rows(path, limit)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------


class SparseProjection(nn.Module):
    """Learnable sparse fanout projection with a fixed wiring pattern."""

    def __init__(self, features: int, neurons: int, fanout: int, *, seed: int, gain: float) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed + 991)
        src = torch.arange(features, dtype=torch.int64).repeat_interleave(fanout)
        dst = torch.randint(neurons, (src.numel(),), generator=generator, dtype=torch.int64)
        scale = math.sqrt(float(neurons) / float(features * fanout))
        weight = torch.randn(src.numel(), generator=generator) * scale
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("gain", torch.tensor(float(gain)))
        self.weight = nn.Parameter(weight)
        self.neurons = neurons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        contributions = x.index_select(1, self.src) * self.weight
        out = x.new_zeros((x.shape[0], self.neurons))
        out.scatter_add_(1, self.dst.unsqueeze(0).expand(x.shape[0], -1), contributions)
        return self.gain * out


class RetinaBrain(nn.Module):
    """Topographic retina + learnable sparse projection + sparse recurrent core."""

    def __init__(self, config: dict[str, Any], seed: int, classes: int) -> None:
        super().__init__()
        self.height = int(config["img_h"])
        self.width = int(config["img_w"])
        self.win = int(config["scan_win"])
        self.stride = int(config["scan_stride"])
        self.steps = (self.width - self.win) // self.stride + 1
        self.neurons = int(config["neurons"])
        self.inner_steps = int(config["inner_steps"])
        self.retina_cells = self.height * self.win

        neurons = self.neurons
        self.register_buffer("alpha_vec", torch.full((neurons,), float(config["alpha"])))
        self.register_buffer("gain_vec", torch.full((neurons,), float(config["gain"])))
        self.dynamics = Dynamics(
            torch.full((neurons,), float(config["alpha"])),
            torch.full((neurons,), float(config["gain"])),
        )

        topology = random_topology(
            neurons,
            neurons * int(config["degree"]),
            device="cpu",
            seed=seed,
            weight_scale=float(config["weight_scale"]),
        )
        self.register_buffer("rec_src", topology.src.detach().clone())
        self.register_buffer("rec_dst", topology.dst.detach().clone())
        self.edge_weight = nn.Parameter(topology.weight.detach().clone())

        self.sensory = SparseProjection(
            self.retina_cells,
            neurons,
            int(config["fanout"]),
            seed=seed,
            gain=float(config["sensory_gain"]),
        )
        self.drop = nn.Dropout(float(config["dropout"]))
        self.readout = nn.Linear(neurons, classes)

    @property
    def edge_count(self) -> int:
        return int(self.edge_weight.numel())

    def _apply(self, fn, recurse: bool = True):
        super()._apply(fn, recurse)
        self.dynamics = Dynamics(self.alpha_vec, self.gain_vec)
        return self

    def _recurrent(self, state: torch.Tensor) -> torch.Tensor:
        contributions = state.index_select(1, self.rec_src) * self.edge_weight
        out = torch.zeros_like(state)
        out.scatter_add_(
            1, self.rec_dst.unsqueeze(0).expand(state.shape[0], -1), contributions
        )
        return out

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """images: [B, H, W] in [0, 1]. Returns log-probabilities [B, T, classes]."""

        batch = images.shape[0]
        state = None
        outputs = []
        for step in range(self.steps):
            start = step * self.stride
            patch = images[:, :, start : start + self.win].reshape(batch, -1)
            sensory = self.sensory(patch)
            # The retina drive enters only through ``drive``; the state itself is
            # carried by the recurrent core, never overwritten by the input.
            if state is None:
                state = torch.sigmoid(sensory)
            drive = 0.75 * torch.tanh(sensory)
            for _ in range(self.inner_steps):
                state = self.dynamics.advance(state, self._recurrent(state), drive)
            outputs.append(self.readout(self.drop(state)))
        return F.log_softmax(torch.stack(outputs, dim=1), dim=-1)


# ---------------------------------------------------------------------------
# CTC helpers
# ---------------------------------------------------------------------------


def levenshtein(a: list[int], b: list[int]) -> int:
    if not a:
        return len(b)
    if not b:
        return len(a)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def greedy_decode(log_probs: torch.Tensor, charset: Charset) -> list[str]:
    best = log_probs.argmax(dim=-1)  # [B, T]
    out = []
    for row in best:
        chars = []
        prev = -1
        for idx in row.tolist():
            if idx != prev and idx != 0:
                chars.append(charset.chars[idx])
            prev = idx
        out.append("".join(chars))
    return out


def edit_distance_str(a: str, b: str) -> int:
    return levenshtein(list(a), list(b))


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


def train_model(model, images, targets, charset, config, device, seed):
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["lr"]), weight_decay=float(config["weight_decay"])
    )
    epochs = int(config["epochs"])
    batch = int(config["batch"])
    valid = [i for i, t in enumerate(targets) if t]
    n = len(valid)
    steps_per_epoch = math.ceil(n / batch)
    total = max(1, epochs * steps_per_epoch)
    warmup = max(1, int(float(config["warmup_frac"]) * total))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        p = (step - warmup) / max(1, total - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    generator = torch.Generator(device="cpu").manual_seed(seed + 777)
    timer = Timer()
    history = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator).tolist()
        total_loss = 0.0
        seen = 0
        for start in range(0, n, batch):
            idx = [valid[j] for j in order[start : start + batch]]
            x = images.index_select(0, torch.tensor(idx)).float().div_(255.0).to(device)
            flat: list[int] = []
            lengths: list[int] = []
            for i in idx:
                if targets[i]:
                    flat.extend(targets[i])
                    lengths.append(len(targets[i]))
            if not lengths:
                continue
            log_probs = model(x)
            input_lengths = torch.full(
                (len(lengths),), log_probs.shape[1], dtype=torch.long, device=device
            )
            loss = F.ctc_loss(
                log_probs.transpose(0, 1),
                torch.tensor(flat, dtype=torch.long, device=device),
                input_lengths,
                torch.tensor(lengths, dtype=torch.long, device=device),
                blank=0,
                zero_infinity=True,
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            total_loss += float(loss.detach().item()) * len(lengths)
            seen += len(lengths)
        history.append({"epoch": epoch + 1, "loss": total_loss / max(seen, 1), "seconds": timer.seconds})
        print(f"  epoch {epoch+1}/{epochs} ctc_loss={history[-1]['loss']:.4f} t={timer.seconds:.0f}s", flush=True)
    return {"history": history, "seconds": timer.seconds}


@torch.no_grad()
def evaluate(model, images, targets, charset, device, batch: int = 64) -> dict[str, float]:
    model.eval()
    correct_words = 0
    char_errors = 0
    char_total = 0
    n = 0
    for start in range(0, images.shape[0], batch):
        chunk = images[start : start + batch]
        x = chunk.float().div_(255.0).to(device)
        log_probs = model(x)
        preds = greedy_decode(log_probs, charset)
        for pred, gold_ids in zip(preds, targets[start : start + batch]):
            gold = "".join(charset.chars[i] for i in gold_ids)
            n += 1
            if pred == gold:
                correct_words += 1
            char_errors += edit_distance_str(pred, gold)
            char_total += max(len(gold), 1)
    return {
        "samples": n,
        "word_accuracy": correct_words / max(n, 1),
        "char_accuracy": max(0.0, 1.0 - char_errors / max(char_total, 1)),
        "wer": char_errors / max(char_total, 1),
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--no-bench", action="store_true", help="skip benchmark subsets")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = dict(DEFAULTS)
    if args.config_json:
        config.update(json.loads(args.config_json))
    if args.smoke:
        config.update({"epochs": 1, "train_limit": 512, "batch": 16})

    requested = str(config.get("device", "auto"))
    device = torch.device("cuda" if requested == "auto" and torch.cuda.is_available() else requested)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    seed_everything(args.seed)

    train_rows = read_parquet_rows(STR_DIR / "train-00000-of-00013.parquet", int(config["train_limit"]))
    charset = Charset.from_texts([t for _, t in train_rows], int(config["charset_limit"]))
    print(f"[data] train rows={len(train_rows)} charset={len(charset)} device={device}", flush=True)

    train_images, train_targets = build_cache(
        train_rows, charset, int(config["img_h"]), int(config["img_w"]),
        max_text_len=int(config["max_text_len"]),
    )
    print(f"[data] usable={train_images.shape[0]} steps={(config['img_w']-config['scan_win'])//config['scan_stride']+1}", flush=True)

    model = RetinaBrain(config, args.seed, len(charset)).to(device)
    with torch.no_grad():
        # CTC otherwise collapses onto "emit blank everywhere"; a negative blank
        # bias at init is the standard escape and is what lets the core align.
        model.readout.bias[0] = float(config["blank_bias"])
    result = train_model(model, train_images, train_targets, charset, config, device, args.seed)
    train_metrics = evaluate(model, train_images[:2000], train_targets[:2000], charset, device)

    per_subset = {}
    for name in BENCH_SUBSETS:
        if args.no_bench:
            break
        rows = load_split(name, None)
        if not rows:
            continue
        imgs, targs = build_cache(
            rows, charset, int(config["img_h"]), int(config["img_w"]),
            max_text_len=int(config["max_text_len"]),
        )
        if imgs.shape[0] == 0:
            continue
        per_subset[name] = evaluate(model, imgs, targs, charset, device)
        print(f"  [{name}] word_acc={per_subset[name]['word_accuracy']:.4f} "
              f"char_acc={per_subset[name]['char_accuracy']:.4f} n={per_subset[name]['samples']}", flush=True)

    payload = {
        "experiment": "str_brain_union14m",
        "config": config,
        "seed": args.seed,
        "device": str(device),
        "charset_size": len(charset),
        "trainable_params": trainable_parameters(model),
        "edges": model.edge_count,
        "train_seconds": result["seconds"],
        "history": result["history"],
        "train_metrics": train_metrics,
        "benchmark": per_subset,
    }
    if args.json:
        emit_result(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
