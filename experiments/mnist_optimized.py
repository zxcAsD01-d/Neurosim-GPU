"""Optimized MNIST training for the sparse rate brain (experiment layer only).

This file deliberately does **not** touch ``src/neurosim_gpu`` (the simulation
engine).  It only changes what the experiment trains and how:

* ``sensory_kind``: ``random`` (the historical fixed sparse projection) or
  ``conv`` (a local, learnable convolutional front-end feeding the same sparse
  recurrent core);
* ``readout_mode``: ``last`` (historical), ``mean`` or ``concat`` over ticks;
* cosine LR schedule with warmup, label smoothing, dropout, affine augmentation;
* defaults to the full 60k train / 10k test split instead of 10k / 2k;
* optional multi-seed ensemble.

Run examples::

    python experiments/mnist_optimized.py --config-json '{"sensory_kind":"conv","epochs":20}'
    python experiments/mnist_optimized.py --config-json '{"sensory_kind":"conv"}' --ens 5
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import DATA, Timer, emit_result, load_mnist, seed_everything, trainable_parameters

from neurosim_gpu import Dynamics, random_topology

DEFAULTS: dict[str, Any] = {
    "sensory_kind": "random",
    "neurons": 256,
    "degree": 8,
    "steps": 8,
    "fanout": 8,
    "conv_width": 16,
    "sensory_gain": 2.0,
    "learn_sensory": True,
    "readout_mode": "mean",
    "dropout": 0.1,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "epochs": 20,
    "batch": 128,
    "train_size": 60000,
    "test_size": 10000,
    "alpha": 0.72,
    "gain": 1.05,
    "weight_scale": 0.18,
    "warmup_frac": 0.1,
    "label_smoothing": 0.1,
    "augment": True,
    "subsample_seed": 0,
    "propagation": "index",
    "device": "auto",
}


# ---------------------------------------------------------------------------
# Sensory front-ends
# ---------------------------------------------------------------------------


class SparseProjection(nn.Module):
    """Fixed fanout projection ``784 -> neurons``; weights optionally learned."""

    def __init__(
        self,
        features: int,
        neurons: int,
        fanout: int,
        *,
        seed: int,
        learn: bool,
        gain: float = 1.0,
    ) -> None:
        super().__init__()
        generator = torch.Generator(device="cpu").manual_seed(seed + 991)
        src = torch.arange(features, dtype=torch.int64).repeat_interleave(fanout)
        dst = torch.randint(neurons, (src.numel(),), generator=generator, dtype=torch.int64)
        scale = math.sqrt(float(neurons) / float(features * fanout))
        weight = torch.randn(src.numel(), generator=generator) * scale
        self.register_buffer("src", src)
        self.register_buffer("dst", dst)
        self.register_buffer("gain", torch.tensor(float(gain)))
        if learn:
            self.weight = nn.Parameter(weight)
        else:
            self.register_buffer("weight", weight)
        self.features = features
        self.neurons = neurons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected = x.index_select(1, self.src)
        contributions = selected * self.weight
        out = x.new_zeros((x.shape[0], self.neurons))
        out.scatter_add_(1, self.dst.expand(x.shape[0], -1), contributions)
        return self.gain * out


class ConvSensory(nn.Module):
    """Local learnable front-end: 2 conv blocks, global average over space."""

    def __init__(self, neurons: int, width: int = 16, gain: float = 1.0) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, width, 5, padding=2, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width, width * 2, 5, padding=2, bias=False),
            nn.BatchNorm2d(width * 2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(width * 2, neurons, 3, padding=1, bias=False),
            nn.BatchNorm2d(neurons),
        )
        self.gain = gain
        self.neurons = neurons

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.body(x.view(-1, 1, 28, 28))
        return self.gain * h.mean(dim=(2, 3))


# ---------------------------------------------------------------------------
# Model: sparse recurrent core + front-end + readout
# ---------------------------------------------------------------------------


class SparseBrainV2(nn.Module):
    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__()
        neurons = int(config["neurons"])
        self.neurons = neurons
        self.steps = int(config["steps"])
        self.readout_mode = str(config["readout_mode"])
        self.propagation = str(config["propagation"])
        # Dynamics tensors are plain attributes on a frozen dataclass, so they
        # have to be registered as buffers to follow ``.to(device)``.
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
        self.register_buffer("src", topology.src.detach().clone())
        self.register_buffer("dst", topology.dst.detach().clone())
        self.register_buffer("in_rowptr", topology.in_rowptr.detach().clone())
        self.register_buffer("in_col", topology.in_col.detach().clone())
        self.register_buffer(
            "in_weight_index",
            torch.argsort(topology.out_weight_index.to(torch.int64)).to(torch.int32),
        )
        self.edge_weight = nn.Parameter(topology.weight.detach().clone())

        if config["sensory_kind"] == "conv":
            self.sensory = ConvSensory(
                neurons, width=int(config["conv_width"]), gain=float(config["sensory_gain"])
            )
        else:
            self.sensory = SparseProjection(
                784,
                neurons,
                int(config["fanout"]),
                seed=seed,
                learn=bool(config["learn_sensory"]),
                gain=float(config["sensory_gain"]),
            )

        width = neurons * (self.steps if self.readout_mode == "concat" else 1)
        self.drop = nn.Dropout(float(config["dropout"]))
        self.readout = nn.Linear(width, 10)

    @property
    def edge_count(self) -> int:
        return int(self.edge_weight.numel())

    def _apply(self, fn, recurse: bool = True):
        # ``Dynamics`` holds plain tensors, so rebuild it after every device or
        # dtype move instead of paying its validation sync per forward pass.
        super()._apply(fn, recurse)
        self.dynamics = Dynamics(self.alpha_vec, self.gain_vec)
        return self

    def _csr(self) -> torch.Tensor:
        values = self.edge_weight[self.in_weight_index.to(torch.int64)]
        return torch.sparse_csr_tensor(
            self.in_rowptr,
            self.in_col,
            values,
            size=(self.neurons, self.neurons),
            check_invariants=False,
        )

    def _propagate(self, state: torch.Tensor) -> torch.Tensor:
        """Sum incoming edge contributions: mathematically the same sparse W @ s."""

        if self.propagation == "csr":
            return torch.sparse.mm(self._csr(), state.T).T
        contributions = state.index_select(1, self.src) * self.edge_weight
        out = torch.zeros_like(state)
        out.scatter_add_(
            1, self.dst.unsqueeze(0).expand(state.shape[0], -1), contributions
        )
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sensory = self.sensory(x)
        state = torch.sigmoid(sensory)
        drive = 0.75 * torch.tanh(sensory)
        dynamics = self.dynamics
        states: list[torch.Tensor] = []
        for _ in range(self.steps):
            current = self._propagate(state)
            state = dynamics.advance(state, current, drive)
            states.append(state)
        if not states:
            feat = state
        elif self.readout_mode == "concat":
            feat = torch.cat(states, dim=1)
        elif self.readout_mode == "mean":
            feat = torch.stack(states, dim=0).mean(dim=0)
        else:
            feat = states[-1]
        return self.readout(self.drop(feat))


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------


def subset(images: torch.Tensor, labels: torch.Tensor, size: int, seed: int):
    size = min(size, images.shape[0])
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(images.shape[0], generator=generator)[:size]
    return images.index_select(0, index), labels.index_select(0, index)


def augment_batch(x: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Small affine jitter: rotation, scale and translation."""

    batch = x.shape[0]
    device = x.device
    angle = (torch.rand(batch, generator=generator, device=device) * 2 - 1) * 0.15
    scale = 1.0 + (torch.rand(batch, generator=generator, device=device) * 2 - 1) * 0.10
    tx = (torch.rand(batch, generator=generator, device=device) * 2 - 1) * 0.08
    ty = (torch.rand(batch, generator=generator, device=device) * 2 - 1) * 0.08
    cos = torch.cos(angle) / scale
    sin = torch.sin(angle) / scale
    theta = torch.stack([cos, -sin, tx, sin, cos, ty], dim=1).view(batch, 2, 3)
    images = x.view(batch, 1, 28, 28)
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    out = F.grid_sample(images, grid, align_corners=False, padding_mode="zeros")
    return out.reshape(batch, -1)


# ---------------------------------------------------------------------------
# Train / evaluate
# ---------------------------------------------------------------------------


def train_model(
    model: nn.Module,
    tx: torch.Tensor,
    ty: torch.Tensor,
    config: dict[str, Any],
    seed: int,
    device: torch.device | str = "cpu",
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    batch = int(config["batch"])
    n = tx.shape[0]
    steps_per_epoch = math.ceil(n / batch)
    total_steps = max(1, epochs * steps_per_epoch)
    warmup = max(1, int(float(config["warmup_frac"]) * total_steps))

    def lr_lambda(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        progress = (step - warmup) / max(1, total_steps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    generator = torch.Generator(device=device).manual_seed(seed + 12345)
    timer = Timer()
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(n, generator=generator, device=device)
        loss_sum = torch.zeros((), device=device)
        correct_t = torch.zeros((), device=device, dtype=torch.long)
        seen = 0
        for start in range(0, n, batch):
            index = order[start : start + batch]
            x = tx.index_select(0, index)
            y = ty.index_select(0, index)
            if config["augment"]:
                x = augment_batch(x, generator)
            logits = model(x)
            loss = F.cross_entropy(
                logits, y, label_smoothing=float(config["label_smoothing"])
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            scheduler.step()
            # Keep accumulation on-device; one sync per epoch, not per batch.
            loss_sum = loss_sum + loss.detach() * y.numel()
            correct_t = correct_t + (logits.argmax(dim=1) == y).sum()
            seen += y.numel()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss_sum.item()) / max(seen, 1),
                "train_acc": float(correct_t.item()) / max(seen, 1),
                "seconds": timer.seconds,
            }
        )
    return {"history": history, "seconds": timer.seconds}


@torch.no_grad()
def predict(model: nn.Module, images: torch.Tensor, batch: int = 500) -> torch.Tensor:
    model.eval()
    chunks = []
    for start in range(0, images.shape[0], batch):
        chunks.append(model(images[start : start + batch]))
    return torch.cat(chunks, dim=0)


def _shift_view(x: torch.Tensor, dx: int, dy: int) -> torch.Tensor:
    """Translate a flattened batch by whole pixels, zero-filling the border."""

    batch = x.shape[0]
    img = x.view(batch, 1, 28, 28)
    out = torch.zeros_like(img)
    src_y = slice(max(0, -dy), 28 - max(0, dy))
    dst_y = slice(max(0, dy), 28 - max(0, -dy))
    src_x = slice(max(0, -dx), 28 - max(0, dx))
    dst_x = slice(max(0, dx), 28 - max(0, -dx))
    out[:, :, dst_y, dst_x] = img[:, :, src_y, src_x]
    return out.reshape(batch, -1)


# Test-time views: identity plus one-pixel translations.
TTA_SHIFTS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))


@torch.no_grad()
def predict_tta(
    model: nn.Module,
    images: torch.Tensor,
    *,
    batch: int = 500,
    shifts: tuple[tuple[int, int], ...] = TTA_SHIFTS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return mean probabilities over the TTA views plus the concatenated logits."""

    probs = None
    logits_all = []
    for dx, dy in shifts:
        view = _shift_view(images, dx, dy) if (dx or dy) else images
        logits = predict(model, view, batch=batch)
        logits_all.append(logits)
        p = torch.softmax(logits, dim=1)
        probs = p if probs is None else probs + p
    probs = probs / float(len(shifts))
    return probs, torch.cat(logits_all, dim=0)


def accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=1) == labels).float().mean().item())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ens", type=int, default=1, help="train N seeds and ensemble")
    parser.add_argument("--tta", action="store_true", help="test-time augmentation")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    config = dict(DEFAULTS)
    if args.config_json:
        config.update(json.loads(args.config_json))

    requested = str(config.get("device", "auto"))
    if requested == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but torch.cuda.is_available() is False")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")

    data = load_mnist(DATA / "mnist")
    train_x, train_y = subset(
        data["train_images"], data["train_labels"], int(config["train_size"]), int(config["subsample_seed"])
    )
    test_x, test_y = subset(
        data["test_images"], data["test_labels"], int(config["test_size"]), int(config["subsample_seed"]) + 1
    )
    train_x, train_y = train_x.to(device), train_y.to(device)
    test_x, test_y = test_x.to(device), test_y.to(device)
    print(f"[device] {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'cpu'})", flush=True)

    runs = []
    probs_sum = None
    for offset in range(max(1, args.ens)):
        seed = args.seed + offset
        seed_everything(seed)
        model = SparseBrainV2(config, seed).to(device)
        trainable = trainable_parameters(model)
        result = train_model(model, train_x, train_y, config, seed, device=device)
        if args.tta:
            probs, _ = predict_tta(model, test_x)
            acc = float((probs.argmax(dim=1) == test_y).float().mean().item())
        else:
            logits = predict(model, test_x)
            acc = accuracy(logits, test_y)
            probs = torch.softmax(logits, dim=1)
        probs_sum = probs if probs_sum is None else probs_sum + probs
        train_acc = accuracy(predict(model, train_x[:5000]), train_y[:5000])
        runs.append(
            {
                "seed": seed,
                "test_accuracy": acc,
                "train_accuracy": train_acc,
                "trainable_params": trainable,
                "train_seconds": result["seconds"],
                "final_loss": result["history"][-1]["loss"] if result["history"] else None,
            }
        )
        print(
            f"[run seed={seed}] test={acc:.4f} train={train_acc:.4f} params={trainable} "
            f"time={result['seconds']:.1f}s",
            flush=True,
        )

    ens_acc = None
    if args.ens > 1 and probs_sum is not None:
        ens_acc = float((probs_sum.argmax(dim=1) == test_y).float().mean().item())
        print(f"[ensemble n={args.ens}] test={ens_acc:.4f}", flush=True)

    payload = {
        "experiment": "mnist_optimized",
        "config": config,
        "runs": runs,
        "ensemble_accuracy": ens_acc,
        "mean_accuracy": sum(r["test_accuracy"] for r in runs) / len(runs),
        "test_size": int(test_x.shape[0]),
        "train_size": int(train_x.shape[0]),
    }
    if args.json:
        emit_result(payload)
    else:
        print(json.dumps(payload, indent=2, default=str))


if __name__ == "__main__":
    main()
