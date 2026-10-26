"""MNIST digit recognition: dense baselines versus the sparse rate brain.

The question this experiment asks is *parameter efficiency at fixed topology*:

    given roughly the same number of trainable scalars, does a fixed sparse
    recurrent core with a cheap readout classify handwritten digits as well as
    a dense multilayer perceptron?

Secondary questions:

* how does accuracy scale with edges per neuron (``degree``), neurons, ticks?
* what does the deployed event path cost in accuracy once a firing threshold
  suppresses small state deltas (``HybridPELSRuntime`` with ``theta``)?
* how much of the work can be skipped at low activity (active edge fraction)?

Everything runs on CPU in pure PyTorch; MNIST is cached under ``data/mnist``.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    DATA,
    Timer,
    emit_result,
    load_mnist,
    measure_latency,
    seed_everything,
    slug,
    trainable_parameters,
)

from neurosim_gpu import Dynamics, HybridPELSRuntime, SparseTopology, random_topology

DEFAULTS: dict[str, Any] = {
    "model": "sparse_brain",
    "neurons": 256,
    "degree": 8,
    "steps": 8,
    "fanout": 4,
    "sensory_gain": 2.0,
    "hidden": 64,
    "learn_sensory": False,
    "lr": 2e-3,
    "weight_decay": 1e-4,
    "epochs": 10,
    "batch": 64,
    "train_size": 10000,
    "test_size": 2000,
    "alpha": 0.72,
    "gain": 1.05,
    "weight_scale": 0.18,
    "theta": 0.05,
    "thetas": [0.0, 0.02, 0.05, 0.1, 0.2],
    "corruptions": False,
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class LogisticRegression(nn.Module):
    def __init__(self, features: int = 784, classes: int = 10) -> None:
        super().__init__()
        self.linear = nn.Linear(features, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class MLP(nn.Module):
    def __init__(self, hidden: int = 64, features: int = 784, classes: int = 10) -> None:
        super().__init__()
        self.hidden = nn.Linear(features, hidden)
        self.out = nn.Linear(hidden, classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(F.relu(self.hidden(x)))


class SparseProjection(nn.Module):
    """Fixed fanout projection ``784 -> neurons`` with no dense matrix."""

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
        dst = torch.randint(
            neurons, (src.numel(),), generator=generator, dtype=torch.int64
        )
        # Keep the aggregate variance O(1) for a uniform fanout.
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
        self.fanout = fanout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        selected = x.index_select(1, self.src)
        contributions = selected * self.weight
        out = x.new_zeros((x.shape[0], self.neurons))
        out.scatter_add_(1, self.dst.expand(x.shape[0], -1), contributions)
        return self.gain * out


class SparseBrain(nn.Module):
    """Fixed sparse recurrent core + sparse sensory fanout + linear readout."""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__()
        neurons = int(config["neurons"])
        degree = int(config["degree"])
        self.neurons = neurons
        self.steps = int(config["steps"])
        self.dynamics = Dynamics(
            torch.full((neurons,), float(config["alpha"])),
            torch.full((neurons,), float(config["gain"])),
        )
        topology = random_topology(
            neurons,
            neurons * degree,
            device="cpu",
            seed=seed,
            weight_scale=float(config["weight_scale"]),
        )
        self.register_buffer("src", topology.src.detach().clone())
        self.register_buffer("dst", topology.dst.detach().clone())
        self.register_buffer("in_rowptr", topology.in_rowptr.detach().clone())
        self.register_buffer("in_col", topology.in_col.detach().clone())
        # CSR value order -> canonical source-major edge id.
        self.register_buffer(
            "in_weight_index",
            torch.argsort(topology.out_weight_index.to(torch.int64)).to(torch.int32),
        )
        self.edge_weight = nn.Parameter(topology.weight.detach().clone())
        self.sensory = SparseProjection(
            784,
            neurons,
            int(config["fanout"]),
            seed=seed,
            learn=bool(config["learn_sensory"]),
            gain=float(config["sensory_gain"]),
        )
        self.readout = nn.Linear(neurons, 10)

    @property
    def edge_count(self) -> int:
        return int(self.edge_weight.numel())

    def _csr(self) -> torch.Tensor:
        values = self.edge_weight[self.in_weight_index.to(torch.int64)]
        return torch.sparse_csr_tensor(
            self.in_rowptr,
            self.in_col,
            values,
            size=(self.neurons, self.neurons),
            check_invariants=False,
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sensory = self.sensory(x)
        state = torch.sigmoid(sensory)
        drive = 0.75 * torch.tanh(sensory)
        return state, drive

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state, drive = self.encode(x)
        weight = self._csr()
        for _ in range(self.steps):
            current = torch.sparse.mm(weight, state.T).T
            state = self.dynamics.advance(state, current, drive)
        return self.readout(state)

    def simulation(self, x: torch.Tensor, theta: float) -> tuple[torch.Tensor, dict[str, float]]:
        """Run the deployed persistent-current runtime and report activity."""

        topology = SparseTopology.from_edges(
            self.neurons,
            self.src,
            self.dst,
            self.edge_weight.detach(),
        )
        state, drive = self.encode(x)
        runtime = HybridPELSRuntime(
            topology,
            Dynamics(
                torch.full((self.neurons,), float(self.dynamics.alpha[0])),
                torch.full((self.neurons,), float(self.dynamics.gain[0])),
            ),
            state,
            theta=theta,
            propagation="matrix",
        )
        active_edge_total = 0.0
        active_neuron_total = 0.0
        with torch.no_grad():
            for _ in range(self.steps):
                state, stats = runtime.step(drive, collect_stats=True)
                active_edge_total += stats.active_edges
                active_neuron_total += stats.active_neurons
        logits = self.readout(state)
        total_edges = max(self.steps * topology.n_edges * x.shape[0], 1)
        total_neurons = max(self.steps * self.neurons * x.shape[0], 1)
        return logits, {
            "active_edge_fraction": active_edge_total / total_edges,
            "active_neuron_fraction": active_neuron_total / total_neurons,
        }


class DenseRNN(nn.Module):
    """Dense recurrent control: same neuron count, all N^2 weights trained."""

    def __init__(self, config: dict[str, Any], seed: int) -> None:
        super().__init__()
        neurons = int(config["neurons"])
        self.neurons = neurons
        self.steps = int(config["steps"])
        self.dynamics = Dynamics(
            torch.full((neurons,), float(config["alpha"])),
            torch.full((neurons,), float(config["gain"])),
        )
        generator = torch.Generator().manual_seed(seed + 7)
        self.recurrent = nn.Parameter(
            torch.randn(neurons, neurons, generator=generator)
            * float(config["weight_scale"])
            / math.sqrt(neurons)
        )
        self.sensory = nn.Linear(784, neurons)
        self.readout = nn.Linear(neurons, 10)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sensory = self.sensory(x)
        return torch.sigmoid(sensory), 0.75 * torch.tanh(sensory)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        state, drive = self.encode(x)
        for _ in range(self.steps):
            current = state @ self.recurrent.T
            state = self.dynamics.advance(state, current, drive)
        return self.readout(state)


def build_model(config: dict[str, Any], seed: int) -> nn.Module:
    model = config["model"]
    if model == "logreg":
        return LogisticRegression()
    if model == "mlp":
        return MLP(hidden=int(config["hidden"]))
    if model == "sparse_brain":
        return SparseBrain(config, seed)
    if model == "dense_rnn":
        return DenseRNN(config, seed)
    raise ValueError(f"unknown model {model!r}")


def matched_hidden(config: dict[str, Any]) -> int:
    """MLP width whose trainable scalar count matches the sparse brain."""

    neurons = int(config["neurons"])
    degree = int(config["degree"])
    edges = neurons * degree
    readout = neurons * 10 + 10
    target = edges + readout
    hidden = round((target - 10) / (784 + 10))
    return max(2, hidden)


# ---------------------------------------------------------------------------
# Data and training
# ---------------------------------------------------------------------------


def subset(
    images: torch.Tensor,
    labels: torch.Tensor,
    size: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    size = min(size, images.shape[0])
    generator = torch.Generator().manual_seed(seed)
    index = torch.randperm(images.shape[0], generator=generator)[:size]
    return images.index_select(0, index), labels.index_select(0, index)


def train(
    model: nn.Module,
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    config: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )
    epochs = int(config["epochs"])
    batch = int(config["batch"])
    generator = torch.Generator().manual_seed(seed + 12345)
    history: list[dict[str, float]] = []
    timer = Timer()
    for epoch in range(epochs):
        model.train()
        order = torch.randperm(train_x.shape[0], generator=generator)
        total_loss = 0.0
        correct = 0
        seen = 0
        for start in range(0, train_x.shape[0], batch):
            index = order[start : start + batch]
            x = train_x.index_select(0, index)
            y = train_y.index_select(0, index)
            logits = model(x)
            loss = F.cross_entropy(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * y.numel()
            correct += int((logits.argmax(dim=1) == y).sum().item())
            seen += y.numel()
        history.append(
            {
                "epoch": epoch + 1,
                "loss": total_loss / max(seen, 1),
                "train_acc": correct / max(seen, 1),
                "seconds": timer.seconds,
            }
        )
    return {"history": history, "seconds": timer.seconds}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    batch: int = 500,
) -> tuple[float, list[torch.Tensor]]:
    model.eval()
    correct = 0
    logits_all: list[torch.Tensor] = []
    for start in range(0, images.shape[0], batch):
        x = images[start : start + batch]
        y = labels[start : start + batch]
        logits = model(x)
        logits_all.append(logits)
        correct += int((logits.argmax(dim=1) == y).sum().item())
    return correct / max(images.shape[0], 1), logits_all


def activity_curve(
    model: SparseBrain,
    images: torch.Tensor,
    labels: torch.Tensor,
    thetas: list[float],
) -> list[dict[str, float]]:
    rows = []
    base, _ = evaluate(model, images, labels)
    for theta in thetas:
        logits, stats = model.simulation(images, theta)
        acc = float((logits.argmax(dim=1) == labels).float().mean().item())
        rows.append(
            {
                "theta": float(theta),
                "accuracy": acc,
                "accuracy_delta_vs_matrix": acc - base,
                "active_edge_fraction": stats["active_edge_fraction"],
                "active_neuron_fraction": stats["active_neuron_fraction"],
            }
        )
    return rows


def corrupted_variants(images: torch.Tensor, seed: int) -> dict[str, torch.Tensor]:
    """Deterministic distribution shift set applied to the test split."""

    generator = torch.Generator().manual_seed(seed + 4242)
    views: dict[str, torch.Tensor] = {}
    for sigma in (0.1, 0.2, 0.3):
        noise = torch.randn(images.shape, generator=generator) * sigma
        views[f"gaussian_{sigma:g}"] = (images + noise).clamp(0.0, 1.0)
    for size in (4, 8):
        mask = torch.zeros_like(images).view(-1, 28, 28)
        top = torch.randint(0, 28 - size + 1, (images.shape[0],), generator=generator)
        left = torch.randint(0, 28 - size + 1, (images.shape[0],), generator=generator)
        for row in range(size):
            for col in range(size):
                mask[torch.arange(images.shape[0]), top + row, left + col] = 1.0
        views[f"occlusion_{size}x{size}"] = images * (1.0 - mask.view_as(images))
    for shift in (2,):
        rolled = images.view(-1, 28, 28).roll(shifts=shift, dims=2)
        views[f"shift_{shift}px"] = rolled.view_as(images)
    return views


def corruption_curve(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    seed: int,
) -> list[dict[str, float]]:
    rows = []
    for name, view in corrupted_variants(images, seed).items():
        acc, _ = evaluate(model, view, labels)
        rows.append({"condition": name, "accuracy": acc})
    return rows


def storage_bytes(model: nn.Module) -> int:
    if isinstance(model, SparseBrain):
        topology = SparseTopology.from_edges(
            model.neurons,
            model.src,
            model.dst,
            model.edge_weight.detach(),
        )
        return int(topology.storage_bytes())
    return int(sum(p.numel() * p.element_size() for p in model.parameters()))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def parse_config(raw: str | None) -> dict[str, Any]:
    config = dict(DEFAULTS)
    if raw:
        config.update(json.loads(raw))
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-json", default=None, help="JSON overrides for DEFAULTS")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--json", action="store_true", help="emit the RESULT line")
    parser.add_argument("--smoke", action="store_true", help="tiny fast configuration")
    args = parser.parse_args()

    config = parse_config(args.config_json)
    if args.smoke:
        config.update({"epochs": 1, "train_size": 512, "test_size": 256, "steps": 4})
    seed_everything(args.seed)
    device = torch.device("cpu")

    total = Timer()
    print(f"[mnist] config={config} seed={args.seed}", flush=True)

    data_timer = Timer()
    data = load_mnist(DATA / "mnist")
    train_x, train_y = subset(data["train_images"], data["train_labels"], int(config["train_size"]), args.seed)
    test_x, test_y = subset(data["test_images"], data["test_labels"], int(config["test_size"]), args.seed + 1)

    if config["model"] == "mlp" and config.get("match_params"):
        config["hidden"] = matched_hidden(config)

    model = build_model(config, args.seed).to(device)
    trainable = trainable_parameters(model)
    fixed = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    fixed += sum(b.numel() for b in model.buffers())

    result = train(model, train_x, train_y, config, args.seed)
    test_acc, _ = evaluate(model, test_x, test_y)
    train_acc, _ = evaluate(model, train_x, train_y)

    latency_source = test_x[:1]
    model.eval()
    with torch.no_grad():
        latency = measure_latency(lambda: model(latency_source), warmup=3, iterations=15)

    metrics: dict[str, Any] = {
        "test_accuracy": test_acc,
        "train_accuracy": train_acc,
        "trainable_params": trainable,
        "fixed_params": int(fixed),
        "storage_bytes": storage_bytes(model),
        "train_seconds": result["seconds"],
        "samples_per_second": train_x.shape[0] * int(config["epochs"]) / max(result["seconds"], 1e-9),
        "epochs": int(config["epochs"]),
        "history": result["history"],
        **latency,
    }

    if config.get("corruptions"):
        metrics["corruptions"] = corruption_curve(model, test_x, test_y, args.seed)

    if isinstance(model, SparseBrain):
        metrics["edges"] = model.edge_count
        metrics["sensory_edges"] = int(model.sensory.weight.numel())
        metrics["degree"] = int(config["degree"])
        metrics["matched_mlp_hidden"] = matched_hidden(config)
        metrics["activity"] = activity_curve(
            model,
            test_x[: min(512, test_x.shape[0])],
            test_y[: min(512, test_y.shape[0])],
            [float(t) for t in config["thetas"]],
        )
        with torch.no_grad():
            state, _ = model.encode(test_x[:256])
        metrics["state_mean"] = float(state.mean().item())
        metrics["state_saturated_fraction"] = float(((state > 0.9) | (state < 0.1)).float().mean().item())

    payload = {
        "experiment": "mnist",
        "run_id": f"mnist_{slug(config)}_seed{args.seed}",
        "config": config,
        "seed": args.seed,
        "device": str(device),
        "metrics": metrics,
        "timing": {
            "data_seconds": data_timer.seconds,
            "total_seconds": total.seconds,
        },
    }

    if args.json:
        # The suite driver is the single writer of results/raw; standalone runs
        # just stream the payload to stdout.
        emit_result(payload)
    else:
        print(json.dumps({"metrics": metrics}, indent=2, default=str))


if __name__ == "__main__":
    main()
