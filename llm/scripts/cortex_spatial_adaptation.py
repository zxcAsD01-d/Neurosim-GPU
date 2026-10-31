#!/usr/bin/env python3
"""Offline migration + rewiring experiment for the cortical sheet.

This uses the two-plate geometry (front input plate at z=0, back output plate at
z=50, nominal neuron spacing 1) and the trained dense matrix. It measures how
far neurons move and how many synapses are formed/pruned per adaptation epoch.

The script does **not** train the language model. It is the first adaptation
experiment; the model checkpoint and optimizer are left untouched.

Usage:
    python llm/scripts/cortex_spatial_adaptation.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import (  # noqa: E402
    Line3DCollection,
    Poly3DCollection,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cortex_connectome import REGION_COLORS, add_network  # noqa: E402

from neurosim_gpu.cortex.checkpoint import load_checkpoint  # noqa: E402
from neurosim_gpu.cortex.config import ExperimentConfig  # noqa: E402
from neurosim_gpu.cortex.data import build_dataloaders  # noqa: E402
from neurosim_gpu.cortex.model import CortexSandwichLM  # noqa: E402
from neurosim_gpu.cortex.spatial import (  # noqa: E402
    PlateGeometry,
    migrate,
    rewire,
    sparsify_weights,
)
from neurosim_gpu.cortex.tokenization import build_tokenizer  # noqa: E402


@torch.no_grad()
def activity_from_checkpoint(
    checkpoint: Path,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, ExperimentConfig, torch.Tensor]:
    """Run one real text batch through the model and return per-neuron state."""

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    config = ExperimentConfig.from_dict(payload["config"])
    tokenizer = build_tokenizer(config.data, config.backbone)
    model = CortexSandwichLM(config, tokenizer.vocab_size)
    load_checkpoint(checkpoint, model, map_location="cpu")
    model.to(device)
    model.eval()

    train_loader, _, _ = build_dataloaders(
        config.data, tokenizer, verbose=False
    )
    batch = next(iter(train_loader))
    input_ids = batch["input_ids"].to(device)
    hidden, _ = model.backbone.encode(input_ids)
    _, state, _ = model.brain(hidden.float())
    activity = state.mean(dim=0).detach().float().cpu()
    low, high = torch.quantile(activity, torch.tensor([0.02, 0.98]))
    activity = ((activity - low) / (high - low).clamp_min(1e-6)).clamp(0.0, 1.0)
    weights = payload["model"]["brain.recurrent"].detach().float()
    return activity, config, weights


def sample_masked_edges(
    weights: torch.Tensor,
    mask: torch.Tensor,
    per_source: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = (weights.abs() * mask).numpy()
    k = min(per_source, score.shape[1] - 1)
    order = np.argpartition(-score, kth=k, axis=1)[:, :k]
    src = np.repeat(np.arange(score.shape[0]), k)
    dst = order.reshape(-1)
    values = score[src, dst] * np.sign(weights.numpy()[src, dst])
    return src, dst, values


def region_array(n: int, regions: int) -> np.ndarray:
    return (np.arange(n) * regions // n).astype(np.int64)


def draw_plate(axis, geometry: PlateGeometry, z: float, color: str) -> None:
    vertices = geometry.plate_vertices(z)
    collection = Poly3DCollection(
        vertices,
        alpha=0.10,
        facecolors=color,
        edgecolors=color,
        linewidths=0.8,
    )
    axis.add_collection3d(collection)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="runs/cortex_long_lora/final.pt"
    )
    parser.add_argument(
        "--output", default="llm/results/cortex/spatial_adaptation_3d.png"
    )
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--edges-per-neuron", type=int, default=64)
    parser.add_argument("--prune-fraction", type=float, default=0.01)
    parser.add_argument("--migration-speed", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    activity, config, weights = activity_from_checkpoint(
        Path(args.checkpoint), device=device
    )
    n = weights.shape[0]
    regions = int(config.brain.regions)
    geometry = PlateGeometry(
        n_neurons=n,
        mean_neuron_distance=1.0,
        plate_distance=50.0,
    )
    positions = geometry.sample_positions(seed=args.seed)
    mask = sparsify_weights(
        weights,
        positions,
        edges_per_neuron=args.edges_per_neuron,
        distance_decay=5.0,
    )
    weights = weights * mask
    starting = {
        "positions": positions.clone(),
        "weights": weights.clone(),
        "mask": mask.clone(),
        "edges": int(mask.sum()),
        "mean_abs_weight": float(weights.abs().mean()),
    }
    print(
        f"geometry: N={n}, d={geometry.mean_neuron_distance}, "
        f"D={geometry.plate_distance}, side={geometry.side:.3f}, "
        f"edges={starting['edges']}"
    )

    history = []
    for epoch in range(1, args.epochs + 1):
        positions, displacement = migrate(
            positions,
            activity,
            geometry,
            maximum_speed=args.migration_speed,
        )
        weights, mask, rewire_stats = rewire(
            weights,
            mask,
            positions,
            activity,
            fraction=args.prune_fraction,
        )
        weights = weights * mask
        moved = float((displacement > 0.05).float().mean())
        mean_displacement = float(displacement.mean())
        history.append(
            {
                "epoch": epoch,
                "moved_fraction": moved,
                "mean_displacement": mean_displacement,
                "formed": rewire_stats["formed"],
                "pruned": rewire_stats["pruned"],
                "rewire_ratio": rewire_stats["rewire_ratio"],
                "edges": int(mask.sum()),
            }
        )
        print(
            f"epoch {epoch}: moved={moved:.3f} "
            f"disp={mean_displacement:.3f} "
            f"formed={rewire_stats['formed']:.0f} "
            f"pruned={rewire_stats['pruned']:.0f} "
            f"rewire={rewire_stats['rewire_ratio']:.4f} "
            f"edges={int(mask.sum())}"
        )

    region = region_array(n, regions)
    strength = torch.ones(n)
    cumulative_displacement = (
        positions - starting["positions"]
    ).norm(dim=1)
    final = dict(history[-1])
    final.update(
        {
            "cumulative_moved_fraction": float(
                (cumulative_displacement > 0.05).float().mean()
            ),
            "cumulative_mean_displacement": float(
                cumulative_displacement.mean()
            ),
            "cumulative_formed": int(
                sum(item["formed"] for item in history)
            ),
            "cumulative_pruned": int(
                sum(item["pruned"] for item in history)
            ),
            "cumulative_rewire_ratio": float(
                sum(item["formed"] + item["pruned"] for item in history)
                / max(starting["edges"], 1)
            ),
        }
    )

    figure = plt.figure(figsize=(16, 8), dpi=180)
    figure.patch.set_facecolor("#0b0f14")
    for panel, (positions_panel, weights_panel, mask_panel, title) in enumerate(
        (
            (
                starting["positions"],
                starting["weights"],
                starting["mask"],
                "before adaptation",
            ),
            (positions, weights, mask, "after adaptation"),
        )
    ):
        axis = figure.add_subplot(1, 2, panel + 1, projection="3d")
        src, dst, values = sample_masked_edges(
            weights_panel,
            mask_panel,
            per_source=2,
        )
        add_network(
            axis,
            positions_panel.numpy(),
            src,
            dst,
            values,
            region,
            strength.numpy(),
            edge_alpha=0.5,
        )
        draw_plate(axis, geometry, 0.0, "#74c0fc")
        draw_plate(axis, geometry, geometry.plate_distance, "#ffa94d")
        axis.text2D(
            0.02,
            0.03,
            "front plate z=0",
            transform=axis.transAxes,
            color="#74c0fc",
            fontsize=8,
        )
        axis.text2D(
            0.02,
            0.93,
            "back plate z=50",
            transform=axis.transAxes,
            color="#ffa94d",
            fontsize=8,
        )
        if panel == 1:
            sample = np.linspace(0, n - 1, 64).astype(int)
            delta = positions_panel.numpy()[sample] - starting["positions"].numpy()[sample]
            axis.quiver(
                starting["positions"].numpy()[sample, 0],
                starting["positions"].numpy()[sample, 1],
                starting["positions"].numpy()[sample, 2],
                delta[:, 0],
                delta[:, 1],
                delta[:, 2],
                color="#ffd43b",
                alpha=0.55,
                linewidth=0.8,
                arrow_length_ratio=0.25,
            )
        axis.set_title(title, color="#f1f3f5", fontsize=13, pad=2)

    last = final
    figure.suptitle(
        (
            "Cortex-LM spatial adaptation · front/back plates at D=50 d̄ · "
            f"{args.edges_per_neuron} synapses/neuron"
        ),
        color="#f1f3f5",
        fontsize=15,
        x=0.04,
        ha="left",
        y=0.96,
    )
    figure.text(
        0.04,
        0.925,
        (
            f"epochs {args.epochs} · moved {last['cumulative_moved_fraction']:.1%} "
            f"(mean {last['cumulative_mean_displacement']:.2f} d̄) · "
            f"rewire {last['cumulative_rewire_ratio']:.2%} total · "
            f"edges {last['edges']:,}"
        ),
        color="#adb5bd",
        fontsize=9.5,
        ha="left",
    )
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=7,
            markerfacecolor=REGION_COLORS[index],
            markeredgecolor="none",
            label=f"R{index}",
        )
        for index in range(regions)
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#ff8c69", lw=2, label="excitatory"),
            Line2D([0], [0], color="#4dd0e1", lw=2, label="inhibitory"),
            Line2D([0], [0], color="#74c0fc", lw=2, label="front plate"),
            Line2D([0], [0], color="#ffa94d", lw=2, label="back plate"),
        ]
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        labelcolor="#f1f3f5",
        fontsize=8,
        bbox_to_anchor=(0.5, 0.01),
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.1,
    )
    plt.close(figure)

    stats = {
        "checkpoint": str(args.checkpoint),
        "geometry": {
            "n_neurons": n,
            "mean_neuron_distance": geometry.mean_neuron_distance,
            "plate_distance": geometry.plate_distance,
            "plate_side": geometry.side,
        },
        "starting": {
            "edges": starting["edges"],
            "migration_ratio": 0.0,
            "rewire_ratio": 0.0,
        },
        "final": last,
        "history": history,
        "mode": "offline adaptation; model weights and optimizer untouched",
    }
    stats_path = Path("runs/cortex_spatial/adaptation_stats.json")
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "weights": weights,
            "mask": mask,
            "positions": positions,
            "activity": activity,
            "geometry": {
                "mean_neuron_distance": geometry.mean_neuron_distance,
                "plate_distance": geometry.plate_distance,
                "side": geometry.side,
            },
            "stats": stats,
        },
        stats_path.with_suffix(".pt"),
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
