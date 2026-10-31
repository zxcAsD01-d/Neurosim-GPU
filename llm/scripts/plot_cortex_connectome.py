#!/usr/bin/env python3
"""Render a 3D view of the Cortex-LM internal connectome.

The current Cortex-LM brain is a 1024-neuron dense matrix with no learned 3D
coordinates. The figure therefore uses two explicit *visualisation layouts*:

* ``region``: eight region groups placed on a sphere, neurons scattered inside
  each group. This shows the region/gain grouping, not biological geometry.
* ``spectral``: a 3D Fiedler embedding of the symmetrised absolute-weight graph,
  which reflects the actual trained connectivity.

Only the strongest outgoing edges per neuron are drawn; the matrix is 100%
dense, so drawing all 1,048,576 edges would be unreadable.

Usage:
    python llm/scripts/plot_cortex_connectome.py
    python llm/scripts/plot_cortex_connectome.py --checkpoint runs/cortex_long_lora/last.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import Line3DCollection  # noqa: E402


REGION_COLORS = [
    "#ff6b6b",
    "#ffa94d",
    "#ffd43b",
    "#69db7c",
    "#38d9a9",
    "#4dabf7",
    "#9775fa",
    "#f783ac",
]


def fibonacci_sphere(count: int, radius: float) -> np.ndarray:
    indices = np.arange(count, dtype=np.float64) + 0.5
    phi = np.arccos(1.0 - 2.0 * indices / count)
    theta = np.pi * (1.0 + 5.0**0.5) * indices
    return np.stack(
        [
            np.cos(theta) * np.sin(phi),
            np.sin(theta) * np.sin(phi),
            np.cos(phi),
        ],
        axis=1,
    ) * radius


def region_layout(region: np.ndarray, regions: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    centers = fibonacci_sphere(regions, radius=2.1)
    positions = np.zeros((region.size, 3), dtype=np.float32)
    for index in range(regions):
        mask = region == index
        count = int(mask.sum())
        positions[mask] = centers[index] + rng.normal(
            scale=0.42, size=(count, 3)
        )
    return positions


def spectral_layout(weights: np.ndarray) -> np.ndarray:
    adjacency = (np.abs(weights) + np.abs(weights).T) * 0.5
    degree = adjacency.sum(axis=1)
    degree = np.maximum(degree, 1e-8)
    inverse_sqrt = 1.0 / np.sqrt(degree)
    normalized = (
        inverse_sqrt[:, None] * adjacency * inverse_sqrt[None, :]
    )
    laplacian = np.eye(weights.shape[0]) - normalized
    _, vectors = np.linalg.eigh(laplacian)
    embedding = vectors[:, 1:4].copy()
    for column in range(3):
        low, high = np.percentile(embedding[:, column], [1, 99])
        scale = max(high - low, 1e-6)
        embedding[:, column] = (embedding[:, column] - low) / scale * 4.0 - 2.0
    return embedding.astype(np.float32)


def sample_edges(
    weights: np.ndarray,
    per_source: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    off_diagonal = weights.copy()
    np.fill_diagonal(off_diagonal, 0.0)
    k = min(per_source, off_diagonal.shape[1] - 1)
    order = np.argpartition(-np.abs(off_diagonal), kth=k, axis=1)[:, :k]
    sources = np.repeat(np.arange(weights.shape[0]), k)
    destinations = order.reshape(-1)
    values = off_diagonal[sources, destinations]
    return sources, destinations, values


def add_network(
    axis,
    positions: np.ndarray,
    sources: np.ndarray,
    destinations: np.ndarray,
    values: np.ndarray,
    region: np.ndarray,
    strength: np.ndarray,
    *,
    edge_alpha: float = 0.24,
) -> None:
    scale = float(np.percentile(np.abs(values), 95))
    scale = max(scale, 1e-6)
    widths = 0.3 + 1.0 * np.minimum(np.abs(values) / scale, 1.5)
    colors = []
    for value in values:
        if value >= 0:
            colors.append((1.0, 0.42, 0.28, edge_alpha))
        else:
            colors.append((0.23, 0.78, 0.88, edge_alpha))
    segments = np.stack(
        [positions[sources], positions[destinations]], axis=1
    )
    collection = Line3DCollection(
        segments, colors=colors, linewidths=widths
    )
    axis.add_collection3d(collection)

    sizes = 12.0 + 32.0 * np.sqrt(strength / max(float(strength.max()), 1e-6))
    node_colors = [REGION_COLORS[int(value) % len(REGION_COLORS)] for value in region]
    axis.scatter(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        s=sizes,
        c=node_colors,
        alpha=0.92,
        edgecolors="#0b0f14",
        linewidths=0.3,
        depthshade=False,
    )
    extent = positions.max(axis=0) - positions.min(axis=0)
    center = (positions.max(axis=0) + positions.min(axis=0)) * 0.5
    half = float(extent.max()) * 0.56
    axis.set_xlim(center[0] - half, center[0] + half)
    axis.set_ylim(center[1] - half, center[1] + half)
    axis.set_zlim(center[2] - half, center[2] + half)
    axis.set_box_aspect((1, 1, 1), zoom=1.18)
    axis.set_axis_off()
    axis.patch.set_facecolor("#0b0f14")
    axis.patch.set_alpha(1.0)
    axis.view_init(elev=20, azim=35)


def add_legends(figure, regions: int) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=REGION_COLORS[index],
            markeredgecolor="none",
            label=f"R{index}",
        )
        for index in range(regions)
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#ff8c69", lw=2, label="excitatory edge"),
            Line2D([0], [0], color="#4dd0e1", lw=2, label="inhibitory edge"),
        ]
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 10),
        frameon=False,
        fontsize=9,
        labelcolor="#f1f3f5",
        bbox_to_anchor=(0.5, 0.012),
    )


def render_single(
    *,
    positions: np.ndarray,
    sources: np.ndarray,
    destinations: np.ndarray,
    values: np.ndarray,
    region: np.ndarray,
    strength: np.ndarray,
    regions: int,
    stats: dict[str, object],
    title: str,
    output: Path,
) -> None:
    figure = plt.figure(figsize=(11, 11), dpi=220)
    figure.patch.set_facecolor("#0b0f14")
    axis = figure.add_axes([-0.02, -0.10, 1.04, 0.98], projection="3d")
    add_network(
        axis, positions, sources, destinations, values, region, strength
    )
    figure.text(
        0.5,
        0.935,
        title,
        color="#f1f3f5",
        fontsize=16,
        ha="center",
    )
    figure.suptitle(
        "Cortex-LM internal connectome · 1024 neurons",
        color="#f1f3f5",
        fontsize=18,
        x=0.04,
        ha="left",
        y=0.985,
    )
    figure.text(
        0.04,
        0.895,
        (
            f"step {stats['step']} · top {stats['drawn_edges'] // 1024} "
            "outgoing synapses per neuron · "
            f"self-loops {stats['self_loops_nonzero']}/1024 · "
            f"mean |w| {stats['abs_weight_mean']:.3f}"
        ),
        color="#adb5bd",
        fontsize=9.5,
        ha="left",
    )
    add_legends(figure, regions)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(figure)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="runs/cortex_long_lora/last.pt"
    )
    parser.add_argument(
        "--output", default="llm/results/cortex/connectome_3d.png"
    )
    parser.add_argument("--edges-per-source", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument(
        "--mode",
        choices=("composite", "region", "spectral"),
        default="composite",
    )
    args = parser.parse_args()

    checkpoint = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    weights = (
        checkpoint["model"]["brain.recurrent"].detach().float().numpy()
    )
    step = int(checkpoint.get("step", -1))
    regions = int(checkpoint["config"]["brain"]["regions"])
    neurons = weights.shape[0]
    if weights.shape != (neurons, neurons):
        raise ValueError(f"expected a square W, got {weights.shape}")

    region = (np.arange(neurons) * regions // neurons).astype(np.int64)
    strength = np.abs(weights).sum(axis=1)
    sources, destinations, values = sample_edges(
        weights, args.edges_per_source
    )

    figure = plt.figure(figsize=(15, 12), dpi=200)
    figure.patch.set_facecolor("#0b0f14")

    region_positions = region_layout(region, regions, args.seed)
    spectral_positions = spectral_layout(weights)

    title_color = "#f1f3f5"
    muted_color = "#adb5bd"

    axis_a = figure.add_subplot(2, 2, 1, projection="3d")
    add_network(
        axis_a,
        region_positions,
        sources,
        destinations,
        values,
        region,
        strength,
    )
    axis_a.set_title(
        "A  region layout (schematic)",
        color=title_color,
        fontsize=13,
        loc="left",
        pad=8,
    )

    axis_b = figure.add_subplot(2, 2, 2, projection="3d")
    add_network(
        axis_b,
        spectral_positions,
        sources,
        destinations,
        values,
        region,
        strength,
    )
    axis_b.set_title(
        "B  spectral layout of trained W",
        color=title_color,
        fontsize=13,
        loc="left",
        pad=8,
    )

    axis_c = figure.add_subplot(2, 2, 3)
    region_matrix = np.zeros((regions, regions), dtype=np.float32)
    for source_region in range(regions):
        source_mask = region == source_region
        for destination_region in range(regions):
            destination_mask = region == destination_region
            region_matrix[source_region, destination_region] = weights[
                source_mask
            ][:, destination_mask].mean()
    limit = float(np.abs(region_matrix).max())
    image = axis_c.imshow(
        region_matrix,
        cmap="RdBu_r",
        vmin=-limit,
        vmax=limit,
        interpolation="nearest",
    )
    axis_c.set_facecolor("#0b0f14")
    axis_c.set_title(
        "C  mean weight between regions",
        color=title_color,
        fontsize=13,
        loc="left",
    )
    axis_c.set_xticks(range(regions))
    axis_c.set_yticks(range(regions))
    axis_c.set_xticklabels([f"R{i}" for i in range(regions)], color=muted_color)
    axis_c.set_yticklabels([f"R{i}" for i in range(regions)], color=muted_color)
    axis_c.set_xlabel("destination", color=muted_color)
    axis_c.set_ylabel("source", color=muted_color)
    for spine in axis_c.spines.values():
        spine.set_color("#343a40")
    colorbar = figure.colorbar(image, ax=axis_c, fraction=0.046, pad=0.04)
    colorbar.ax.tick_params(colors=muted_color)
    colorbar.outline.set_edgecolor("#343a40")

    axis_d = figure.add_subplot(2, 2, 4)
    off_diagonal = weights[~np.eye(neurons, dtype=bool)]
    positive = off_diagonal[off_diagonal >= 0]
    negative = off_diagonal[off_diagonal < 0]
    bins = np.linspace(
        float(np.percentile(off_diagonal, 0.1)),
        float(np.percentile(off_diagonal, 99.9)),
        80,
    )
    axis_d.hist(
        positive,
        bins=bins,
        color="#ff8c69",
        alpha=0.72,
        label=f"excitatory ({positive.size:,})",
    )
    axis_d.hist(
        negative,
        bins=bins,
        color="#4dd0e1",
        alpha=0.72,
        label=f"inhibitory ({negative.size:,})",
    )
    axis_d.set_facecolor("#0b0f14")
    axis_d.set_title(
        "D  off-diagonal weight distribution",
        color=title_color,
        fontsize=13,
        loc="left",
    )
    axis_d.tick_params(colors=muted_color)
    axis_d.set_xlabel("weight", color=muted_color)
    axis_d.set_ylabel("synapses", color=muted_color)
    axis_d.grid(color="#212529", alpha=0.5)
    for spine in axis_d.spines.values():
        spine.set_color("#343a40")
    legend = axis_d.legend(facecolor="#161b22", edgecolor="#343a40")
    for text in legend.get_texts():
        text.set_color(title_color)

    diagonal = np.diag(weights)
    stats = {
        "checkpoint": str(args.checkpoint),
        "step": step,
        "neurons": int(neurons),
        "synapses": int(weights.size),
        "density": float((weights != 0).mean()),
        "self_loops_nonzero": int((diagonal != 0).sum()),
        "self_loop_abs_mean": float(np.abs(diagonal).mean()),
        "abs_weight_mean": float(np.abs(weights).mean()),
        "positive_fraction": float((weights > 0).mean()),
        "drawn_edges": int(sources.size),
        "regions": regions,
    }

    if args.mode != "composite":
        positions = (
            region_positions
            if args.mode == "region"
            else spectral_positions
        )
        title = (
            "Region layout (schematic)"
            if args.mode == "region"
            else "Spectral layout of trained W"
        )
        output = Path(args.output)
        render_single(
            positions=positions,
            sources=sources,
            destinations=destinations,
            values=values,
            region=region,
            strength=strength,
            regions=regions,
            stats=stats,
            title=title,
            output=output,
        )
        output.with_suffix(".json").write_text(
            json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        print(f"wrote {output}")
        return 0

    region_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markersize=8,
            markerfacecolor=REGION_COLORS[index],
            markeredgecolor="none",
            label=f"R{index}",
        )
        for index in range(regions)
    ]
    region_handles.extend(
        [
            Line2D([0], [0], color="#ff8c69", lw=2, label="excitatory edge"),
            Line2D([0], [0], color="#4dd0e1", lw=2, label="inhibitory edge"),
        ]
    )
    figure.legend(
        handles=region_handles,
        loc="lower center",
        ncol=min(len(region_handles), 10),
        frameon=False,
        fontsize=9,
        labelcolor=title_color,
        bbox_to_anchor=(0.5, 0.012),
    )

    figure.suptitle(
        "Cortex-LM internal connectome · 1024 neurons · 1,048,576 dense synapses",
        color=title_color,
        fontsize=18,
        x=0.06,
        ha="left",
        y=0.975,
    )
    figure.text(
        0.06,
        0.94,
        (
            f"step {step} · drawn: top {args.edges_per_source} outgoing "
            "synapses per neuron (4,096) · diagonal/self-loops: "
            f"{stats['self_loops_nonzero']}/{neurons} nonzero · "
            "positions are visualisation layouts, not learned coordinates"
        ),
        color=muted_color,
        fontsize=9.5,
        ha="left",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        facecolor=figure.get_facecolor(),
        bbox_inches="tight",
        pad_inches=0.15,
    )
    plt.close(figure)

    stats_path = output.with_suffix(".json")
    stats_path.write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
