#!/usr/bin/env python3
"""Draw the current stage3 cortical sheet: structure, positions, connections.

The checkpoint does not persist positions or the edge mask (they are
non-persistent buffers), so this reconstructs the *current reloaded* view from
the trained W, the seed and the top-k sparsifier.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from mpl_toolkits.mplot3d.art3d import (  # noqa: E402
    Line3DCollection,
    Poly3DCollection,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from plot_cortex_connectome import REGION_COLORS  # noqa: E402

from neurosim_gpu.cortex.config import ExperimentConfig  # noqa: E402
from neurosim_gpu.cortex.spatial import (  # noqa: E402
    PlateGeometry,
    sparsify_weights,
)

BACKGROUND = "#0b0f14"
TEXT = "#f1f3f5"
MUTED = "#adb5bd"


def top_edges(
    weights: np.ndarray,
    mask: np.ndarray,
    per_source: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    score = np.abs(weights) * mask
    k = min(per_source, score.shape[1] - 1)
    order = np.argpartition(-score, kth=k, axis=1)[:, :k]
    src = np.repeat(np.arange(score.shape[0]), k)
    dst = order.reshape(-1)
    values = weights[src, dst]
    return src, dst, values


def edge_segments(
    positions: np.ndarray, src: np.ndarray, dst: np.ndarray
) -> np.ndarray:
    return np.stack([positions[src], positions[dst]], axis=1)


def plate_mesh(geometry: PlateGeometry, z: float) -> Poly3DCollection:
    return Poly3DCollection(
        geometry.plate_vertices(z),
        alpha=0.09,
        facecolors="#74c0fc" if z == 0 else "#ffa94d",
        edgecolors="#74c0fc" if z == 0 else "#ffa94d",
        linewidths=0.8,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint", default="runs/cortex_stage3/final.pt"
    )
    parser.add_argument(
        "--output", default="llm/results/cortex/current_structure.png"
    )
    parser.add_argument("--edges-per-source", type=int, default=3)
    args = parser.parse_args()

    payload = torch.load(
        args.checkpoint, map_location="cpu", weights_only=False
    )
    config = ExperimentConfig.from_dict(payload["config"])
    weights = payload["model"]["brain.recurrent"].float()
    n = weights.shape[0]
    geometry = PlateGeometry(
        n_neurons=n, mean_neuron_distance=1.0, plate_distance=50.0
    )
    positions = geometry.sample_positions(seed=config.brain.seed)
    mask = sparsify_weights(
        weights,
        positions,
        edges_per_neuron=config.brain.spatial_edges_per_neuron,
        distance_decay=5.0,
    )
    weight_np = (weights * mask).numpy()
    mask_np = mask.numpy()
    position_np = positions.numpy()
    region = (np.arange(n) * int(config.brain.regions) // n).astype(int)
    src, dst, _ = top_edges(weight_np, mask_np, args.edges_per_source)
    edge_values = weight_np[src, dst]
    all_idx = mask.nonzero(as_tuple=False)
    edge_length = torch.cdist(positions, positions)[
        all_idx[:, 0], all_idx[:, 1]
    ].numpy()
    nearest = torch.cdist(positions, positions).topk(
        2, largest=False
    ).values[:, 1]

    figure = plt.figure(figsize=(18, 10.5), dpi=170)
    figure.patch.set_facecolor(BACKGROUND)
    grid = figure.add_gridspec(
        2,
        3,
        left=0.035,
        right=0.975,
        top=0.87,
        bottom=0.10,
        wspace=0.18,
        hspace=0.28,
    )

    # ---- 3D close-up ----------------------------------------------------
    axis = figure.add_subplot(grid[0, 0], projection="3d")
    axis.set_facecolor(BACKGROUND)
    colors = [REGION_COLORS[r] for r in region]
    close = (position_np[:, 2] > 18) & (position_np[:, 2] < 32)
    close_indices = np.flatnonzero(close)
    local = -np.ones(n, dtype=int)
    local[close_indices] = np.arange(close_indices.size)
    close_src, close_dst, _ = top_edges(
        weight_np, mask_np, args.edges_per_source
    )
    keep = close[close_src] & close[close_dst]
    close_src = local[close_src[keep]]
    close_dst = local[close_dst[keep]]
    close_positions = position_np[close_indices]
    close_values = weight_np[
        close_indices[close_src], close_indices[close_dst]
    ]
    axis.scatter(
        close_positions[:, 0],
        close_positions[:, 1],
        close_positions[:, 2],
        c=[colors[index] for index in close_indices],
        s=18,
        alpha=0.9,
        edgecolors="none",
    )
    axis.add_collection3d(
        Line3DCollection(
            edge_segments(close_positions, close_src, close_dst),
            colors=[
                (1.0, 0.42, 0.28, 0.28)
                if value >= 0
                else (0.23, 0.78, 0.88, 0.28)
                for value in close_values
            ],
            linewidths=0.5,
        )
    )
    axis.set_xlim(
        close_positions[:, 0].min(), close_positions[:, 0].max()
    )
    axis.set_ylim(
        close_positions[:, 1].min(), close_positions[:, 1].max()
    )
    axis.set_zlim(18, 32)
    axis.set_box_aspect(
        (
            float(np.ptp(close_positions[:, 0])),
            float(np.ptp(close_positions[:, 1])),
            14,
        )
    )
    axis.set_axis_off()
    axis.view_init(elev=18, azim=32)
    axis.set_title(
        f"3D close-up: z 18–32 d̄ ({len(close_indices)} neurons)",
        color=TEXT,
        fontsize=11,
        pad=0,
    )

    # ---- side view x-z --------------------------------------------------
    axis = figure.add_subplot(grid[0, 1])
    axis.set_facecolor(BACKGROUND)
    segments = edge_segments(position_np, src, dst)[:, :, [0, 2]]
    axis.add_collection(
        LineCollection(segments, colors=[(0.75, 0.78, 0.82, 0.11)], linewidths=0.3)
    )
    axis.scatter(
        position_np[:, 0],
        position_np[:, 2],
        c=colors,
        s=8,
        alpha=0.9,
        edgecolors="none",
    )
    axis.axhline(0, color="#74c0fc", lw=1.4)
    axis.axhline(50, color="#ffa94d", lw=1.4)
    axis.text(0.05, 1.2, "front plate", color="#74c0fc", fontsize=8)
    axis.text(0.05, 51.5, "back plate", color="#ffa94d", fontsize=8)
    axis.set_xlim(0, geometry.side)
    axis.set_ylim(-2, 52)
    axis.set_xlabel("x (d̄ units)", color=MUTED, fontsize=8)
    axis.set_ylabel("z (d̄ units)", color=MUTED, fontsize=8)
    axis.tick_params(colors=MUTED, labelsize=7)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")
    axis.set_title(
        "side view: plate separation D = 50 d̄",
        color=TEXT,
        fontsize=11,
        pad=8,
    )

    # ---- top view x-y ---------------------------------------------------
    axis = figure.add_subplot(grid[0, 2])
    axis.set_facecolor(BACKGROUND)
    axis.add_collection(
        LineCollection(
            edge_segments(position_np, src, dst)[:, :, :2],
            colors=[(0.75, 0.78, 0.82, 0.08)],
            linewidths=0.25,
        )
    )
    axis.scatter(
        position_np[:, 0],
        position_np[:, 1],
        c=colors,
        s=12,
        alpha=0.9,
        edgecolors="none",
    )
    axis.set_xlim(0, geometry.side)
    axis.set_ylim(0, geometry.side)
    axis.set_aspect("equal")
    axis.set_xlabel("x (d̄ units)", color=MUTED, fontsize=8)
    axis.set_ylabel("y (d̄ units)", color=MUTED, fontsize=8)
    axis.tick_params(colors=MUTED, labelsize=7)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")
    axis.set_title(
        "top view: 8 contiguous regions",
        color=TEXT,
        fontsize=11,
        pad=8,
    )

    # ---- connection matrix ---------------------------------------------
    axis = figure.add_subplot(grid[1, 0])
    axis.set_facecolor(BACKGROUND)
    signed = np.sign(weight_np) * np.log1p(np.abs(weight_np))
    signed[~mask_np] = np.nan
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad((0, 0, 0, 0))
    image = axis.imshow(
        signed,
        cmap=cmap,
        vmin=-np.nanmax(np.abs(signed)),
        vmax=np.nanmax(np.abs(signed)),
        interpolation="nearest",
        aspect="auto",
    )
    axis.set_xlabel("post neuron", color=MUTED, fontsize=8)
    axis.set_ylabel("pre neuron", color=MUTED, fontsize=8)
    axis.tick_params(colors=MUTED, labelsize=7)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")
    colorbar = figure.colorbar(image, ax=axis, fraction=0.046, pad=0.02)
    colorbar.ax.tick_params(colors=MUTED, labelsize=7)
    colorbar.outline.set_edgecolor("#2b3038")
    axis.set_title(
        "connection matrix: 65,536 / 1,048,576 (6.25%)",
        color=TEXT,
        fontsize=11,
        pad=8,
    )

    # ---- z distribution -------------------------------------------------
    axis = figure.add_subplot(grid[1, 1])
    axis.set_facecolor(BACKGROUND)
    axis.hist(
        position_np[:, 2],
        bins=50,
        color="#74c0fc",
        alpha=0.82,
        edgecolor="#0b0f14",
    )
    axis.axvline(0, color="#74c0fc", lw=1.2)
    axis.axvline(50, color="#ffa94d", lw=1.2)
    axis.axvline(
        float(position_np[:, 2].mean()),
        color="#ffd43b",
        lw=1.0,
        ls="--",
        label=f"mean {position_np[:, 2].mean():.1f}",
    )
    axis.set_xlabel("z (d̄ units)", color=MUTED, fontsize=8)
    axis.set_ylabel("neurons", color=MUTED, fontsize=8)
    axis.tick_params(colors=MUTED, labelsize=7)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")
    axis.set_title(
        "position: uniform slab, no learned coordinates",
        color=TEXT,
        fontsize=11,
        pad=8,
    )

    # ---- edge length ----------------------------------------------------
    axis = figure.add_subplot(grid[1, 2])
    axis.set_facecolor(BACKGROUND)
    axis.hist(
        edge_length,
        bins=50,
        color="#ffa94d",
        alpha=0.82,
        edgecolor="#0b0f14",
    )
    axis.axvline(
        float(edge_length.mean()),
        color="#ffd43b",
        lw=1.0,
        ls="--",
        label=f"mean {edge_length.mean():.2f} d̄",
    )
    axis.set_xlabel("synapse length (d̄ units)", color=MUTED, fontsize=8)
    axis.set_ylabel("synapses", color=MUTED, fontsize=8)
    axis.tick_params(colors=MUTED, labelsize=7)
    axis.legend(frameon=False, labelcolor=MUTED, fontsize=7)
    for spine in axis.spines.values():
        spine.set_color("#2b3038")
    axis.set_title(
        "connections: top-64 distance-weighted edges per source",
        color=TEXT,
        fontsize=11,
        pad=8,
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
        for index in range(int(config.brain.regions))
    ]
    handles.extend(
        [
            Line2D([0], [0], color="#ff8c69", lw=2, label="excitatory"),
            Line2D([0], [0], color="#4dd0e1", lw=2, label="inhibitory"),
        ]
    )
    figure.legend(
        handles=handles,
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        labelcolor=TEXT,
        fontsize=8,
        bbox_to_anchor=(0.5, 0.015),
    )
    figure.suptitle(
        "Cortex-LM stage3 · current reloaded structure · step 21,200",
        color=TEXT,
        fontsize=16,
        x=0.035,
        ha="left",
        y=0.965,
    )
    figure.text(
        0.035,
        0.925,
        (
            f"1024 single-compartment rate neurons · {int(mask.sum()):,} synapses "
            f"(6.25% dense) · {config.brain.ticks_per_token} ticks/token · "
            f"edge length mean {edge_length.mean():.2f} d̄ · D = 50 d̄ · "
            f"3D panel is a local close-up; positions/mask reconstructed from "
            f"W + seed (not persisted)"
        ),
        color=MUTED,
        fontsize=9,
        ha="left",
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(
        output,
        facecolor=BACKGROUND,
        bbox_inches="tight",
        pad_inches=0.12,
    )
    plt.close(figure)

    stats = {
        "checkpoint": args.checkpoint,
        "step": int(payload["step"]),
        "architecture": {
            "neurons": n,
            "compartments": 1,
            "rate_units": True,
            "ticks_per_token": int(config.brain.ticks_per_token),
            "readout": config.brain.readout,
            "regions": int(config.brain.regions),
        },
        "geometry": {
            "mean_neuron_distance": 1.0,
            "plate_distance": 50.0,
            "plate_side": geometry.side,
            "volume": geometry.volume,
            "z_mean": float(position_np[:, 2].mean()),
            "nearest_neighbor_mean": float(nearest.mean()),
        },
        "connections": {
            "edges": int(mask.sum()),
            "density": float(mask.float().mean()),
            "edges_per_neuron": int(mask.sum(dim=1).min()),
            "self_loops": int((all_idx[:, 0] == all_idx[:, 1]).sum()),
            "positive_fraction": float(
                (weight_np[all_idx[:, 0], all_idx[:, 1]] > 0).mean()
            ),
            "edge_length_mean": float(edge_length.mean()),
            "edge_length_max": float(edge_length.max()),
        },
        "caveat": (
            "positions and edge mask are non-persistent buffers; this is the "
            "state after reload, not the exact training trajectory"
        ),
    }
    Path("llm/results/cortex/current_structure.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
