"""Plot measured low-precision drift against each backend's FP32 trajectory.

Example: python benchmarks/plot_precision_drift.py --input results/precision_drift.json
Outputs one three-state-by-scenario figure per backend.
Uses matplotlib; tick counts are simulation updates, not biological time.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D


STATES = ("state", "dendritic_state", "adaptation_state")
STATE_LABELS = {"state": "Soma", "dendritic_state": "Dendrite",
                "adaptation_state": "Adaptation"}
PRECISIONS = ("fp16", "fp8", "nvfp4")
COLORS = {"fp16": "#0072B2", "fp8": "#D55E00", "nvfp4": "#009E73"}
SCENARIO_ORDER = {"stable_drive": 0, "slow_adaptation": 1, "recurrent_stress": 2}


def read_cases(path: Path):
    """Return validated {backend, scenario, seed, samples} measurement cases."""
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if "samples_file" in payload:
        samples_path = path.parent / payload["samples_file"]
        compressed = samples_path.read_bytes()
        digest = payload.get("samples_sha256")
        if digest is not None and hashlib.sha256(compressed).hexdigest() != digest:
            raise ValueError("compressed samples SHA256 does not match summary")
        full_payload = json.loads(gzip.decompress(compressed))
        if payload.get("source_sha256") != full_payload.get("source_sha256"):
            raise ValueError("summary and compressed samples have different source hashes")
        payload = full_payload
    cases = payload["cases"]
    if not isinstance(cases, list) or not cases:
        raise ValueError("input must contain a non-empty cases list")
    for case in cases:
        samples = case["samples"]
        ticks = [row["tick"] for row in samples]
        if not ticks or ticks[0] < 0 or any(b <= a for a, b in zip(ticks, ticks[1:])):
            raise ValueError("sample ticks must be non-negative and strictly increasing")
        for row in samples:
            statistics = [row["precisions"][precision] for precision in PRECISIONS]
            if "fp32_other_backend" in row:
                statistics.append(row["fp32_other_backend"])
            for values in statistics:
                for state in STATES:
                    rmse = values[state]["rmse"]
                    if not np.isfinite(rmse) or rmse < 0:
                        raise ValueError("RMSE must be finite and non-negative")
    return payload, cases


def draw(cases, output: Path, title: str, note: str):
    groups = defaultdict(list)
    for case in cases:
        groups[(case["backend"], case.get("scenario", case.get("name", "scenario")))].append(case)
    keys = sorted(groups, key=lambda key: (key[0], SCENARIO_ORDER.get(key[1], 99), key[1]))
    if not keys:
        raise ValueError("no cases match the selected filters")
    with plt.rc_context({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 10,
        "axes.labelsize": 10, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.facecolor": "white",
    }):
        figure, axes = plt.subplots(3, len(keys), squeeze=False, sharex="col", sharey="row",
                                    figsize=(max(7.4, 3.5 * len(keys)), 8.6))
        state_ceiling = {}
        for state in STATES:
            observed = []
            for case in cases:
                for row in case["samples"]:
                    observed.extend(row["precisions"][precision][state]["rmse"]
                                    for precision in PRECISIONS)
                    if "fp32_other_backend" in row:
                        observed.append(row["fp32_other_backend"][state]["rmse"])
            state_ceiling[state] = max(max(observed, default=0.0) * 1.08, 1e-7)
        for column, key in enumerate(keys):
            group = groups[key]
            ticks = np.asarray([row["tick"] for row in group[0]["samples"]])
            for case in group[1:]:
                if [row["tick"] for row in case["samples"]] != ticks.tolist():
                    raise ValueError("seed cases in each group must use the same sampled ticks")
            for row_index, state in enumerate(STATES):
                axis = axes[row_index, column]
                for precision in PRECISIONS:
                    values = np.asarray([[row["precisions"][precision][state]["rmse"]
                                          for row in case["samples"]] for case in group])
                    axis.plot(ticks, np.median(values, axis=0), color=COLORS[precision],
                              linewidth=1.7, label=precision.upper())
                    if len(group) > 1:
                        axis.fill_between(ticks, values.min(axis=0), values.max(axis=0),
                                          color=COLORS[precision], alpha=0.13, linewidth=0)
                if all("fp32_other_backend" in row for case in group for row in case["samples"]):
                    control = np.asarray([[row["fp32_other_backend"][state]["rmse"]
                                           for row in case["samples"]] for case in group])
                    axis.plot(ticks, np.median(control, axis=0), color="#6B6F76",
                              linewidth=1.5, linestyle="--", zorder=4)
                    if len(group) > 1:
                        axis.fill_between(ticks, control.min(axis=0), control.max(axis=0),
                                          color="#6B6F76", alpha=0.08, linewidth=0)
                axis.set_xscale("symlog", linthresh=1)
                axis.set_yscale("symlog", linthresh=1e-7)
                axis.grid(True, which="major", color="#D6D9DE", alpha=0.7, linewidth=0.6)
                axis.tick_params(labelsize=9)
                axis.set_ylim(0, state_ceiling[state])
                axis.set_xlim(left=max(0, ticks[0]), right=ticks[-1])
                if column == 0:
                    axis.set_ylabel(f"{STATE_LABELS[state]} RMSE")
                if row_index == 0:
                    backend = key[0].replace("_", " ").capitalize()
                    scenario = key[1].replace("_", " ")
                    axis.set_title(f"{backend}: {scenario}\n{len(group)} seed(s)")
                if row_index == 2:
                    axis.set_xlabel("Simulation tick")
        figure.suptitle(title, y=0.995, fontsize=14)
        figure.text(0.5, 0.952, note, ha="center", va="top", fontsize=9, color="#40454D")
        handles = [Line2D([0], [0], color=COLORS[mode], lw=2, label=mode.upper())
                   for mode in PRECISIONS]
        if all("fp32_other_backend" in row for case in cases for row in case["samples"]):
            handles.append(Line2D([0], [0], color="#6B6F76", lw=1.5, linestyle="--",
                                  label="FP32 other backend"))
        figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.925),
                      ncol=len(handles), frameon=False)
        figure.text(0.5, 0.018, "Sampled RMSE against FP32 within each backend; line = seed median, band = seed range.\n"
                    "Samples do not bound errors between checkpoints. Symlog axes retain zero.\n"
                    "States remain FP32; ticks do not imply biological seconds.",
                    ha="center", va="bottom", fontsize=9, color="#40454D")
        figure.tight_layout(rect=(0, 0.10, 1, 0.89))
        output.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(output, dpi=180)
        plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=Path("results/precision_drift.json"))
    parser.add_argument("--output", type=Path, default=Path("results/precision_drift"),
                        help="output basename; writes BASENAME_native.png and BASENAME_reference.png")
    parser.add_argument("--backend", help="include one backend; by default show all")
    parser.add_argument("--scenario", help="include one scenario; by default show all")
    args = parser.parse_args()
    _, cases = read_cases(args.input)
    selected = [case for case in cases
                if (args.backend is None or case["backend"] == args.backend)
                and (args.scenario is None or case.get("scenario", case.get("name")) == args.scenario)]
    if not selected:
        parser.error("no cases match the selected filters")
    for backend in sorted({case["backend"] for case in selected}):
        output = args.output.with_name(args.output.stem + f"_{backend}.png")
        if backend == "native":
            note = "Native matrix: compressed weights decoded for FP32 multiply-adds."
        else:
            note = "Reference matrix: FP16 CSR for FP16; FP8/NVFP4 quantize/dequantize into FP32 CSR."
        draw([case for case in selected if case["backend"] == backend], output,
             "Long-horizon low-precision trajectory drift", note)
        print(f"wrote {output}")


if __name__ == "__main__":
    main()
