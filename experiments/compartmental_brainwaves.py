"""Render a six-panel activity comparison for the mainline dendritic model."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import matplotlib
import numpy as np
import scipy
import torch
from scipy.signal import correlate, lfilter, welch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.ticker import PercentFormatter

from neurosim_gpu import (
    DendriticDynamics, DendriticMatrixRuntime, Dynamics,
    NativeDendriticMatrixRuntime, SparseTopology, random_topology,
)

ROOT = Path(__file__).resolve().parents[1]
BACKGROUND = "#0b0f14"
TEXT = "#f1f3f5"
MUTED = "#adb5bd"
CONDITIONS = (("driven", "Continued input", "#4dd0e1"),
              ("zero_input", "Input withdrawn", "#ffa94d"))
BANDS = (("Delta", .5, 4, "#4c6ef5"), ("Theta", 4, 8, "#22b8cf"),
         ("Alpha", 8, 13, "#51cf66"), ("Beta", 13, 30, "#fcc419"),
         ("Gamma", 30, 45, "#ff6b6b"))
SAMPLE_RATE = 100.0  # Display mapping shared with the reference figure.
GROUPS = 8
NEURONS = 1024
CANDIDATE_EDGES = 20_480
WARMUP = 512
TIME_DISPLAY_TICKS = 1500
DETAIL_TICKS = 24
HEATMAP_TICKS = 800


def external_drive(ticks: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Independent innovations filtered into smooth group and shared inputs."""
    def ar1(columns: int, rho: float) -> np.ndarray:
        innovations = rng.standard_normal((ticks, columns))
        return lfilter([math.sqrt(1 - rho**2)], [1, -rho], innovations, axis=0)

    soma = .20 + .035 * ar1(1, .75) + .075 * ar1(GROUPS, .55)
    dendrite = .10 + .08 * ar1(GROUPS, .85)
    return soma.astype(np.float32), dendrite.astype(np.float32)


@torch.inference_mode()
def simulate(ticks: int, seed: int, backend: str) -> tuple[np.ndarray, dict, dict]:
    device = "cuda" if backend == "native" else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("Native mode requires a CUDA-enabled PyTorch and GPU.")
    # Generate and coalesce the graph on CPU for the same graph across backends.
    graph = random_topology(NEURONS, CANDIDATE_EDGES, device="cpu", seed=seed)
    graph_hash = hashlib.sha256()
    for values in (graph.src, graph.dst, graph.weight):
        graph_hash.update(values.numpy().tobytes())
    if device == "cuda":
        graph = SparseTopology.from_edges(
            NEURONS, graph.src.to(device), graph.dst.to(device), graph.weight.to(device)
        )
    model = DendriticDynamics.from_time_constants(
        Dynamics(torch.full((NEURONS,), .72, device=device),
                 torch.full((NEURONS,), .8, device=device)),
        dt=1., tau_dendrite=4., tau_adaptation=25., coupling=.5, adaptation_strength=.2,
    )
    runtime_cls = NativeDendriticMatrixRuntime if backend == "native" else DendriticMatrixRuntime
    runtime = runtime_cls(graph, model, torch.zeros(2, NEURONS, device=device))
    drives = external_drive(WARMUP + ticks, np.random.default_rng(seed + 1))
    plot_inputs = dict(zip(("somatic", "dendritic"), (a[WARMUP:].T.tolist() for a in drives)))
    stimulated = [torch.from_numpy(a).repeat_interleave(NEURONS // GROUPS, dim=1).to(device)
                  for a in drives]
    branch_inputs = [torch.stack((a[WARMUP:], torch.zeros_like(a[WARMUP:])), dim=1)
                     for a in stimulated]
    for tick in range(WARMUP):
        runtime.step(stimulated[0][tick], dendritic_drive=stimulated[1][tick])
    initial_means = {}
    for name in ("state", "dendritic_state", "adaptation_state"):
        value = getattr(runtime, name)
        torch.testing.assert_close(value[0], value[1], rtol=0, atol=0)
        initial_means[name] = float(value[0].mean())

    soma_trace = torch.empty((ticks, 2, NEURONS), device=device)
    for tick in range(ticks):
        soma_trace[tick] = runtime.step(branch_inputs[0][tick], dendritic_drive=branch_inputs[1][tick])
    actual_backend = getattr(runtime, "last_backend", "reference")
    if actual_backend != backend:
        raise RuntimeError(f"Requested {backend}, executed {actual_backend}")
    metadata = {
        "runtime": runtime_cls.__name__, "backend": actual_backend, "device": device,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch": str(torch.__version__), "dtype": "float32", "neurons": NEURONS,
        "actual_edges": graph.n_edges, "candidate_edges": CANDIDATE_EDGES,
        "topology_sha256": graph_hash.hexdigest(), "graph_seed": seed, "input_seed": seed + 1,
        "warmup_ticks": WARMUP, "ticks_per_condition": ticks, "groups": GROUPS,
        "parameters": {"alpha": .72, "gain": .8, "dt": 1., "tau_dendrite": 4.,
                       "tau_adaptation": 25., "coupling": .5, "adaptation_strength": .2},
        "shared_initial_state_means": initial_means,
        "protocol": "Both rows receive identical warmup inputs. At tick 0 the first row continues "
                    "the seeded AR(1) inputs and both drives for the second row become zero.",
    }
    return soma_trace.cpu().numpy(), metadata, plot_inputs


def analyze(trace: np.ndarray) -> dict:
    if not np.isfinite(trace).all() or trace.min() < 0 or trace.max() > 1:
        raise ValueError("Expected finite somatic activity in [0, 1]")
    mean = trace.mean(axis=1, dtype=np.float64)
    centered = mean - mean.mean()
    freq, power = welch(centered, fs=SAMPLE_RATE, window="hann", nperseg=512,
                        noverlap=256, detrend="constant", scaling="density")
    energy = power.sum()
    band_power = {name: float(power[(freq >= low) & (freq < high)].sum() / energy)
                  if energy > 0 else None for name, low, high, _ in BANDS}
    acf = correlate(centered, centered, mode="full", method="fft")[len(mean) - 1:]
    acf = acf / acf[0] if acf[0] > 0 else None
    below = np.flatnonzero(np.abs(acf) < 1 / math.e) if acf is not None else []
    subset = trace[::2].astype(np.float64)
    subset -= subset.mean(axis=0)
    # Squared singular values are the eigenvalues of the neuron-by-neuron Gram matrix.
    eigenvalues = np.maximum(np.linalg.eigvalsh(subset.T @ subset), 0)
    positive = eigenvalues[eigenvalues > 0]
    rank = None
    if positive.size:
        probabilities = positive / positive.sum()
        rank = float(np.exp(-np.sum(probabilities * np.log(probabilities))))
    group_trace = trace.reshape(len(mean), GROUPS, -1).mean(2).T
    return {
        "mean_activity": float(mean.mean()), "population_mean_std": float(mean.std()),
        "mean_last_512_ticks": float(mean[-512:].mean()),
        "first_lag_below_abs_inv_e": int(below[0]) if len(below) else None,
        "effective_rank": rank, "relative_band_power": band_power,
        "frequency_reference_hz": freq.tolist(), "psd": power.tolist(),
        "autocorrelation_lags_0_to_300": acf[:301].tolist() if acf is not None else None,
        "population_mean_trace": mean.tolist(), "somatic_group_mean_traces": group_trace.tolist(),
    }


def plot(result: dict, output: Path) -> None:
    metadata = result["simulation"]
    analysis = result["analysis"]
    measurements = result["measurements"]
    figure, axes = plt.subplots(2, 3, figsize=(18, 10), dpi=170, facecolor=BACKGROUND)
    figure.subplots_adjust(left=.048, right=.977, top=.861, bottom=.09, wspace=.28, hspace=.35)
    activity, spectrum, acf, bands, heat, summary = axes.flat
    for axis in axes.flat:
        axis.set_facecolor(BACKGROUND)
        axis.tick_params(colors=MUTED, labelsize=8)
        for spine in axis.spines.values():
            spine.set_color("#2b3038")
        axis.xaxis.label.set_color(MUTED)
        axis.yaxis.label.set_color(MUTED)
    for _, low, high, color in BANDS:
        spectrum.axvspan(low, high, color=color, alpha=.085, lw=0)
    acf.axhline(0, color=MUTED, lw=.6)
    acf.axhline(1 / math.e, color=MUTED, ls=":", lw=.7)
    detail = activity.inset_axes([.56, .12, .40, .28], facecolor=BACKGROUND)
    detail.set_title(f"First {DETAIL_TICKS} ticks", color=MUTED, fontsize=7, pad=2)
    detail.tick_params(colors=MUTED, labelsize=7, length=2)
    detail.set_xticks([0, 10, 20])
    detail.set_yticks([0, .2])
    for spine in detail.spines.values():
        spine.set_color("#464d58")
    for index, (key, label, color) in enumerate(CONDITIONS):
        metric = measurements[key]
        population = metric["population_mean_trace"]
        activity.plot(population[:analysis["time_domain_display_ticks"]], color=color, lw=1., label=label)
        detail.plot(population[:analysis["detail_display_ticks"]], color=color, lw=1.)
        frequency, power = map(np.asarray, (metric["frequency_reference_hz"], metric["psd"]))
        valid = (frequency > 0) & (power > 0)
        spectrum.loglog(frequency[valid], power[valid], color=color, lw=1.2, label=label)
        autocorrelation = metric["autocorrelation_lags_0_to_300"]
        if autocorrelation is not None:
            acf.plot(autocorrelation, color=color, lw=1.2, label=label)
        shares = [metric["relative_band_power"][name] for name, *_ in BANDS]
        if all(value is not None for value in shares):
            bands.bar(np.arange(len(BANDS)) + (index - .5) * .36, shares,
                      width=.36, color=color, label=label)
    for axis in (activity, spectrum, acf, bands):
        handles, labels = axis.get_legend_handles_labels()
        if handles:
            axis.legend(handles, labels, labelcolor=MUTED, fontsize=8, frameon=False)
    activity.set(xlabel="Tick from branch point", ylabel="Mean somatic activity")
    spectrum.set(xlim=(.195, 50), xlabel="Frequency (Hz, display reference)", ylabel="Welch PSD")
    acf.set(xlim=(0, 300), xlabel="Lag (ticks)", ylabel="Normalized correlation")
    bands.set_xticks(np.arange(len(BANDS)), [name for name, *_ in BANDS])
    bands.set_ylabel("Share of total spectral power")
    bands.yaxis.set_major_formatter(PercentFormatter(xmax=1.))
    withdrawn_shares = measurements["zero_input"]["relative_band_power"].values()
    if all(value is not None and value < 1e-5 for value in withdrawn_shares):
        bands.text(.98, .80, "Input withdrawn: <0.001% per band", transform=bands.transAxes,
                   color=CONDITIONS[1][2], ha="right", va="top", fontsize=8)

    samples = analysis["heatmap_display_ticks"]
    rows = [np.asarray(measurements[key]["somatic_group_mean_traces"])[:, :samples]
            for key, _, _ in CONDITIONS]
    stacked = np.concatenate((rows[0], np.full((1, samples), np.nan), rows[1]))
    image = heat.imshow(stacked, cmap="magma", aspect="auto", interpolation="nearest",
                        norm=Normalize(vmin=0., vmax=float(np.nanmax(stacked))))
    heat.set_yticks([3.5, 12.5], ["Continued\ninput", "Input\nwithdrawn"])
    heat.set_xlabel("Tick from branch point")
    bar = figure.colorbar(image, ax=heat, fraction=.046, pad=.025)
    bar.ax.tick_params(colors=MUTED, labelsize=8)
    bar.outline.set_edgecolor("#2b3038")

    def metric_pair(name: str, spec: str) -> str:
        values = [measurements[key][name] for key, _, _ in CONDITIONS]
        return "  /  ".join(format(value, spec) if value is not None else "n/a" for value in values)

    summary.axis("off")
    summary.text(0, .96, "Continued input  /  Input withdrawn", color=TEXT, fontsize=10, va="top")
    rows = [("Ticks per condition", str(metadata["ticks_per_condition"])),
            ("Mean activity", metric_pair("mean_activity", ".3e")),
            ("Population mean SD", metric_pair("population_mean_std", ".3e")),
            ("Last 512 ticks (mean)", metric_pair("mean_last_512_ticks", ".3e")),
            ("First |ACF| < 1/e", metric_pair("first_lag_below_abs_inv_e", "d") + " ticks"),
            ("Effective rank", metric_pair("effective_rank", ".1f"))]
    summary.text(0, .84, "\n".join(f"{label:23s} {value}" for label, value in rows),
                 color=TEXT, fontsize=8.6, family="monospace", va="top", linespacing=1.55)
    summary.text(0, .36, "Both conditions begin from a shared warmed state.\n"
                 "Hann windowing downweights the brief decay at onset.\n"
                 "Heatmaps use eight equally sized index groups.\n"
                 "EEG band labels use a 100 Hz display reference.\n"
                 "All curves come from simulated rate activity.",
                 color=MUTED, fontsize=9, va="top", linespacing=1.5)
    titles = ("Population activity over time", "Population power spectrum",
              "Autocorrelation and decay", "Relative power by reference band",
              "Eight-group somatic activity", "Measurement summary")
    for axis, title in zip(axes.flat, titles):
        axis.set_title(title, color=TEXT, fontsize=12, pad=9)
    figure.suptitle("NeuroSim GPU | Mainline population activity", x=.048, ha="left",
                    y=.965, color=TEXT, fontsize=19, fontweight="medium")
    figure.text(.048, .914, f"v0.2.0 soma + dendrite + adaptation  |  {metadata['runtime']}  |  "
                f"{metadata['neurons']:,} neurons  |  {metadata['actual_edges']:,} synapses",
                color=MUTED, fontsize=10)
    figure.text(.048, .025, "Synthetic input protocol  •  1 tick = 10 ms for display only  •  "
                "Frequency and autocorrelation summarize the full observation window",
                color=MUTED, fontsize=9)
    figure.savefig(output, facecolor=BACKGROUND)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=("reference", "native"), default="reference")
    parser.add_argument("--ticks", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument("--output", type=Path, default=ROOT / "results/compartmental_brainwaves.png")
    args = parser.parse_args()
    if args.ticks < 1024 or args.seed < 0 or args.output.suffix.lower() != ".png":
        parser.error("requires at least 1024 ticks, a nonnegative seed and a .png output")
    torch.set_num_threads(1)
    trace, metadata, drives = simulate(args.ticks, args.seed, args.backend)
    result = {
        "simulation": metadata,
        "analysis": {"frequency_display_fs_hz": SAMPLE_RATE, "welch_window": "hann",
                     "welch_detrend": "constant", "welch_scaling": "density",
                     "welch_segment_ticks": 512, "welch_overlap_ticks": 256,
                     "effective_rank_stride": 2, "time_domain_display_ticks": min(args.ticks, TIME_DISPLAY_TICKS),
                     "detail_display_ticks": DETAIL_TICKS, "heatmap_display_ticks": HEATMAP_TICKS,
                     "versions": {"numpy": np.__version__, "scipy": scipy.__version__,
                                  "matplotlib": matplotlib.__version__},
                     "frequency_bands_reference_hz": {name: [lo, hi] for name, lo, hi, _ in BANDS},
                     "band_fraction_denominator": "sum of all one-sided Welch PSD bins",
                     "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                       for name in ("experiments/compartmental_brainwaves.py",
                                                    "src/neurosim_gpu/compartmental.py",
                                                    "src/neurosim_gpu/native_compartmental.py",
                                                    "src/neurosim_gpu/topology.py",
                                                    "csrc/compartmental.cu")}},
        "continued_input_group_traces": drives,
        "measurements": {key: analyze(trace[:, index]) for index, (key, _, _) in enumerate(CONDITIONS)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    plot(result, args.output)
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n",
                                               encoding="utf-8")
    print(f"{metadata['runtime']}  N={metadata['neurons']}  M={metadata['actual_edges']}  ticks={args.ticks}")
    for key, _, _ in CONDITIONS:
        row = result["measurements"][key]
        print(f"{key} mean={row['mean_activity']:.6g} tail={row['mean_last_512_ticks']:.6g} "
              f"acf_lag={row['first_lag_below_abs_inv_e']} rank={row['effective_rank']}")
    print(f"Wrote {args.output} and {args.output.with_suffix('.json')}")


if __name__ == "__main__":
    main()
