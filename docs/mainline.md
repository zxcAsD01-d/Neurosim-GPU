# Mainline simulation guide

The recommended core is the three-state soma–dendrite–adaptation model introduced
in 0.2.0. The unified runner is a post-release addition; use the current checkout.
The old point-neuron APIs remain compatible baselines, not silently changed aliases.

## What “complete” means

Connected workflow: sparse graph → full three-state updates → matrix or PELS
propagation → explicit reference or native backend → diagnostics and checkpoint resume.
This does **not** automatically include dopamine chemistry, eligibility, structural
plasticity, delayed communication, replay or Cortex-LM. Those extensions have their
own APIs/tests; joint integration with the dendritic native core remains open.

## Choose a path, do not assume a universal fastest one

| Workload | Start with | Why / caveat |
|---|---|---|
| CPU, numerical oracle, training | `DendriticMatrixRuntime` | Portable differentiable reference; no threshold approximation |
| CUDA FP32 inference | `NativeDendriticMatrixRuntime` | Fused one-kernel baseline; requires local JIT toolchain |
| Sparse changes, approximation allowed | `NativeDendriticPELSRuntime` | Two-kernel event route; benchmark activity, latency and trajectory error |
| PELS semantics/debugging | `DendriticPELSRuntime` | Reference delta propagation, not guaranteed event acceleration |

Native classes use reference operations when gradients are needed; `last_backend`
reports the route. Native PELS statistics requests also use the reference path.
The CLI is inference-only. Backend selection is explicit: no hidden autotuner,
no native-to-reference fallback that conceals a build failure, no model switch.

## Runner

```powershell
python -m pip install -e .
python -m neurosim_gpu.simulate --help
python -m neurosim_gpu.simulate --steps 100 --output runs/mainline/cpu.json
```

Defaults: N=1000, 20,000 candidate edges, B=1, seed=0, FP32, CPU reference matrix,
alpha=.72, gain=.8, dt=1, tau_d=4, tau_a=25, coupling=.5, adaptation strength=.2.
Initial soma/dendrite/adaptation are zero. Drives are constant .2 (soma) and .1
(dendrite). Parameters illustrate the model; they are not biological fits.
Duplicates are coalesced, so `actual_edges` may be below `--edges`.

```powershell
scripts\native_env.bat python -m neurosim_gpu.simulate --device cuda --backend native --steps 1000 --output runs/mainline/cuda-matrix.json
scripts\native_env.bat python -m neurosim_gpu.simulate --device cuda --backend native --propagation pels --theta 0.02 --steps 1000 --output runs/mainline/cuda-pels.json
```

Windows uses the helper; on Linux run Python directly after configuring the CUDA/C++
toolchain. First-use compilation is not a latency measurement. The runner only
synchronizes for final diagnostics/checkpointing, not per-tick progress reporting.
Use [the benchmark](../benchmarks/compartmental.py) for warmed, interleaved timing.

Other options: `--neurons`, `--edges`, `--batch`, `--seed`, `--drive`,
`--dendritic-drive`, `--dt`, `--tau-dendrite`, `--tau-adaptation`, `--coupling`
and `--adaptation-strength`. The output stores all supplied configuration plus
actual runtime/backend/device, graph size and three-state statistics. There is no
claim of output accuracy improvement merely because all three states are enabled.

## Activity comparison

The [six-panel figure](../results/compartmental_brainwaves.png) plots measured soma activity
under continued and withdrawn synthetic input using the same graph and warmed state.
The power spectrum uses a 512-tick Hann window with 256 ticks of overlap. The Hann window
downweights the brief withdrawal transient at the start of the observation window.
The 100 Hz frequency axis is a display convention. The [measurements](../results/compartmental_brainwaves.json)
include seeds, input and population traces, analysis settings, and source hashes.

From an installed checkout, the first run below uses the CPU reference. The second uses
the native CUDA matrix path on Windows. Save the reproduced outputs under `runs/`.

```powershell
python -m pip install numpy scipy matplotlib
python experiments/compartmental_brainwaves.py --backend reference --output runs/brainwaves-reference.png
scripts\native_env.bat python experiments/compartmental_brainwaves.py --backend native --output runs/brainwaves-native.png
```

## Resume semantics

```powershell
python -m neurosim_gpu.simulate --steps 50 --checkpoint runs/mainline/state.pt
python -m neurosim_gpu.simulate --steps 50 --resume runs/mainline/state.pt --checkpoint runs/mainline/state.pt --output runs/mainline/resumed.json
```

Reconstruct the same graph (seed, N, candidate edges), model parameters, batch,
propagation and runtime class before restoring. The checkpoint checks graph structure,
model signature and tensor shape/type; it restores weights and all runtime state.
It does not construct a different class or convert matrix checkpoints into PELS.
The JSON records invocation config; arbitrary external input schedules, optimizers,
gradient graphs and their time cursor are not included. The CLI uses constant inputs;
keeping them unchanged makes 50+50 equivalent to 100 continuous ticks (within native
floating-point nondeterminism). `steps_this_run` is not a lifetime tick counter.

## Validation and evidence boundaries

- Reference matrix and theta=0 PELS are compared for all three states.
- Runner tests cover configuration rejection, JSON output and complete checkpoint
  continuation for matrix and nonzero-threshold PELS, including current/emitted state.
- Existing model/native suites cover gradients, broadcasting, long trajectories,
  current-stream behavior, parameter mutation and retained-output validity.
- Native PELS uses FP32 atomics; a nonzero threshold can amplify differences in
  emission timing. It is not a formal trajectory-error bound.
- The [0.2.0 performance report](performance-0.2.0.md) owns the measured three-state
  results. [Older performance](performance.md) and [results](results.md) describe
  point-neuron measurements; do not transfer their training or 50M-edge claims.

[Research/equations](neuron-research-2026.md) · [Build](native-build.md) ·
[Verification](verification.md) · [Architecture/integration](architecture.md)

[← Documentation index](index.md)
