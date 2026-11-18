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
| CUDA matrix inference | `NativeDendriticMatrixRuntime` | Fused one-kernel path with FP32 or compressed weights |
| Sparse changes, approximation allowed | `NativeDendriticPELSRuntime` | Two-kernel event route; benchmark activity, latency and trajectory error |
| PELS semantics/debugging | `DendriticPELSRuntime` | Reference delta propagation, not guaranteed event acceleration |

The FP32 native classes use reference operations when gradients are needed.
`last_backend` reports the route. Native PELS statistics requests use the reference path.
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

## Precision and benchmarks

Set `--precision fp32|fp16|fp8|nvfp4` for inference. The default is FP32.
The JSON fields `precision` and `precision_execution` identify the selected format
and the executed arithmetic. Every reduced-precision mode keeps FP32 master weights
and advances the soma, dendrite and adaptation states in FP32.

| Format | Differentiable reference matrix and PELS | Fused CUDA matrix inference |
|---|---|---|
| FP32 | CPU/CUDA CSR | FP32 weights and FMA |
| FP16 | CUDA Half CSR forward with FP32 sparse backward on the rounded operands | FP16 weight storage, FP32 FMA |
| FP8 | Scaled E4M3 weights and sources, dequantized for FP32 CSR, gradients via a straight-through estimator | E4M3 weight bytes and one FP32 scale, FP32 FMA |
| NVFP4 | E2M1 values in 16-element blocks, E4M3 block scales and an FP32 global scale, dequantized for FP32 CSR, gradients via a straight-through estimator | Two E2M1 weight codes per byte with block/global scales, FP32 FMA |

FP8 and NVFP4 reference quantization runs on CPU and CUDA. Native PELS supports
FP32. Reduced-precision reference PELS requires `propagation="matrix"` in the API
and recomputes the current from the full emitted state each tick. The reference
CLI sets this automatically for `--propagation pels`.

FP16 weights must stay in the finite Half range. The reference FP16 implementation
also returns the sparse multiplication result in Half before casting it to FP32.

Use `DendriticMatrixRuntime(graph, model, initial, precision="fp16")` for CUDA
training. For FP8 or NVFP4 quantization-aware training, set `precision="fp8"` or
`precision="nvfp4"`. The benchmark provides an end-to-end SGD example that
rebuilds the CSR values via a differentiable gather from the FP32 leaf weights
on each optimizer step. Use a fresh runtime for each rollout. [Training example](../benchmarks/precision_compartmental.py)

```powershell
scripts\native_env.bat python -m neurosim_gpu.simulate --device cuda --backend native --precision fp8 --steps 1000
python -m neurosim_gpu.simulate --backend reference --precision nvfp4 --propagation pels --theta 0.02 --steps 100
```

### Fused matrix inference

RTX 5070 Laptop GPU, batch 8. Median synchronized wall time in ms/tick after
warming the extension and preparing the weight cache.

| Format | 1,024 neurons / 16,273 edges | 50,000 neurons / 999,787 edges | Large-graph weight payload (bytes) |
|---|---:|---:|---:|
| FP32 | 0.0221 | 0.4510 | 3,999,148 |
| FP16 | 0.0255 | 0.4480 | 1,999,574 |
| FP8 | 0.0206 | 0.4446 | 999,791 |
| NVFP4 | 0.0249 | 0.4999 | 562,385 |

The payload measures the format consumed by the kernel. These runtimes also retain
the FP32 topology, so total GPU allocation includes that topology and any separate
encoded cache. The JSON reports allocator measurements and three-state trajectory
errors alongside payload sizes. [Fused benchmark samples](../results/precision_native.json)

### Reference inference and training

RTX 5070 Laptop GPU, 4,096 neurons, 65,416 edges, batch 8. Median synchronized wall times
over seven interleaved repeats. Each training step includes an eight-tick rollout,
backward and SGD updates to the sparse weights and six sets of dynamics parameters.

| Format | Matrix inference (ms/tick) | Matrix training (ms/step) | PELS inference (ms/tick) | PELS training (ms/step) |
|---|---:|---:|---:|---:|
| FP32 | 0.407 | 22.217 | 0.602 | 25.205 |
| FP16 | 0.480 | 17.238 | 0.587 | 19.684 |
| FP8 | 0.577 | 31.981 | 0.947 | 35.827 |
| NVFP4 | 1.350 | 46.224 | 1.511 | 49.852 |

FP16 reduced training step time on both reference paths. FP8 and NVFP4
quantization increased it relative to FP32. PELS used `theta=0`.
[Timing samples, allocator measurements, trajectory and gradient errors](../results/precision_compartmental.json)

```powershell
python benchmarks/precision_compartmental.py --neurons 4096 --edges 65536 --batch 8 --iterations 100 --warmup 30 --repeats 7 --train-warmup 3 --train-steps 16 --train-rollout 8 --output runs/mainline/precision-reference.json
scripts\native_env.bat python benchmarks/precision_native.py --output runs/mainline/precision-native.json
```

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
propagation, precision and runtime class before restoring. The checkpoint checks graph structure,
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
