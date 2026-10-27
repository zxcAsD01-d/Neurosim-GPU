# Performance

What the runtime costs, how it scales, and how to measure it on your own hardware.

- **Source of record for accepted numbers:** [results](results.md) (RTX 5070
  Laptop, CUDA 13.1).
- **This document:** the cost model, the scaling story, and a reproduced CPU
  reference baseline measured with `benchmarks/performance_suite.py`.
- **Task accuracy on CPU:** [experiments](experiments.md).

Never mix the two stacks. A CPU ratio is a dispatch-overhead measurement; a GPU
ratio is a kernel measurement.

## The tick

```text
synaptic_current = W_csr @ state              # O(B·M) multiply-adds
state = clamp(alpha·state + (1-alpha)·gain·tanh(current + drive))
```

One sparse matrix multiplication, one elementwise update, one clip. Everything
else in the simulator is either optional per tick or runs at its own cadence.

| Term | Work | Cadence |
|---|---|---|
| propagation + neuron update | `O(B·M + B·N)` | every tick |
| emitted-delta propagation (events) | `O(B·N + M_A)` where `M_A` is active edges | every tick |
| eligibility traces | `O(B·M)` extra | every tick when plasticity is on |
| regional dopamine chemistry | `O(B·(R + E_R))` | every `K_c` ticks |
| delayed communication | `O(B·M)` + ring history | every tick |
| migration | `O(N)` | every `K_m` ticks |
| structural formation/pruning | `O(K)` candidates + rebuild | every `K_s` ticks |

The bounded-error argument, the contraction certificate and the tolerance table
are in [validation](validation.md); the derivation is in [design](design.md).

## Accepted GPU numbers

`raw-spmm` is a same-parameter cuSPARSE linear layer; `canonical-fused` performs
the whole synaptic reduction, `tanh`, leak/gain update and clip in one kernel.
Medians after warmup, FP32, RTX 5070 Laptop 8 GiB.

| N | M | B | raw-spmm | canonical-spmm | complete neural tick | neural/raw |
|---:|---:|---:|---:|---:|---:|---:|
| 166k | 10.00M | 1 | 0.725 ms | 0.558 ms | 0.595 ms | **0.821×** |
| 166k | 24.99M | 1 | 1.820 ms | 1.313 ms | 1.350 ms | **0.742×** |
| 166k | 49.95M | 1 | 3.305 ms | 2.686 ms | 2.466 ms | **0.746×** |
| 50k | 1.00M | 8 | 0.447 ms | 0.460 ms | 0.513 ms | 1.148× |

Relative to the *matching* custom SpMM, the neural dynamics adds 2.9–6.6% at
10–25M edges. At batch one the complete tick is 18–26% faster than the cuSPARSE
denominator because it fuses operations cuSPARSE cannot.

> **What the denominator is.** `raw-spmm` is the PyTorch CSR multiplication
> (`torch.sparse.mm` over the topology's CSR arrays) as invoked by our harness,
> **not** a hand-tuned cuSPARSE kernel. PyTorch's sparse CSR support is beta and
> may not select the fastest library path, so `0.74–0.82×` means "faster than the
> library call we can express from PyTorch", not "faster than cuSPARSE at its
> best". The claim that holds either way is the matching-SpMM comparison: the
> neural update adds 2.8–6.6%.

Event path after 20 warmup ticks at `theta=0.05` (≈2.64% of edges emitting):

| N | M | B | matrix tick | native event tick | speedup |
|---:|---:|---:|---:|---:|---:|
| 166k | 10.00M | 1 | 0.595 ms | 0.200 ms | **3.0×** |
| 166k | 24.99M | 1 | 1.350 ms | 0.301 ms | **4.5×** |
| 50k | 1.00M | 8 | 0.513 ms | 0.123 ms | **4.2×** |

Training (forward + backward, `N=16,384`, `M=998,187`, `B=4`, `T=32`):

| Training graph | Time | Same-parameter ratio |
|---|---:|---:|
| Sparse RNN denominator | 185.6 ms | 1.000× |
| PyTorch matrix dynamics | 209.9 ms | 1.131× |
| Native fused pointwise dynamics | 199.5 ms | **1.075×** |
| PyTorch persistent `theta=0` | 227.4 ms | 1.225× |

The native analytic pointwise backward removes about half of the training
overhead (13.1% → 7.5%). Sparse-weight backward still uses cuSPARSE; a fused
whole-rollout adjoint is not implemented.

## CPU reference baseline

Measured 2026-10-26 with `benchmarks/performance_suite.py` on a 16-thread
x86-64 workstation, `torch 2.14.0+cpu`, FP32, median of 3 repeats × 10 steps
after warmup. Raw payloads: [`../results/performance_cpu.json`](../results/performance_cpu.json).

The same topology, weights, activity and batch are used for every runtime in a
row, so the ratios are directly comparable.

| N | M | B | raw-spmm | matrix tick | matrix/raw | PELS/raw | M synapses/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 2,000 | 16k | 1 | 0.062 ms | 0.110 ms | 1.78× | 3.34× | 145 |
| 8,000 | 64k | 1 | 0.182 ms | 0.287 ms | 1.58× | 2.05× | 223 |
| 8,000 | 256k | 1 | 0.160 ms | 0.187 ms | **1.18×** | 2.44× | 1,363 |
| 32,000 | 256k | 1 | 0.538 ms | 0.741 ms | 1.38× | 1.75× | 346 |
| 32,000 | 1.02M | 1 | 0.754 ms | 1.162 ms | 1.54× | 1.69× | 881 |
| 32,000 | 1.02M | 8 | 1.527 ms | 2.777 ms | 1.82× | 2.77× | 2,948 |

Emitted-event path on CPU (reference implementation, Python scatter):

| N | M | B | theta | active edges | event/raw |
|---:|---:|---:|---:|---:|---:|
| 8,000 | 64k | 1 | 0.02 | 50.5% | 12.1× |
| 8,000 | 64k | 1 | 0.05 | 23.0% | 7.4× |
| 8,000 | 64k | 1 | 0.10 | 11.6% | 7.3× |
| 32,000 | 1.02M | 8 | 0.02 | 50.3% | 80.9× |
| 32,000 | 1.02M | 8 | 0.10 | 11.6% | 15.7× |

Read this table correctly:

- **The matrix tick stays close to a raw sparse multiplication** (1.18–1.82×)
  even though it also does `tanh`, leak, gain, clip and Python dispatch. The
  remaining gap is elementwise work plus per-tick interpreter overhead; it
  shrinks as `M` grows (1.78× at 16k edges → 1.18× at 256k edges).
- **The event path is a semantic reference on CPU, not a speed path.** It
  performs a Python-level `nonzero`/`repeat_interleave`/`index_add` scatter per
  tick, so it is 5–80× slower than the dense path. The 3–4.5× speedup belongs to
  the native CUDA kernel and appears only there.
- Active-edge fractions here (50%/23%/12%) are higher than the GPU run's 2.64%
  because the dynamics parameters and warmup differ; activity is a property of
  the run, not of `theta` alone.

## Memory model

| Representation | Bytes | 125M edges |
|---|---|---|
| Mutable topology (source-major + destination views) | `24M + 8N` | ≈ 3.0 GB |
| Read-only single-weight inference view | `16M + 8N` | ≈ 2.0 GB |
| Native kernel arrays | `16M + 8N` bytes plus row pointers | same as above |

Measured on CPU: 24.2–25.0 bytes per synapse for `N` from 2k to 32k (indices are
`int32`, weights `float32`), e.g. 1.02M synapses = 23.7 MiB. The constants are
structural, not implementation accidents: two edge endpoints, two weight views,
two row-pointer arrays and one permutation. [efficiency](efficiency.md) derives
them.

## Reproducing

```powershell
# CPU reference baseline (no CUDA needed)
python benchmarks/performance_suite.py --device cpu --json results/performance_cpu.json

# GPU parity, event path and auto-policy behaviour
python benchmarks/benchmark.py --neurons 100000 --edges 5000000 --steps 100
scripts\native_env.bat python benchmarks/benchmark.py --native

# Per-tick latency percentiles and deadline misses (CUDA)
scripts\native_env.bat python benchmarks/realtime_latency.py --mode matrix

# Forward + backward parity against a same-parameter sparse RNN
python benchmarks/training_parity.py

# Where does the event path overtake the matrix path? (device-agnostic)
python benchmarks/activity_sweep.py --device cpu --json results/activity_cpu.json
```

Record with every number: device and driver, CUDA toolkit and PyTorch build,
`N`, requested and coalesced `M`, batch, dtype, activity and whether the device
was otherwise idle. [validation](validation.md) defines the protocol.

## Interpretation rules

**Do**

- Compare a runtime against a *same-parameter* baseline on the same device.
- Report coalesced `M` (duplicate edges are merged) and the actual activity.
- Say whether eligibility, chemistry and weight updates are enabled: they change
  the answer by an order of magnitude.
- Treat event sparsity as a forward-propagation property; with eligibility on,
  the runtime still scans all `M` edges per tick.

**Don't**

- Compare CPU ratios with GPU ratios.
- Quote the 3–4.5× event speedup for the Python reference runtime.
- Extrapolate a 50M-edge measurement to 125M without noting that the 125M point
  has not been executed end to end.
- Present the 1.075× training parity as "training is free": it is 93% of a
  same-parameter sparse RNN, and the remaining gap is the missing fused
  whole-rollout adjoint.
