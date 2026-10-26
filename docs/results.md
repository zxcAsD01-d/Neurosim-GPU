# Measured results (runtime v0.3, 2026-09-27)

Hardware: NVIDIA GeForce RTX 5070 Laptop GPU (8 GiB), CUDA 13.1 toolkit,
PyTorch 2.11 CUDA 13.0, FP32 runtime. Timings are medians after warmup.

`Runtime v0.3` labels the measurement snapshot; the package in this repository is
version 0.1.0. Interpretation rules, the cost model and the reproduced CPU
reference baseline are in [performance](performance.md).

## Correctness

- Dense, CSR and `theta=0` persistent forward paths match.
- Dense BPTT and `theta=0` persistent gradients match to numerical precision.
- Destination-major single-weight SpMM and its fused dynamics kernel pass a
  30-step comparison with the cuSPARSE reference.
- The native training advance and its analytic state/current gradients match
  PyTorch autograd.
- Native event state is checked against the reference and its persistent current
  is independently checked against `W @ emitted_state`.
- Signed-weight current budgets use `abs(W)` and pass their allocation test.
- The historical core-runtime table was produced with 90 passing tests; the
  current complete Python suite contains 107 passing tests. Native CUDA
  validation additionally covers modulated matrix/event paths, analytic local
  backward, fused eligibility, nonzero dopamine learning, both weight views and
  topology-version invalidation.

## Fixed-Topology Core Forward Parity

`raw-spmm` is a same-parameter cuSPARSE linear layer. `canonical-fused` performs
the full synaptic reduction, tanh, leak/gain update and clip in one kernel.

| N | M | B | raw-spmm | canonical-spmm | complete neural tick | neural/raw | Logical slim inference topology |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 166k | 10.00M | 1 | 0.725 ms | 0.558 ms | 0.595 ms | 0.821x | 153.8 MiB |
| 166k | 24.99M | 1 | 1.820 ms | 1.313 ms | 1.350 ms | 0.742x | 382.6 MiB |
| 166k | 49.95M | 1 | 3.305 ms | 2.686 ms | 2.466 ms | 0.746x | 763.5 MiB |
| 50k | 1.00M | 8 | 0.447 ms | 0.460 ms | 0.513 ms | 1.148x | 15.6 MiB |

These measurements exclude dopamine chemistry, eligibility, delayed
communication, migration and structural rebuilding. At batch one the complete
fixed-topology neural tick is 18-26% faster than the same-parameter
cuSPARSE denominator. Relative to the matching custom SpMM, neural dynamics adds
approximately 2.9-6.6% at 10-25M edges. At batch eight and lower mean degree,
the complete tick remains within 15% of cuSPARSE.

The projected read-only inference representation uses one destination-major
FP32 weight array plus source-major event permutations: approximately
`16M + 8N` bytes. For 125M edges this is about 2.0 GB in decimal units. Current
online learning retains dual weight views and is closer to `24M + 8N`, about
3.0 GB at 125M. The measured 18.5-20.3
billion edge/s fused throughput projects 125M edges to approximately 6.2-6.8
ms/tick on this laptop GPU. This is a projection; the largest executed graph is
49.95M edges because random topology construction has a much larger temporary
memory peak than loading a prebuilt graph.

## Event Path

After 20 warmup ticks with `theta=0.05`, about 2.64% of edges emitted per tick.

| N | M | B | Complete matrix tick | Native event tick |
|---:|---:|---:|---:|---:|
| 166k | 10.00M | 1 | 0.595 ms | 0.200 ms |
| 166k | 24.99M | 1 | 1.350 ms | 0.301 ms |
| 50k | 1.00M | 8 | 0.513 ms | 0.123 ms |

The event path remains faster despite its weight-permutation lookup. A fully
device-side matrix/event policy is still required for production switching.

## Training Parity

Full forward/backward benchmark: `N=16,384`, `M=998,187`, `B=4`, `T=32`.

| Training graph | Time | Same-parameter ratio | Peak graph memory |
|---|---:|---:|---:|
| Sparse RNN denominator | 185.6 ms | 1.000x | 1341.4 MiB |
| PyTorch matrix dynamics | 209.9 ms | 1.131x | 1341.4 MiB |
| Native fused pointwise dynamics | 199.5 ms | 1.075x | 1341.4 MiB |
| PyTorch persistent theta=0 | 227.4 ms | 1.225x | 1341.4 MiB |

The native analytic pointwise backward reduces dynamics overhead from 13.1% to
7.5%, or about 93.0% same-parameter training efficiency. Sparse weight backward
still uses PyTorch/cuSPARSE; a fused rollout adjoint is not yet implemented.

## Accuracy And Activity

Small-network sweep: `N=128`, `M≈1,500`, `T=128`, FP64 reference.

| theta | Maximum state error | Gradient cosine | Gradient relative error | Active edge fraction |
|---:|---:|---:|---:|---:|
| 0 | 1.11e-16 | 1.000000 | 1.11e-15 | 100.0% |
| 0.001 | 2.03e-4 | 0.999957 | 0.00993 | 51.0% |
| 0.005 | 1.07e-3 | 0.999693 | 0.0311 | 28.5% |
| 0.010 | 2.09e-3 | 0.999435 | 0.0513 | 20.0% |
| 0.020 | 4.45e-3 | 0.998776 | 0.0869 | 12.3% |
| 0.050 | 9.66e-3 | 0.993234 | 0.222 | 5.3% |

Production thresholds must come from the implemented contraction certificate
or empirical calibration on the actual topology and task.

## Biological Mechanism Expansion

The current functional model additionally implements and tests:

- non-negative regional dopamine with tonic fixed point, source-resolved bursts
  and dips, volume-weighted diffusion, uptake/decay and autoreceptor feedback;
- D1/D2 Hill occupancy, configurable pathway coupling, TD-RPE and independent
  external release channels;
- causal multirate chemistry, transport-delay buckets, bounded eligibility and
  magnitude-based sign-preserving three-factor plasticity;
- portable matrix/persistent runtimes plus native fused matrix/event,
  eligibility and weight-update kernels;
- delayed multi-transmitter communication with exact integer conduction delay;
- bounded 3-D migration and candidate-only structural formation/pruning.

Absolute neurochemical performance numbers are intentionally not recorded in
this table yet: during the 2026-09-27 session the measurement device was shared
with an unrelated GPU workload. Native/reference numerical validation passed
under that load; a clean isolated benchmark remains required before publishing
latency ratios.
The event kernel sparsifies propagation only; default full eligibility tracking
still scans M edges and is not included in the earlier event-latency table.

## Gomoku CNN Versus Sparse Brain (Diagnostic)

The task-level probe is implemented in `benchmarks/gomoku_learning.py` and
specified by `gomoku-protocol.md`. It is a supervised tactical
upper-bound experiment, not self-play Elo and not dopamine/eligibility
learning. Both rows use the same three-plane 9x9 observations, legal-action
masking, policy/value loss, AdamW updates, and validation split. The brain row
uses a recurrent sparse CSR graph (`K=4`, recurrent `M=480`) and a fixed sparse
sensory fanout (`F*fanout=972`); its frozen scalar/storage costs are reported
separately.

One low-load RTX 5070 Laptop GPU smoke (64 positions, one epoch, batch 16,
independent train/validation seeds, `--parameter-match effective`) produced the
following diagnostic values:

| Model | Trainable scalars | Effective scalars | Optimizer bytes | Train positions/s | Val policy top-1 | Decision p50/p99 ms |
|---|---:|---:|---:|---:|---:|---:|
| CNN | 6,892 | 6,892 | 55,168 | 35.8 | 7.69% | 1.317 / 2.309 |
| Sparse brain, fixed sensory | 5,810 | 6,910 | 46,500 | 79.2 | 0% | 3.763 / 6.328 |

The run is intentionally too small for a learning claim. It shows that a
small sparse-rate loop can process more optimizer samples per second in this
implementation, while its policy accuracy and p99 decision latency are worse.
The brain's sparse CSR tensor is rebuilt in Python each forward and the random
reservoir has no convolutional spatial prior; neither number should be
extrapolated to the full-brain target. The `--brain-learn-sensory` ablation,
independent game splits, multiple seeds, PELS, and dopamine online learning
remain separate follow-up tracks.

## Remaining Work

1. Load a prebuilt native topology and execute the full 125M-edge target.
2. Implement a device-side matrix/event hybrid policy without host synchronization.
3. Implement a checkpointed whole-rollout adjoint and fused sparse backward.
4. Add CUDA graph capture and tune high-batch/low-degree kernels.
5. Add global `C/D/E` tensors and validate their causal phase order.

## Measurement Provenance

The accepted tables were produced on 2026-09-27 with driver 616.56, PyTorch
2.11 CUDA 13.0, CUDA toolkit 13.1, default benchmark seed 3, native canonical
blocks 4096, and event scatter blocks 128. Representative commands were:

```powershell
scripts\native_env.bat python benchmarks\benchmark.py --neurons 166000 --edges 10000000 --steps 100 --warmup 20 --theta 0.05 --native
scripts\native_env.bat python benchmarks\benchmark.py --neurons 166000 --edges 25000000 --steps 100 --warmup 20 --theta 0.05 --native
scripts\native_env.bat python benchmarks\benchmark.py --neurons 166000 --edges 50000000 --steps 30 --warmup 10 --samples 3 --native --suite parity
scripts\native_env.bat python benchmarks\training_parity.py --neurons 16384 --edges 1000000 --batch 4 --steps 32 --samples 7 --native
```

Raw per-sample logs and a stable power-profile record were not persisted, so
these measurements are prototype evidence rather than publication-grade data.
The clean protocol in `validation.md` is mandatory for the next result set.

---

[← Documentation index](index.md)
