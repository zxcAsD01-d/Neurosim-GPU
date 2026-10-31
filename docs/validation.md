# Validation Protocol

## Test Layers

### Python correctness suite

```powershell
python -m unittest discover -s tests -v
```

The current suite contains 174 tests across sparse dynamics, PELS, error
budgets, dopamine chemistry/receptors/plasticity, delayed communication,
migration, structural connectome updates, the Gomoku environment/learning
probe, and the Cortex-LM pipeline contract. CPU and CUDA are both exercised
where a CUDA device is available; one CUDA-only test skips without a GPU and
eleven Cortex-LM tests skip when the optional `transformers` dependency is
absent.

The focused task-level checks are:

```powershell
python -m unittest tests.test_gomoku tests.test_gomoku_learning -v
python benchmarks\gomoku_learning.py --device cpu --samples 512 --epochs 3 --batch 32 --json
```

The Gomoku runner's validation is intentionally separate from core runtime
parity. It checks legal-action masking, three-plane encoding, source-to-CSR
weight permutation, sparse sensory fanout, finite recurrent gradients, sample
efficiency checkpoints, and p50/p95/p99 decision timing. Its default dataset is
a synthetic tactical supervised task; the optional `random-rollouts` source
uses legal CPU rollouts plus an immediate-win/block/center teacher. Neither
source must be described as self-play strength or dopamine-learning evidence.

### Native CUDA comparison

```powershell
scripts\native_env.bat python benchmarks\validate_native.py
```

This compiles/loads the native extension and compares native versus PyTorch
reference paths for:

- fused matrix dynamics;
- canonical SpMM;
- native PELS event propagation;
- analytic local training backward;
- dopamine matrix and event modulation;
- fused eligibility;
- nonzero dopamine-gated learning;
- source/destination weight views and weight-version changes;
- persistent-current reconstruction.

FP32 event scatter uses atomics and is not bitwise deterministic. Its tolerance
is wider than deterministic FP64/reference paths and must remain explicit.

### Accuracy/activity sweep

```powershell
python benchmarks\accuracy_sweep.py
```

This compares bounded-error PELS against dense FP64 reference state and full
BPTT gradients. It reports state infinity error, gradient cosine, gradient
relative error, and active-edge fraction for each threshold.

### Same-parameter performance

```powershell
scripts\native_env.bat python benchmarks\benchmark.py --native --suite parity
scripts\native_env.bat python benchmarks\training_parity.py --native
```

`raw-spmm` and the sparse RNN are denominators. Topology construction, sorting,
and JIT compilation must be outside timed windows.

The published sub-1.0 parity ratios apply only to the fixed-topology core, not
to the full neurochemical, eligibility, communication, or development stack.

### Neurochemical performance

```powershell
scripts\native_env.bat python benchmarks\dopamine_benchmark.py --native
```

Run this only on an otherwise idle GPU. It separates matrix, portable
chemistry, portable eligibility, native matrix dopamine, and native event
dopamine paths.

### Real-time control latency

```powershell
scripts\native_env.bat python benchmarks\realtime_latency.py --mode matrix --deadline-ms 5
```

This synchronizes every tick and reports p50/p95/p99 and deadline misses. It is
the acceptance test for embodied control loops; aggregate throughput alone is
not sufficient.

## Required Numerical Invariants

| Invariant | Required check |
|---|---|
| theta=0 forward | Matrix and persistent states agree to dtype tolerance |
| theta=0 gradient | Dense and persistent gradients agree numerically |
| emitted residual | `abs(state-emitted) <= theta`; theta=0 is exact |
| persistent current | `current` agrees with `W @ emitted_state` |
| external weight mutation | `apply_source_weights_`/versioned native updates force rebuild; direct tensor writes are invalid |
| signed error budget | all bounds use `abs(W)` |
| topology views | source and destination weights remain identical by permutation |
| exceptional theta | NaN/-inf and +inf with outgoing edges rejected; +inf allowed only for zero-outdegree sources |

## Required Neurochemical Invariants

| Invariant | Required check |
|---|---|
| concentration | finite and non-negative in safe mode |
| tonic baseline | fixed under zero RPE/release |
| diffusion | volume-weighted mass conserved without source/clearance |
| positive RPE | projected burst above neutral |
| negative RPE | source-specific tonic dip, never negative concentration |
| autoreceptor | high concentration attenuates excess burst |
| receptor occupancy | in `[0,1]`, monotonic and finite at zero concentration |
| D1/D2 coupling | burst/dip reverse pathway signs |
| causality | RPE cannot change the transition that produced it |
| transport delay | response begins on the exact delayed tick |
| multirate hold | concentration/modulation constant between boundaries |
| online plasticity | exact shared update restricted to `B=1` |
| weight projection | bounds and configured sign preserved |
| invalid input | rejection occurs before state/trace mutation |

## Communication And Development Invariants

Delayed communication must match direct scatter for delay zero, return zero for
missing history, deliver on the exact delayed tick, isolate batches, preserve
ring behavior after wraparound, and keep transmitter channel independent from
weight sign.

Migration must remain within speed/domain limits and use no implicit pairwise
search. Structural updates may inspect only the explicit candidate list, must
be deterministic, enforce per-source formation caps and type/sign constraints,
and return unique canonical edges.

## Clean Performance Protocol

Before publishing timing data:

1. confirm no unrelated process is using the GPU with `nvidia-smi`;
2. synchronize and release topology-construction cache;
3. warm every path from the same initial state;
4. restart each timed sample from that state;
5. report median and best latency plus activity fraction;
6. report `N`, unique `M`, `B`, dtype, topology bytes, GPU, clocks and power;
7. distinguish processed edges from full-graph-equivalent throughput;
8. never mix projected 125M results with executed scale points.
9. disable strict per-tick validation and statistics only after validating
   inputs outside the timed region.

The reference workstation shared its GPU with an unrelated workload during part
of the measurement campaign. New neurochemical absolute timings are not
publishable until the device is isolated; do not terminate another user's
workload without authorization.

## Release Gates Still Open

- execute a prebuilt 125M-edge topology without construction oversubscription;
- integrate delayed communication, chemistry and development in one scheduler;
- implement optimizer-aware structural-boundary state/history remapping (the
  scheduler's basic topology/eligibility/current transaction is present);
- implement and validate a checkpointed whole-rollout sparse adjoint;
- run long-horizon stability and drift tests;
- calibrate against real connectome, activity, delay and dopamine datasets;
- validate multi-GPU halo and topology synchronization.

Until these gates close, describe the system as a tested functional
neurochemical simulator, not a biologically validated whole brain.

---

[← Documentation index](index.md)
