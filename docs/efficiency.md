# Efficiency Contract, Evidence, and Optimization Plan

## Scope

The performance objective is a neural simulation whose dominant fast-tick cost
is comparable to a sparse model with the same `N` state values and `M` learned
connections. The comparison is meaningful only when the numerator and
denominator use the same graph, batch, precision, hardware, and timing method.

The current evidence supports three narrow conclusions:

1. The fixed-connectome neural forward tick has reached same-parameter sparse
   matrix parity at the tested points.
2. Full forward plus backward is close to, but has not yet reached, the stated
   `1.05x` training target.
3. Dopamine, eligibility, delayed communication, migration, and structural
   development are implemented and tested, but no clean integrated biological
   latency result is published yet.

No result in this file is evidence of biological validity. It is an accounting
of computational cost and measured implementation performance.

For GPU-independent screening, run:

```powershell
python benchmarks\theoretical_cost.py --neurons 166000 --edges 125000000 --activity 0.05 --json
# Add --communication --eligibility --plasticity --development for the full embodied envelope.
```

This is a work/byte model only. It performs no CUDA allocation and does not
claim a latency in milliseconds. It is useful for rejecting configurations
whose eligibility, delay history, or slow-boundary work dominates before using
hardware time.

### GPU-free reference envelope (2026-10-11)

For `N=166,000`, `M=125,000,000`, `B=1`, `R=256`, `S=4`, `K_c=10`,
`K_p=10`, 5% active edges, and the default delay/candidate settings:

| Quantity | Theoretical value |
|---|---:|
| core matrix work units | 125,166,000 |
| core event propagation work units | 6,416,000 |
| event/matrix propagation ratio | 0.0514 |
| slim read-only inference topology | 2,001,328,008 bytes |
| mutable online topology | 3,001,328,008 bytes |
| estimated core persistent state | 3,005,314,120 bytes |
| selected full embodied state lower bound (target compact communication) | 6,513,282,120 bytes |
| full event/matrix work ratio with all optional terms | 0.6860 |

The last ratio includes full eligibility, communication, plasticity,
migration, and structural terms. It demonstrates why an event-sparse synaptic
propagator does not by itself guarantee an event-sparse continual learner.
These values come from `CostModel`, not a GPU run. The selected-state number is a
lower bound and omits receptor-signal tensors, RPE accumulators, event buffers,
candidate buffers, and allocator/checkpoint transients. The full-state number
assumes
compact int32 communication metadata; the current portable communication module
uses int64 edge metadata and would add roughly 2 GB at 125M edges before other
temporary/receptor allocations. It is therefore an optimization target, not a
claim that the full model fits an 8 GiB device.

## Symbols

| Symbol | Meaning |
|---|---|
| `N` | neuron count |
| `M` | fixed synapse count |
| `B` | batch count |
| `M_A` | outgoing edges expanded from active `(batch, source)` pairs |
| `R` | chemical region count |
| `E_R` | undirected edges in the regional diffusion graph |
| `S` | dopamine source-population count |
| `K_c` | neural ticks per chemistry update |
| `K_p` | chemistry updates per plasticity update |
| `D_da` | dopamine transport-delay slots |
| `D` | maximum integer axonal delay |
| `C` | transmitter-channel count |
| `K` | explicitly supplied structural candidate-edge count |
| `L_m` | neural ticks per migration update |
| `L_s` | neural ticks per structural-connectome update |
| `T` | rollout length used by a training measurement |

Unless stated otherwise, complexity is for FP32 neural state and weights. Graph
construction, validation, statistics collection, and allocator transients are
excluded from fast-tick formulas and called out separately.

## Parity Ratios

Ratios below are latency ratios. Lower is better, and `1.0x` is equal latency.

### Core forward parity

```text
R_forward = time(complete neural tick) / time(raw cuSPARSE SpMM)
```

The denominator is `W @ state` for the same destination-major CSR graph. The
numerator is the native canonical fused kernel: sparse reduction plus `tanh`,
leak, gain, drive, and clipping. A value below `1.0x` means that the custom fused
kernel is faster than the selected cuSPARSE call at that shape. It does not mean
that the extra neural operations have zero mathematical cost.

### Fused dynamics overhead

```text
R_dynamics = time(canonical fused neural tick) /
             time(canonical custom SpMM)
```

This is the cleanest local measure of the pointwise neural update because both
paths use the custom destination-major sparse reduction. It is shape-specific.

### Event-path ratio

```text
R_event = time(native fixed event tick) /
          time(native complete matrix tick)
```

This compares two inference paths with the same initial state and graph. At
`theta>0`, their trajectories can diverge after the first omitted emission. The
ratio must therefore be reported with `theta`, active-edge fraction, and a state
or task-error metric against the matrix path. The recorded result uses a forced
event path, not an automatic production policy.

### Training parity

```text
R_train = time(neural rollout forward + backward) /
          time(same-parameter sparse RNN forward + backward)
```

The denominator is a length-`T` recurrence
`state = tanh(W @ state + drive)` with the same sparse weights. The numerator
adds the neural leak/gain/clip semantics. Both include sparse-weight gradients.
Training efficiency is `1 / R_train`; for example, `1.075x` is about `93.0%`.

### Functional-system ratios

Future neurochemical results must use two distinct ratios:

```text
R_da_matrix = time(native dopamine matrix runtime) /
              time(native canonical matrix runtime)

R_da_event = time(native dopamine event runtime) /
             time(native dopamine matrix runtime)
```

Both sides must share `N`, `M`, `B`, topology, initial state, drive, and dtype.
The report must also include `R`, `E_R`, `S`, `K_c`, `K_p`, `D_da`, eligibility
state, learning rate, `theta`, and active-edge fraction. Comparing a portable
PyTorch dopamine path with a native fused denominator is not a kernel-overhead
measurement. When `theta>0`, `R_da_event` must also report divergence from the
matrix trajectory or task metric.

Delayed communication is an alternative propagation operator in the current
code, not part of the integrated dopamine runtime. Its future ratio is:

```text
R_delay = time(delayed multi-channel communication tick) /
          time(raw SpMM on the same M edges)
```

It must be labeled as a semantic comparison because the delayed operator reads
multiple historical states while raw SpMM reads one current state.

## Recorded Measurements

The accepted core measurements in `results.md` were taken on an NVIDIA GeForce
RTX 5070 Laptop GPU with 8 GiB, CUDA toolkit 13.1, PyTorch 2.11 with CUDA 13.0,
and FP32 runtime. `benchmark.py` constructs a fresh runtime per sample, warms it,
synchronizes around the timed tick loop, and reports the median and best sample.
`training_parity.py` synchronizes around each full forward/backward sample and
reports the median, but currently has no explicit untimed rollout warmup.

The repository records the summary tables, code, and environment description,
but not the original command lines or raw per-sample logs. The tables are the
current source of record, while the clean protocol below defines the stronger
provenance required for the next publication.

### Forward

| N | M | B | Raw SpMM | Canonical SpMM | Complete tick | `R_forward` |
|---:|---:|---:|---:|---:|---:|---:|
| 166k | 10.00M | 1 | 0.725 ms | 0.558 ms | 0.595 ms | 0.821x |
| 166k | 24.99M | 1 | 1.820 ms | 1.313 ms | 1.350 ms | 0.742x |
| 166k | 49.95M | 1 | 3.305 ms | 2.686 ms | 2.466 ms | 0.746x |
| 50k | 1.00M | 8 | 0.447 ms | 0.460 ms | 0.513 ms | 1.148x |

At `B=1`, the complete tick was 18% to 26% faster than the raw cuSPARSE
denominator. At the directly comparable 10M and 24.99M custom-kernel points,
the fused neural operations added about 6.6% and 2.8%, respectively, over the
canonical custom SpMM. The 49.95M fused result is faster than its canonical
SpMM measurement; it should be treated as an empirical kernel result, not as a
negative cost for neural dynamics.

At the tested `B=8`, low-degree point, `R_forward=1.148x`. This meets the current
`1.15x` point target, but it is not a batch-wide scaling curve.

### Fixed event path

After 20 warmup ticks at `theta=0.05`, the recorded workload emitted about
2.64% of edges per tick.

| N | M | B | Complete matrix tick | Native event tick | `R_event` |
|---:|---:|---:|---:|---:|---:|
| 166k | 10.00M | 1 | 0.595 ms | 0.200 ms | 0.336x |
| 166k | 24.99M | 1 | 1.350 ms | 0.301 ms | 0.223x |
| 50k | 1.00M | 8 | 0.513 ms | 0.123 ms | 0.240x |

These are inference measurements without full eligibility scanning. They do not
show that an eligibility-enabled event runtime has the same ratio. The
large-scale timing run did not co-report state or task error, so this table is
throughput evidence only; the separate small-network accuracy sweep is not a
substitute for an error measurement on this workload.

### Forward plus backward

The recorded training workload is `N=16,384`, `M=998,187`, `B=4`, `T=32`.

| Training graph | Time | `R_train` | Peak graph memory |
|---|---:|---:|---:|
| Same-parameter sparse RNN | 185.6 ms | 1.000x | 1341.4 MiB |
| PyTorch matrix dynamics | 209.9 ms | 1.131x | 1341.4 MiB |
| Native pointwise dynamics | 199.5 ms | 1.075x | 1341.4 MiB |
| PyTorch persistent `theta=0` | 227.4 ms | 1.225x | 1341.4 MiB |

The native pointwise adjoint recovers about 93.0% same-parameter training
efficiency. It misses the `R_train <= 1.05x` target by 0.025 ratio points. Sparse
weight backward still uses PyTorch/cuSPARSE, and no checkpointed whole-rollout
adjoint is implemented.

## Executed Scale Versus Projection

| Item | Executed or measured | Projection or target |
|---|---|---|
| Neurons | `N=166k` executed | `N=166k` target reached |
| Connections | `M=49.95M` executed | `M=125M` target |
| Target fraction | 39.96% of 125M executed | 100% not yet executed |
| 125M latency | Not measured | 6.2 to 6.8 ms/tick |
| 125M fixed-topology bytes | Not allocated as a deployed graph | about 2.001 GB decimal |
| Integrated biology latency | Not published | pending clean isolated run |

The 125M latency is extrapolated from 18.5 to 20.3 billion fused edges/s. It is
not a benchmark result. The largest run stopped at 49.95M because random graph
construction retains large temporary and duplicate views. Loading a prebuilt
canonical graph is required to test 125M without that builder peak.

For `M < 2^31`, the intended fixed-weight native representation contains four
`int32` or FP32 edge arrays and two `int32` row-pointer arrays:

```text
bytes_native_topology = 16 M + 8 (N + 1)
```

For `M=125,000,000` and `N=166,000`, this is 2,001,328,008 bytes, or about
2.001 GB decimal and 1.864 GiB. This is a logical deployed representation, not
the peak memory of `random_topology` or the current Python object. The current
`SparseTopology` retains six `M`-length FP32/int32 arrays plus two row pointers:

```text
bytes_current_topology_tensors = 24 M + 8 (N + 1)
```

The forward results table's `Native topology` column uses
`native_storage_bytes()`, the conceptual slim `16M + 8(N+1)` layout, rather than
the full live Python allocation or full runtime memory. Online dopamine
plasticity currently consumes the duplicate source-major and destination-major
weight arrays, so it cannot use the single-weight figure unchanged.

Checkpoint cost is separate from live tick cost. A compact topology payload is
approximately `4*(5M + 2(N+1))` bytes for FP32/int32 fields; capture additionally
copies runtime state and optional communication history from device to CPU.
Restore validates the graph structure, performs linear permutation checks, and
copies the payload. It is correctness-oriented `torch.save`, not mmap/zero-copy
startup storage; current restore still performs multiple CPU/device comparisons
and should be treated as a boundary operation rather than a real-time tick.
The validated loader is not yet a zero-copy 125M-edge startup path; its safety
copies can create a large temporary peak that has not been measured on 8 GiB.

Structural and episode boundaries can create p99 spikes even when their
amortized cost is small: topology rebuild/resync is O(BM), checkpoint copies are
PCIe/storage-bound, and replay sampling copies transition payloads. Real-time
acceptance must report boundary ticks separately from steady-state p99.

## Fast-Tick Cost Model

### Base matrix runtime

One fixed-connectome matrix tick performs one sparse matrix/dense matrix
multiplication and pointwise neuron dynamics:

```text
time:       O(BM + BN)
persistent: O(M + BN + N)
```

The native fused path combines sparse reduction and pointwise dynamics in one
kernel. The `16M + 8(N+1)` formula covers only native graph storage. Neural
state, output, alpha, gain, and allocator workspace are additional.

### PELS event runtime

The native event path advances and thresholds all state values, compacts active
sources, and scatters their outgoing edges:

```text
time:       O(BN + M_A)
persistent: O(M + BN + N)
```

`M_A` counts expanded edges across active batch/source pairs, not unique graph
edges. The native path keeps the active count on device. Reading it for stats
causes synchronization. The portable event path dynamically sizes tensors and
uses `.item()`; it is a semantic reference, not the production event kernel.

### Regional dopamine without eligibility

Receptor modulation is cached between chemistry boundaries and consumed inside
the native fused neural kernel. RPE and optional release are accumulated every
neural tick. At each chemistry boundary, the regional field performs a dense
source-to-region projection, deterministic diffusion reduction, clearance, and
neuron receptor refresh.

```text
per neural tick:
  O(BM + BN + BS)

amortized chemistry addition:
  O((B(R + E_R + S R) + BN) / K_c)

persistent addition over the base runtime:
  O(BN + BR + S R + E_R + (D_da + 2) B S + N)

boundary working storage:
  O(B S R + B E_R + BN)
```

The `BN` term at a chemistry boundary is the region-to-neuron gather and D1/D2
occupancy/modulation refresh. Several cached `BN` signal tensors exist, so the
constant is material even though the asymptotic class is unchanged. The delay
ring stores exactly `D_da * B * S` values when enabled, and the two source
accumulators each store `B*S` values. The `[B,S,R]` dip tensor is a chemistry
boundary temporary, not persistent state.

### Eligibility

The integrated rule updates pre/post neuron traces and one eligibility value per
edge on every tick:

```text
time:       O(BN + BM)
persistent: +2BN + M floating values
```

The native implementation launches one neuron-trace kernel and one edge kernel.
Each edge thread loops over `B`. It avoids a persistent `[B,M]` tensor, but it
still reads `O(BM)` state values. The portable implementation materializes
`[B,M]` intermediates. The separate exact batch plasticity API intentionally
stores `[B,M]` eligibility, adding `O(BM)` persistent memory.

This is the largest current threat to event-mode scaling: even if propagation
falls to `O(BN + M_A)`, full eligibility keeps total work at `O(BM)`.

### Dopamine-gated weight plasticity

For exact shared online learning, the runtime currently requires `B=1`. At a
plasticity boundary it scans all edges, applies the bounded sign-preserving
update, and writes both source-major and destination-major weight views:

```text
weight update:                   O(M)
amortized over neural ticks:     O(M / (K_c K_p))
```

The current native online update needs a second FP32 source-major weight view,
adding `4M` bytes relative to the intended single-weight fixed-topology layout.
Eligibility adds another `4M` bytes.

A persistent-current runtime must also rebuild
`current = W @ emitted_state` after every scheduled plasticity boundary when
learning rate or weight decay is nonzero, even if projection makes the observed
maximum delta zero:

```text
resynchronization boundary:      O(BM)
amortized resynchronization:     O(BM / (K_c K_p))
```

The matrix runtime already recomputes current on its next normal tick and does
not maintain this cache. Weight-version checks also rebuild a persistent cache
after an external mutation.

### Delayed multi-channel communication

The current reference module sorts edges once, gathers one historical source
value per batch/edge, applies a cached receptor gain, and scatter-adds by
destination:

```text
construction: O(M log M)
tick:         O(BM + BN)
persistent:   O(M + (D + 1)BN + C)       global receptor gains
              O(M + (D + 1)BN + NC)      destination-local gains
transient:    O(BM + BN)
```

For FP32 weights with the code's four persistent `int64` edge fields (`src`,
`dst`, `delay`, `transmitter`), FP32 weight, and FP32 cached edge gain, edge
storage alone is approximately `40M` bytes, before receptor gains and history.
When all delays are zero, the history tensor is omitted. Otherwise it stores
`(D+1)BN` FP32 values, which can dominate memory.

This module is correct but is not yet an efficiency-parity implementation.
Delay/channel bucketing into a small set of CSR/SpMM operations is needed before
claiming matrix-like constants.

When the scheduler attaches this communication graph to a neural event runtime,
its full `O(BM_comm)` gather/scatter is additive; it does not automatically
inherit the neural event path's `M_A` sparsity. Shared delay buckets or a fused
communication/event kernel are required for end-to-end event parity.

### Migration

The migration primitive updates 3-D positions from target attraction,
chemotactic gradient, optional noise, motility, speed limits, and domain bounds:

```text
one migration epoch:         O(N)
persistent/working storage:  O(N)
amortized per neural tick:   O(N / L_m)
```

It never constructs neuron pairs. It is a tested standalone slow process and is
not yet scheduled inside the integrated neural/chemical runtime.

### Structural-connectome update

The implementation scores exactly `K` caller-supplied candidate edges, matches
them against `M` existing edges, applies hysteretic pruning/formation, performs
deterministic per-source top-k selection, and emits canonical edges:

```text
candidate scoring:              O(N + K)
full structural epoch:          O(N + M log M + K log K)
persistent/working storage:     O(N + M + K)
amortized per neural tick:      O((N + M log M + K log K) / L_s)
```

Candidate generation is deliberately outside this function and outside the
cost above. It must use a bounded spatial index; an all-pairs `O(N^2)` candidate
search would violate the project contract. Rebuilding `SparseTopology` adds
source-major and destination-major sorting and invalidates persistent currents.

## Future Combined Amortized Model

For a future unified multi-rate runtime, a useful accounting equation is:

```text
T_tick ~= T_sparse_neural
        + I_eligibility * T_full_edge_eligibility
        + T_rpe_accumulation
        + (T_chemistry + T_receptor_refresh) / K_c
        + (T_weight_update + I_persistent * T_resync) / (K_c K_p)
        + T_migration / L_m
        + (T_structure + I_persistent * T_resync) / L_s
```

Delayed communication should replace `T_sparse_neural` when integrated. Adding
it on top would count synaptic propagation twice.

Today, migration, structural development, and delayed communication are
standalone modules, so their terms are zero in the integrated dopamine runtime.
The equation is an implementation budget, not a description of one currently
executing scheduler.

With small `R` and `S`, bounded `E_R`, and large update intervals, chemistry,
migration, and development can remain lower-order amortized work. Full
eligibility is different: it is another `BM` edge pass every tick and therefore
changes the constant factor even though the asymptotic class remains `O(BM)`.

For the event runtime:

```text
without eligibility: O(BN + M_A) plus amortized slow processes
with full eligibility: O(BN + M_A + BM)
```

Therefore low emission activity alone is insufficient for end-to-end event
speedup when full per-edge learning is enabled.

## Validation And Synchronization

### Safe research mode

`DopaminePlasticityConfig(validate_inputs=True)` checks every per-tick RPE and
external release for finite values and rejects negative release before mutating
state. On CUDA, conversions such as `bool(torch.all(...))` require a
device-to-host synchronization. This mode is appropriate for model development,
tests, and short correctness runs, not latency measurement.

Initialization also performs value, range, stability, and topology checks that
may synchronize. They are outside the normal tick and should stay enabled.

### Trusted fast mode

`validate_inputs=False` preserves shape, dtype, and device checks based on tensor
metadata but skips per-tick value reductions. The owning runtime then calls the
unchecked chemistry and receptor paths. This keeps the normal CUDA tick
asynchronous. Inputs containing NaN, infinity, or negative external release are
outside the contract in this mode.

Fast measurement also requires `collect_stats=False`. Statistics convert GPU
minima, maxima, active counts, and weight deltas to Python values and therefore
synchronize. `NativeDopamineEventRuntime` keeps its active count on device;
`.item()` is used only when stats are requested.

The portable `HybridPELSRuntime(propagation="auto")` synchronizes at policy
sampling ticks to count active edges. Explicit matrix mode avoids that count,
and the native event kernel itself has no per-tick host count read. A fully
device-side production matrix/event selector remains open work.

## Clean Benchmark Protocol

No new biological latency should be published until all of the following are
true:

1. The GPU is isolated from unrelated compute. Record `nvidia-smi` before and
   after every suite, including utilization, memory, temperature, power, clocks,
   and compute processes.
2. Run on AC power with a stable performance profile. Allow temperature and
   clocks to settle, then reject runs with unrelated utilization or throttling.
3. Compile/load the CUDA extension and load or construct topology before timing.
   Use a prebuilt canonical graph for the 125M test.
4. Use one topology, initial state, drive sequence, dtype, batch, and stream for
   every numerator/denominator pair. Reset state identically for every sample.
5. For state-dependent event tests, repeat the same warmup trajectory and report
   `theta`, active-neuron fraction, and active-edge fraction.
6. Use `validate_inputs=False` and `collect_stats=False` only inside trusted
   timed loops. Run safe validation and numerical comparisons immediately before
   the timed suite, outside the timing window.
7. Warm up at least the existing 20 ticks. Time at least the existing 100 ticks
   and 5 independent samples. Synchronize only immediately before and after the
   timed loop, or use CUDA events on the same stream.
8. Report median latency and dispersion, not only the best sample. Store the
   command, complete raw output, seed, software versions, and exact edge count.
9. Measure memory separately after initialization and peak memory during the
   timed phase. Distinguish logical deployed topology, live Python allocation,
   allocator reservation, and graph-construction peak.
10. Run ablations in this order: raw SpMM, canonical SpMM, fused neural tick,
    dopamine without eligibility, dopamine with eligibility, nonzero learning,
    persistent resynchronization, fixed event path, and delayed communication.
11. For training, time the complete forward, scalar loss, and backward exactly
    as `training_parity.py` does. Report `N`, `M`, `B`, `T`, graph memory, and the
    sparse-weight gradient path.
12. Re-run correctness after timing to detect state, current, weight-view, or
    delay-ring divergence.

`dopamine_benchmark.py` currently creates each runtime once and continues its
state through the warmup and timed phases of later samples. It needs a runtime
factory or explicit complete reset before it satisfies item 4. This does not
invalidate it as a development tool, but its output should not be promoted as
the clean integrated result.

On 2026-09-27, the measurement workstation was not clean: a read-only snapshot
showed 99% GPU utilization, 6635 MiB of 8151 MiB in use, 87 C and about 90.76 W,
with unrelated GPU workloads present. No absolute neurochemical timing from that
state should be treated as publishable. Existing correctness validation remains
useful under contention.

## Current Bottlenecks

1. There is no clean integrated dopamine performance baseline, so the constant
   cost of receptor refresh, chemistry, eligibility, and plasticity is unknown.
2. Full eligibility scans all `M` edges and loops over `B` every neural tick. It
   can erase the benefit of sparse event propagation.
3. Training still delegates sparse-weight backward to PyTorch/cuSPARSE. The
   measured `R_train=1.075x` remains above the `1.05x` target.
4. The 125M point is projected, not executed. Random topology construction, not
   the intended deployed layout, currently blocks the scale test.
5. Automatic matrix/event selection in the portable runtime synchronizes with
   the host; there is no device-only production policy.
6. Online plasticity updates two weight views. This costs bandwidth and retains
   an extra FP32 edge array relative to the fixed-weight single-view target.
7. Persistent current requires a full SpMM after weight mutation, making a high
   plasticity frequency expensive.
8. Delayed communication uses per-edge gathers, scatter-add, `int64` metadata,
   and a dense maximum-delay history ring. It has not reached CSR-like constants.
9. Migration and connectome development are tested standalone modules, not one
   causally scheduled multi-rate runtime. Structural candidate generation is
   still external.
10. CUDA graph capture and high-batch/low-degree tuning are not implemented.

## Optimization Priorities

1. Run the clean ablation suite and publish component latency, bandwidth, active
   fractions, and memory. Optimization without this baseline risks targeting a
   lower-order chemistry cost while eligibility dominates.
2. Add a prebuilt topology format and execute `N=166k, M=125M` with no random
   builder in the measured process.
3. Fuse eligibility work with propagation where practical, or introduce a
   rigorously defined sampled/event-local eligibility mode. Preserve the full
   rule as the correctness reference.
4. Implement the checkpointed whole-rollout adjoint and fused sparse backward,
   then remeasure the `R_train <= 1.05x` gate.
5. Implement a device-only matrix/event selector and CUDA graph capture without
   active-count host reads or dynamic allocation.
6. Choose one canonical online weight layout and align eligibility to it so a
   plasticity kernel updates one weight array. Retain permutation metadata, not
   a second mutable weight copy.
7. Bucket delayed edges by a small number of `(delay, channel)` groups and use
   CSR/SpMM or fused segmented kernels. Replace the dense `D+1` history when a
   sparse or quantized history contract is valid.
8. Integrate neural, chemistry, plasticity, migration, and structural epochs in
   one explicit multi-rate scheduler, with every slow boundary measured.
9. Add bounded spatial candidate generation and rebuild only affected topology
   blocks or regions instead of globally sorting every structural epoch.
10. Revisit chemistry kernel fusion only after the ablation shows it matters;
    for small `R` and `S`, it should be amortized below edge-dominant work.

## Status Against The Main Goal

The fixed-connectome forward engine has reached the requested same-parameter
class and has achieved latency parity at the measured `B=1` and tested `B=8`
points. Training is close at `1.075x`. The 125M scale point and the fully
integrated biological runtime are not yet demonstrated.

The key engineering constraint for the next phase is clear: regional chemistry,
migration, and structural change can be amortized on slow clocks, but full
eligibility and delayed per-edge communication sit on the fast edge path. Those
two mechanisms must be reorganized around fused, bucketed, or selectively
active matrix operations for the complete functional system to retain
same-parameter efficiency.

## Real-Time Control Metric

For embodied use, report the per-tick distribution rather than only average
throughput:

```text
deadline_miss_rate = count(tick_latency > deadline) / tick_count
```

`benchmarks/realtime_latency.py` synchronizes around every native tick and
reports p50, p95, p99, max, mean, and deadline miss rate. It supports a saved
prebuilt topology so graph construction and JIT compilation do not contaminate
control-loop measurements. A throughput ratio below 1.0x does not imply a
meeting of a p99 control deadline.

---

[← Documentation index](index.md)
