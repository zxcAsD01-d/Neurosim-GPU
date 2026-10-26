# Runtime Architecture

## Objective

For `N` neurons and `M` fixed synapses, the fast neural tick must remain a
sparse linear algebra workload:

```text
time:   O(BM + BN)
memory: O(M + BN)
```

Regional chemistry, delayed communication, online plasticity, migration, and
structural development are added only when their state, cadence, and ownership
are explicit. No feature may introduce an implicit all-pairs neuron operation.

## State Ownership

| State | Shape | Owner | Lifetime |
|---|---:|---|---|
| Canonical topology/weights | O(M) | `SparseTopology` | fixed between structural boundaries |
| Neural activity | `[B,N]` | runtime | every neural tick |
| Emitted state/current | `[B,N]` | persistent runtime | every neural tick |
| Dopamine concentration | `[B,R]` | dopamine runtime | every chemistry boundary |
| Cached receptor modulation | `[B,N]` | dopamine runtime | held between chemistry boundaries |
| Pre/post traces | `2[B,N]` | plasticity runtime | every neural tick |
| Eligibility | `[M]` for exact B=1 online mode | plasticity runtime | every neural tick |
| Delay history | `[D+1,B,N]` | communication module | every communication tick |
| Positions | `[N,3]` | development scheduler | slow migration boundary |
| Candidate edges | `[K]` | spatial index/caller | slow structural boundary |

The topology carries a monotonic `weight_version`. Any persistent-current
runtime compares this version at tick entry and rebuilds `W @ emitted_state`
when another owner changed weights.

## Fast Neural Tick

For the integrated dopamine runtime, entry state is
`(s_t, W_t, C_t, eligibility_t)`:

```text
cached C_t receptor modulation
          |
          v
W_t @ s_t + drive -> s_(t+1)
          |
          +-> update pre/post traces and eligibility
          |
          +-> accumulate outcome RPE/release
```

At a chemistry boundary:

```text
delayed/averaged release -> C_(t+1)
C_(t+1) + eligibility -> W_(t+1)
weight version increments
persistent current resynchronizes if present
```

The outcome cannot alter `s_(t+1)`, which has already been computed. New
chemistry and weights affect only the following transition.

## Runtime Modes

### Reference matrix

`MatrixRuntime` and `DopamineMatrixRuntime` use PyTorch CSR. They are portable,
readable correctness references. The dopamine reference path uses Python tensor
operators and is not the production latency path.

### Reference persistent

`HybridPELSRuntime` and `DopaminePersistentRuntime` implement emitted-state and
persistent-current semantics. Their delta propagation is a full SpMM, so they
validate bounded error but do not provide event-sparse edge cost.

### Native matrix

`NativeCanonicalMatrixRuntime` fuses destination-major sparse reduction and
neural dynamics. `NativeDopamineMatrixRuntime` additionally reads cached
receptor drive/gain. Both use one canonical destination-major weight array.

### Native event

`NativeEventPELSRuntime` and `NativeDopamineEventRuntime` fuse neuron advance,
thresholding, event compaction, and source-major edge scatter. They are the
actual `O(BN + M_A)` low-activity propagation paths. With
`track_eligibility=True`, dopamine runtime still scans all M edges for
eligibility each tick; a weight-update boundary also performs O(M) projection
and a full current rebuild. Whole-tick event sparsity therefore currently
requires eligibility disabled and no weight boundary on that tick. A production
device-side policy and sparse eligibility design remain open.

## Topology Layout

The logical read-only native inference layout uses:

```text
destination CSR: in_rowptr, in_src, weight
source event view: out_rowptr, out_dst, out_weight_index
```

For read-only inference, `out_weight_index` can address one destination-major
weight array, giving approximately `16M + 8N` bytes. The current mutable
`SparseTopology` keeps source and destination weight views so an online update
can write both deterministically; its practical graph allocation is about
`24M + 8N` bytes. Single-weight online learning remains an optimization target.

Duplicate input `(source,destination)` edges are coalesced. Online updates write
both source and destination views in one native kernel and then increment the
topology version.

## Multi-Rate Schedule

Recommended cadence is configuration-dependent, but ownership is fixed:

```text
every 1 neural tick:
    neural dynamics
    communication/history
    local traces and eligibility

every K_c neural ticks:
    regional chemistry
    receptor-cache refresh
    optional dopamine-gated weight update

every K_m chemistry ticks:
    migration
    region remapping

every K_s migration ticks:
    candidate scoring
    formation/pruning
    canonical topology rebuild
    optimizer/eligibility/history remap
```

Chemistry `dt` must represent its chemistry boundary duration. RPE samples are
averaged into a release rate over `K_c`; transport delay is measured in neural
ticks.

## Integrated Versus Standalone

Integrated today:

- sparse neural dynamics;
- regional dopamine, D1/D2 modulation, delay and multirate chemistry;
- exact B=1 online eligibility/weight plasticity;
- matrix, persistent, native matrix, and native event dopamine runtimes;
- topology versioning and persistent-current invalidation.
- checkpointable scheduler/replay state for episode-boundary continuation.

Standalone tested primitives, not yet one combined scheduler:

- `DelayedMultiChannelCommunication`;
- `migrate_neurons`;
- `update_structural_connectome`.

A structural update returns canonical edge tensors suitable for rebuilding
`SparseTopology`; it does not currently hot-remap optimizer state, delay rings,
or eligibility. Calling it a live developmental whole-brain runtime would be
premature.

The scheduler and replay controller now provide the minimal embodied continual
learning loop: environment step, transition capture, replay sampling, episode
counter, and deterministic resume. They deliberately do not prescribe an
optimizer, policy loss, or environment API.

## Safety And Performance Modes

`validate_inputs=True` rejects non-finite RPE and invalid release before any
state mutation, but CUDA scalar checks synchronize the host. Trusted native
benchmarks may set it to `False`; shape/device/dtype still remain checked, while
non-finite values are outside the contract.

`collect_stats=True` also reads scalar values back to the host and is not part
of the asynchronous latency path. Fast timing requires both
`validate_inputs=False` and `collect_stats=False` after external validation.

All integrated dopamine `step` methods run under `torch.no_grad`. Their weight
changes are local online three-factor plasticity, not BPTT through chemistry.

Exact shared online dopamine plasticity requires `B=1`. Offline exact batch
plasticity uses `[B,M]` eligibility and reduces the within-sample product. The
one-trace shared approximation must not be described as exact batch learning.

## Structural Boundary Contract

Before replacing a topology, the scheduler must:

1. finish all kernels using the old topology;
2. produce unique canonical edges from a bounded candidate list;
3. transfer retained weights and initialize formed weights;
4. remap or reset eligibility and optimizer state;
5. rebuild delayed communication views/history as required;
6. rebuild persistent current from the new weights and emitted state;
7. resume only after every runtime observes the new topology version.

This orchestration is a remaining integration milestone.

---

[← Documentation index](index.md)
