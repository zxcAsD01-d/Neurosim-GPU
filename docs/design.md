# Design: numerical and runtime contract

This file records the fixed-topology numerical/runtime contract. See
`architecture.md` for state ownership and multi-rate integration, and
`biology.md` for the functional neurochemical model.

## Target

For `N` neurons, `M` synapses and batch `B`, a normal simulation tick must cost
`O(BN + BM)` work and use `O(BN + M)` persistent memory. Runtime bookkeeping
must not introduce an `N log N`, `M log M` or per-neuron host loop.

For `M < 2^31`, graph indices are stored as `int32`. A read-only native
inference layout can use one destination-major FP32 weight array, destination
CSR, and a source-major event permutation: approximately `16M + 8N` bytes.
The current mutable `SparseTopology` and native online-learning path retain both
source- and destination-major weight views plus their indices, approximately
`24M + 8N` bytes. The smaller layout is a deployment target, not the current
online-learning allocation.

At low emission activity, the optional event path targets `O(BN + M_A)`, where
`M_A` is the number of outgoing edges of active source neurons. The event path
is an optimization of the same tensor state, not a separate object model.

## Tick contract

At entry to tick `t`, `state` is `s_t` and the synaptic current is based on the
state visible to the selected runtime:

1. Compute `s_(t+1)` from `s_t`, current `C_t`, and external drive at `t`.
2. Compare `s_(t+1)` with `s_emit`.
3. Emit selected deltas and update persistent current for tick `t+1`.
4. Commit `state = s_(t+1)`.

With `theta=0`, every delta is emitted and `current_(t+1) = W @ s_(t+1)`.
This removes the index ambiguity in the source v3.0 specification.

## Execution paths

### Matrix runtime

Destination-major CSR is passed to cuSPARSE. Each tick performs one sparse
matrix/dense matrix multiplication and vector point operations. This is the
performance and correctness baseline, not merely a fallback.

### Persistent matrix runtime

`current = W @ s_emit` is maintained using `W @ emitted_delta`. It has the same
worst-case complexity as the matrix runtime and provides bounded-error PELS
semantics without irregular GPU execution.

### Event runtime

Duplicate `(source, destination)` inputs are coalesced by summing weights. The
single canonical weight array is destination-major; a source-major edge view
uses `out_weight_index` to address it. Active `(batch, source)` pairs expand
contiguous outgoing intervals and accumulate by destination. This avoids
scanning inactive edges, but introduces compaction, permutation reads and
atomics. A future device-side policy will select it only below a calibrated
active-edge fraction.

The reference PyTorch event path contains host synchronizations in event-path
sizing. The native inference path instead uses two kernels: fused neuron
advance/threshold/compaction, followed by source-major edge scatter. The active
count remains on device and the scatter kernel uses a fixed persistent grid.
This removes all per-tick host synchronization and dynamic tensor allocation.
The scatter kernel assigns one warp to each active source and uses a fixed grid;
the default 128 blocks provide 1,024 concurrent source workers.

## Numerical contract

Signed synapses are supported. If `r = state - s_emit` and `|r_j| < theta_j`,
the omitted current satisfies:

```text
|delta_u_i| <= sum_j |W_ij| theta_j
```

Let `A = diag(alpha) + diag((1-alpha)*gain) |W|`. A global infinite-horizon
bound requires `rho(A) < 1` (or a valid weighted induced-norm bound). A simple
sufficient uniform bound is based on `||A||_inf < 1`; independent scalar bounds
per neuron are not generally valid for a coupled recurrent network.

`uniform_error_budget` implements the conservative sufficient condition
`beta_i gain_i sum_j |W_ij| theta_j <= (1-kappa_i) epsilon`. It returns an
independent threshold for every source neuron and rejects networks whose row
infinity norm cannot certify contraction.

The certificate belongs to a specific weight/gain envelope. Online weight
updates or positive dopamine gain modulation invalidate it unless thresholds
are recomputed from worst-case/current values.

## Training contract

Scheduler predicates use detached states. Conditional on the selected event
schedule, emitted values remain differentiable. `theta=0` must match dense BPTT
to numerical precision. `theta>0` is a piecewise differentiable surrogate and
must be evaluated against the dense gradient, not labeled globally exact.
The native fused pointwise training operator provides an analytic backward for
state and synaptic current while treating alpha/gain as fixed. Sparse weight
backward still uses PyTorch CSR until a checkpointed whole-rollout adjoint is
implemented.

## Next Optimization Sequence

Completed gates include dense/CSR/theta-zero equivalence, threshold/error
sweeps, canonical fused matrix/event kernels, analytic pointwise backward, and
native dopamine modulation/eligibility/weight kernels. Remaining order is:

1. Load and execute a prebuilt 125M-edge native topology.
2. Add a device-side matrix/event policy and CUDA graph capture.
3. Integrate delayed communication into the dopamine runtime.
4. Add the structural-boundary remap transaction.
5. Add macro advancement only for inputs proven constant over the segment.
6. Add checkpointed whole-rollout BPTT and fused sparse adjoints.

Regional dopamine `D` is now explicit and multirate. Generic global chemical
channels `C` and environment state `E` remain external inputs. Any future
eventization or macro-holding of them requires a separate error budget.

---

[← Documentation index](index.md)
