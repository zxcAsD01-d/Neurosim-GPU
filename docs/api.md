# NeuroSim GPU API Guide

This guide covers the primary implemented workflows and runtime APIs. The model is a functional
neurochemical network simulator, not a molecular or clinically validated brain
model. Unless a section says otherwise, examples run on CPU or CUDA and use
only the installed `neurosim_gpu` package.

## Install and conventions

From the project directory:

```powershell
python -m pip install -e .
python -m unittest discover -s tests -v
```

The examples use these dimensions:

| Symbol | Meaning | Tensor shape |
|---|---|---|
| `B` | independent simulation batches | leading dimension |
| `N` | neurons | state `[B, N]` |
| `M` | canonical synapses | weight, eligibility `[M]` |
| `R` | chemical regions | concentration `[B, R]` |
| `S` | dopamine source populations | RPE/release `[B, S]` |
| `C` | transmitter channels | receptor gains `[C]` or `[N, C]` |
| `K` | structural candidate edges | candidate arrays `[K]` |

Floating tensors participating in one operation must have the same dtype and
device. Edge and region indices must be `torch.int32` or `torch.int64` on that
device. Native kernels require contiguous CUDA `torch.float32` state.

The following helper is reused below. Call it again for each runtime that may
change weights; `SparseTopology` contains mutable weight storage.

```python
import torch

from neurosim_gpu import (
    DopamineReceptors,
    Dynamics,
    RegionalDopamineField,
    SparseTopology,
)


def make_system(device="cpu", dtype=torch.float32, batch=1):
    device = torch.device(device)
    n = 4
    src = torch.tensor([0, 0, 1, 2], dtype=torch.int64, device=device)
    dst = torch.tensor([1, 2, 3, 3], dtype=torch.int64, device=device)
    weight = torch.tensor(
        [0.12, 0.10, -0.08, 0.09], dtype=dtype, device=device
    )
    topology = SparseTopology.from_edges(n, src, dst, weight)
    dynamics = Dynamics(
        alpha=torch.full((n,), 0.80, dtype=dtype, device=device),
        gain=torch.full((n,), 0.70, dtype=dtype, device=device),
    )
    state = torch.full((batch, n), 0.40, dtype=dtype, device=device)

    field = RegionalDopamineField(
        tonic_baseline=torch.full((2,), 0.50, dtype=dtype, device=device),
        uptake_rate=torch.full((2,), 0.20, dtype=dtype, device=device),
        decay_rate=torch.zeros(2, dtype=dtype, device=device),
        release_projection=torch.tensor(
            [[1.0, 1.0]], dtype=dtype, device=device
        ),  # [S=1, R=2]
        edge_src=torch.tensor([0], dtype=torch.int64, device=device),
        edge_dst=torch.tensor([1], dtype=torch.int64, device=device),
        diffusion_rate=torch.tensor([0.10], dtype=dtype, device=device),
        dt=0.10,
        burst_gain=0.80,
        dip_gain=1.0,
        autoreceptor_gain=0.25,
        autoreceptor_half_saturation=0.25,
    )
    neuron_region = torch.tensor(
        [0, 0, 1, 1], dtype=torch.int64, device=device
    )
    receptors = DopamineReceptors(
        d1_density=torch.tensor([1, 0, 1, 0], dtype=dtype, device=device),
        d2_density=torch.tensor([0, 1, 0, 1], dtype=dtype, device=device),
        tonic_concentration=0.50,
        d1_half_saturation=0.50,
        d2_half_saturation=0.25,
    )
    return topology, dynamics, state, field, receptors, neuron_region
```

`SparseTopology.from_edges` sorts edges and coalesces duplicate `(src, dst)`
pairs by summing their weights. Consequently `topology.n_edges` can be smaller
than the input edge count. Its canonical `topology.weight` is source-major;
`topology.in_weight` is the destination-major CSR weight view.

## Gomoku task probe

The package also exposes a small CPU-first `GomokuEnv` and player-relative
`encode_board` helper for task-level experiments. A position is encoded as
`[own, opponent, empty]` planes; legal actions remain an explicit mask and
must be applied to policy logits before selecting a move.

```python
from neurosim_gpu import GomokuEnv, encode_board

env = GomokuEnv(board_size=9, win_length=5)
observation = encode_board(env.board, env.current_player)
legal = env.legal_move_mask().reshape(-1)
```

`canonical_move` and `canonical_symmetry_index` provide action-aligned
dihedral transforms for replay preprocessing. The reproducible CNN versus
sparse-rate comparison is in `benchmarks/gomoku_learning.py`; its scope and
claim boundaries are documented in `gomoku-protocol.md`.

## Base matrix runtime

`MatrixRuntime` is the exact sparse baseline. One tick computes `W @ s_t` and
then synchronously advances all neurons to `s_(t+1)`. Its work is
`O(B*M + B*N)`.

```python
from neurosim_gpu import MatrixRuntime

topology, dynamics, initial, _, _, _ = make_system(batch=2)
runtime = MatrixRuntime(topology, dynamics, initial)

for _ in range(8):
    state = runtime.step(0.15)  # scalar or broadcastable tensor drive

assert state.shape == (2, 4)
assert state.dtype == torch.float32
```

`MatrixRuntime` uses normal PyTorch operations and supports autograd. Do not
wrap a training rollout in `torch.no_grad()`. A runtime owns its evolving
`state`, so create a fresh runtime or explicitly detach state between training
rollouts as appropriate.

For bounded persistent propagation, `HybridPELSRuntime(..., theta=0,
propagation="matrix")` has exact matrix semantics to numerical precision.
Positive `theta` omits sub-threshold source deltas and is an approximation.
`propagation="event"` is useful for correctness experiments but performs host
synchronization; use the native event runtime for an asynchronous GPU path.

### Error-budgeted persistent reference

`uniform_error_budget` constructs conservative per-source thresholds only when
the current signed graph and gain envelope satisfy its contraction test.

```python
from neurosim_gpu import HybridPELSRuntime, uniform_error_budget

topology, dynamics, initial, _, _, _ = make_system(batch=1)
budget = uniform_error_budget(topology, dynamics, target_state_error=1e-3)
runtime = HybridPELSRuntime(
    topology,
    dynamics,
    initial,
    theta=budget.theta,
    propagation="matrix",  # semantic reference, still scans all M edges
)
state = runtime.step(0.15)
```

The certificate is tied to the weight, `alpha`, and gain envelope used to build
it, and construction raises when any row cannot be certified below one. Online
plasticity, structural change, or positive gain modulation requires recomputing
it or using a separately proven worst-case envelope.
`budget.theta` can contain `+inf` for a source with no outgoing edges; the
hybrid and dopamine persistent runtimes accept infinity only for such inert
sources and reject it elsewhere.

## Native CUDA setup

Native execution needs a CUDA-enabled PyTorch build, a compatible CUDA toolkit,
and MSVC. The current loader passes the MSVC `/O2` flag and is Windows-oriented;
another host platform requires adapting the build flags in `native.py`. The
extension is JIT compiled and cached on first use. On the provided workstation,
this command initializes MSVC and CUDA:

```powershell
scripts\native_env.bat python benchmarks\validate_native.py
```

`scripts/native_env.bat` contains workstation-specific compiler, toolkit, and
GPU architecture paths. Edit those values for another machine. A minimal
native matrix run is:

```python
import torch

from neurosim_gpu.native import (
    NativeCanonicalMatrixRuntime,
    load_native_extension,
)

if not torch.cuda.is_available():
    raise SystemExit("CUDA is required")

extension = load_native_extension(verbose=False)
topology, dynamics, initial, _, _, _ = make_system(
    device="cuda", dtype=torch.float32, batch=1
)
runtime = NativeCanonicalMatrixRuntime(
    topology, dynamics, initial, blocks=4096, extension=extension
)
with torch.inference_mode():
    for _ in range(8):
        state = runtime.step(0.15)
```

This runtime fuses canonical sparse propagation and neuron dynamics. Keep all
native inputs contiguous, including state, `alpha`, `gain`, topology arrays,
and modulation tensors. It is an inference runtime and rejects an
autograd-enabled state. The separate
`native_advance_training` API has an analytic backward only for the local
state/current update and treats `alpha` and `gain` as fixed; it is not a full
native recurrent backward.

## Regional dopamine field and receptors

`RegionalDopamineField` stores a non-negative concentration. A positive RPE
adds release; a negative RPE suppresses source-specific tonic release, making a
dip without creating negative dopamine.

```python
_, _, _, field, receptors, neuron_region = make_system(batch=2)

concentration = field.tonic_baseline.expand(2, -1).clone()  # [B=2, R=2]
burst_rpe = torch.tensor([[0.5], [0.0]], dtype=concentration.dtype)  # [B, S]
dip_rpe = torch.tensor([[-0.5], [0.0]], dtype=concentration.dtype)

after_burst = field.advance(concentration, burst_rpe)
after_dip = field.advance(concentration, dip_rpe)
assert torch.all(after_burst >= 0) and torch.all(after_dip >= 0)

# Project regional concentration to neurons before evaluating receptors.
neuron_concentration = after_burst.index_select(1, neuron_region)  # [B, N]
signals = receptors.response(neuron_concentration)
print(signals.d1_occupancy.shape, signals.net_phasic.shape)  # [B, N], [B, N]
```

The field graph is undirected: supply each region pair once, not both
directions. `release_projection` is non-negative `[S, R]`. Optional
`source_release` is also non-negative `[B, S]` and uses the same projection.
The explicit Euler step must satisfy `dt <= field.maximum_stable_timestep`;
construction rejects an unstable value. Region volume can be supplied as a
positive `[R]` tensor.

`source_release` is a direct external release rate. It bypasses both the RPE
delay ring and the RPE burst autoreceptor attenuation; the owning model must
apply any desired latency or feedback before passing it.

`DopamineReceptors` accepts scalar or `[N]` tonic, half-saturation, and Hill
parameters. Densities are non-negative `[N]`. Both occupancies rise with
concentration, but the implemented functional pathways have opposite signs:

```text
D1 pathway = +D1 density * (occupancy - tonic occupancy)
D2 pathway = -D2 density * (occupancy - tonic occupancy)
```

The runtime requires each receptor's tonic concentration to match the tonic
baseline of its neuron's assigned region.

An optional clipped temporal-difference helper can produce a one-source RPE:

```python
from neurosim_gpu import TemporalDifferenceRPE

critic = TemporalDifferenceRPE(discount=0.99, maximum_absolute=5.0)
reward = torch.tensor([1.0, -0.2])
value = torch.tensor([0.4, 0.3])
next_value = torch.tensor([0.6, 0.0])
terminal = torch.tensor([0.0, 1.0])
rpe = critic.compute(reward, value, next_value, terminal).unsqueeze(1)
assert rpe.shape == (2, 1)  # [B, S=1]
```

### Optional temporal-difference RPE

```python
from neurosim_gpu import TemporalDifferenceRPE

critic = TemporalDifferenceRPE(discount=0.99, maximum_absolute=5.0)
rpe = critic.compute(
    reward=torch.tensor([1.0]),
    value=torch.tensor([0.2]),
    next_value=torch.tensor([0.4]),
    terminal=torch.tensor([0.0]),
)
```

This helper supplies one possible source signal. Dopamine is not assumed to
encode only reward prediction error; external release can represent other
experimentally defined signals.

## DopamineMatrixRuntime: validated reference mode

This is the portable matrix runtime and the recommended research/debug mode.
`validate_inputs=True` rejects non-finite RPE and negative/non-finite external
release before mutating runtime state.

```python
from neurosim_gpu import DopamineMatrixRuntime, DopaminePlasticityConfig

topology, dynamics, initial, field, receptors, neuron_region = make_system(
    batch=2
)
runtime = DopamineMatrixRuntime(
    topology,
    dynamics,
    initial,
    chemistry=field,
    receptors=receptors,
    neuron_region=neuron_region,
    config=DopaminePlasticityConfig(
        drive_scale=1.5,
        log_gain_scale=0.0,
        learning_rate=0.0,
        track_eligibility=True,
        chemistry_interval=2,
        dopamine_delay_steps=1,
        validate_inputs=True,
    ),
)

rpe = torch.tensor([[0.5], [-0.25]], dtype=initial.dtype)  # [B=2, S=1]
release = torch.tensor([[0.0], [0.1]], dtype=initial.dtype)  # non-negative
state, stats = runtime.step(
    0.15, rpe=rpe, source_release=release, collect_stats=True
)
assert state.shape == (2, 4)
print(stats.chemistry_applied, stats.concentration_maximum)
```

On CUDA, create `rpe` and `release` on `initial.device`. Input validation and
`collect_stats=True` read device values on the host and therefore synchronize
CUDA. Disable statistics in timed loops.

`chemistry_interval=K` averages accumulated RPE/release over `K` neural ticks
and holds concentration and receptor modulation between boundaries.
`dopamine_delay_steps=D` delays the RPE by `D` neural ticks; a chemistry
boundary can add further waiting. External `source_release` is accumulated but
is not passed through the RPE delay ring.

## NativeDopamineMatrixRuntime: trusted fast mode

The native matrix path fuses sparse propagation with cached receptor-modulated
neuron dynamics, and uses native eligibility and weight-update kernels. The
regional chemistry step remains tensor code.

```python
from neurosim_gpu import NativeDopamineMatrixRuntime
from neurosim_gpu.native import load_native_extension

extension = load_native_extension()
topology, dynamics, initial, field, receptors, neuron_region = make_system(
    device="cuda", dtype=torch.float32, batch=2
)
runtime = NativeDopamineMatrixRuntime(
    topology,
    dynamics,
    initial,
    chemistry=field,
    receptors=receptors,
    neuron_region=neuron_region,
    config=DopaminePlasticityConfig(
        learning_rate=0.0,
        track_eligibility=False,  # avoid an O(B*M) scan for inference
        chemistry_interval=4,
        validate_inputs=False,
    ),
    blocks=4096,
    extension=extension,
)
rpe = torch.zeros((2, field.n_sources), device="cuda", dtype=torch.float32)

with torch.inference_mode():
    for tick in range(16):
        rpe.fill_(0.25 if tick % 8 == 0 else 0.0)
        state = runtime.step(0.15, rpe=rpe)
```

Use `validate_inputs=False` only for a trusted pipeline that already guarantees
finite RPE and non-negative finite release. Shape, dtype, and device are still
checked. Invalid values in fast mode are outside the runtime contract. Native
dopamine runtimes require CUDA FP32, accept a scalar `drive`, and do not
implement autograd. Leave `track_eligibility=False` for pure inference;
enabling it adds `O(B*M)` work even when `learning_rate=0`.

For a neutral real-time control profile where dopamine is logged but does not
modulate neural drive or weights, set `drive_scale=0`, `log_gain_scale=0`,
`learning_rate=0`, `weight_decay=0`, and `refresh_receptors=False`. Chemistry
still advances, but the repeated `BN` receptor-refresh phase is skipped. This
profile is not appropriate when D1/D2 modulation is part of the behavior.

## NativeDopamineEventRuntime

The native event runtime keeps `persistent_current = W @ emitted_state` and
propagates only source deltas whose absolute residual reaches `theta`.
`DopaminePersistentRuntime` is the portable semantic reference with the same
dopamine/PELS contract, but it uses a full sparse matrix multiply for emitted
deltas and is not the low-activity native optimization.

```python
from neurosim_gpu import NativeDopamineEventRuntime
from neurosim_gpu.native import load_native_extension

extension = load_native_extension()
topology, dynamics, initial, field, receptors, neuron_region = make_system(
    device="cuda", dtype=torch.float32, batch=1
)
runtime = NativeDopamineEventRuntime(
    topology,
    dynamics,
    initial,
    chemistry=field,
    receptors=receptors,
    neuron_region=neuron_region,
    theta=0.02,  # scalar or finite non-negative [N]
    config=DopaminePlasticityConfig(
        learning_rate=0.0,
        track_eligibility=False,  # avoid an O(B*M) scan for inference
        chemistry_interval=4,
        validate_inputs=False,
    ),
    scatter_blocks=128,
    extension=extension,
)
rpe = torch.zeros((1, field.n_sources), device="cuda")

with torch.inference_mode():
    for _ in range(16):
        state = runtime.step(0.15, rpe=rpe)
    active_sources = runtime.active_count.item()  # explicit host sync
```

`theta=0` emits every delta and matches matrix PELS semantics up to floating
point ordering. Positive `theta` is bounded residual approximation, not an
exact neural trajectory. There is not yet a device-side automatic matrix/event
switch, so choose the matrix runtime at high emission density and the event
runtime only after measuring activity. `collect_stats=True` reads the active
count and synchronizes. `active_count`, `RuntimeStats.active_neurons`, and
`DopaminePersistentStats.active_neurons` count active `(batch, neuron)` pairs,
not unique neuron IDs; `RuntimeStats.active_edges` is likewise batch-expanded.

Weight changes made through the integrated plasticity path trigger an exact
`W @ emitted_state` resynchronization. External updates must use
`topology.apply_source_weights_(new_source_major_weight)`. Directly mutating
only `topology.weight` or `topology.in_weight` leaves the two graph views and
runtime cache inconsistent.

For a portable bounded-error correctness reference, use
`DopaminePersistentRuntime` with the same constructor arguments and `theta`.
It updates emitted-state semantics correctly but propagates deltas with a full
SpMM; it is not the event-sparse performance implementation.

## Causal step order

Every integrated dopamine runtime implements this order at tick `t`:

1. `W_t`, `C_t`, and `s_t` produce `s_(t+1)`.
2. `s_t` and `s_(t+1)` update pre/post traces and edge eligibility.
3. The supplied RPE/release enters delay and multirate chemistry state.
4. At a chemistry boundary, chemistry produces `C_(t+1)`.
5. `C_(t+1)` gates an optional update from `W_t` to `W_(t+1)`.
6. New concentration, modulation, state, and weights are committed.

Therefore reward supplied to `step` cannot change the state returned by that
same call. It can change the next transition. A weight update likewise affects
only the following neural transition.

## Exact B=1 online plasticity

A shared weight vector cannot exactly apply different simultaneous batch
updates without defining a reduction. The integrated online rule is therefore
exact for dopamine-gated learning only at `B=1`; construction rejects
`batch > 1` when `learning_rate > 0`.

```python
topology, dynamics, initial, field, receptors, neuron_region = make_system(
    batch=1
)
runtime = DopamineMatrixRuntime(
    topology,
    dynamics,
    initial,
    chemistry=field,
    receptors=receptors,
    neuron_region=neuron_region,
    config=DopaminePlasticityConfig(
        pre_trace_decay=0.9,
        post_trace_decay=0.9,
        eligibility_decay=0.95,
        potentiation_scale=1.0,
        depression_scale=1.0,
        learning_rate=0.01,
        weight_decay=0.0,
        minimum_weight=-1.0,
        maximum_weight=1.0,
        preserve_weight_sign=True,
        plasticity_updates_magnitude=True,
        plasticity_interval=1,
        chemistry_interval=1,
    ),
)

before = topology.weight.clone()
for value in (0.5, 0.0, -0.25, 0.0):
    rpe = torch.tensor([[value]], dtype=initial.dtype)  # [1, S=1]
    runtime.step(0.15, rpe=rpe)
after = topology.weight
print((after - before).abs().max())
```

By default the plasticity update acts on weight magnitude and restores the
existing sign. This lets LTD weaken an inhibitory edge instead of making it
more negative. `preserve_weight_sign=True` also prevents sign crossing.
`plasticity_interval` counts chemistry updates, not neural ticks. Weight decay
and learning require `track_eligibility=True`.

For native online plasticity, use `B=1` whenever either `learning_rate` or
`weight_decay` can change weights. The current constructor catches `B>1` with
nonzero learning rate; `B>1` with decay alone reaches the native kernel's
`[1, N]` assertion and should also be treated as unsupported.

## Exact offline batch plasticity

For `B > 1`, preserve each sample's eligibility-dopamine correlation in
`[B, M]` storage, then reduce their product with
`dopamine_gated_batch_weight_update`. Do not multiply independently averaged
eligibility and dopamine.

```python
from neurosim_gpu import dopamine_gated_batch_weight_update

topology, _, _, _, receptors, _ = make_system(batch=3)
b, n, m = 3, topology.n_neurons, topology.n_edges
pre = torch.rand((b, n), dtype=topology.weight.dtype)
post = torch.rand((b, n), dtype=topology.weight.dtype)
src = topology.src.to(torch.int64)
dst = topology.dst.to(torch.int64)

# Maintain this [B, M] trace per sample over an offline rollout.
eligibility_by_batch = pre.index_select(1, src) * post.index_select(1, dst)
concentration_by_batch = torch.tensor(
    [[0.70] * n, [0.30] * n, [0.50] * n], dtype=topology.weight.dtype
)
dopamine_by_batch = receptors.response(
    concentration_by_batch
).net_phasic  # [B, N]

update = dopamine_gated_batch_weight_update(
    topology.weight,
    eligibility_by_batch,
    dopamine_by_batch,
    dst=topology.dst,
    learning_rate=0.01,
    minimum_weight=-1.0,
    maximum_weight=1.0,
    preserve_sign=True,
    update_magnitude=True,
    batch_reduction="mean",
)
topology.apply_source_weights_(update.weight)
assert update.delta.shape == (m,)
```

All four tensors (`weight`, `[B,M]` eligibility, `[B,N]` dopamine, and `dst`)
must share a device. The API is exact for the supplied batch reduction, costs
`O(B*M)`, and is intended for offline or small-batch use.

## Delayed multi-channel communication

`DelayedMultiChannelCommunication` is a standalone fixed-graph module. Edge
sign comes from `weight`; `transmitter` selects a receptor channel. Delay `d`
at tick `t` reads `s_(t-d)`, and missing pre-history is zero.

```python
from neurosim_gpu import DelayedMultiChannelCommunication

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
communication = DelayedMultiChannelCommunication(
    n_neurons=4,
    src=torch.tensor([0, 0, 1], dtype=torch.int64, device=device),
    dst=torch.tensor([1, 2, 3], dtype=torch.int64, device=device),
    weight=torch.tensor([1.0, -0.5, 0.8], dtype=dtype, device=device),
    delay=torch.tensor([0, 2, 1], dtype=torch.int64, device=device),
    transmitter=torch.tensor([0, 1, 0], dtype=torch.int64, device=device),
    receptor_gain=torch.tensor(
        [
            [1.0, 1.0],
            [1.2, 0.5],
            [0.8, 1.5],
            [1.0, 0.7],
        ],
        dtype=dtype,
        device=device,
    ),  # [N, C]; use [C] for global gains
)

for value in (1.0, 2.0, 4.0, 8.0):
    state_t = torch.tensor(
        [[value, 0.25, 0.0, 0.0]], dtype=dtype, device=device
    )
    current_t = communication.step(state_t)  # [B=1, N=4]
    print(communication.tick, current_t)
```

The module allocates a `[max_delay+1, B, N]` history ring on first use. If the
batch size changes, call `communication.reset_history(new_batch_size)` first.
An all-zero-delay graph bypasses the ring. This module is not yet wired into
the integrated dopamine runtime; an owning simulation must call it and apply
the returned current in its own causal schedule.

## Migration and structural connectome update

Migration is an `O(N)` slow-timescale position update. Noise is supplied by the
caller so checkpoint/replay behavior stays deterministic.

```python
from neurosim_gpu import (
    MigrationConfig,
    StructuralPlasticityConfig,
    migrate_neurons,
    update_structural_connectome,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dtype = torch.float32
positions = torch.tensor(
    [[0.1, 0.1, 0.1], [0.3, 0.1, 0.1],
     [0.7, 0.8, 0.8], [0.9, 0.8, 0.8]],
    dtype=dtype,
    device=device,
)  # [N=4, 3]
target = torch.full_like(positions, 0.5)
gradient = torch.tensor(
    [[0.1, 0.0, 0.0]] * 4, dtype=dtype, device=device
)
positions = migrate_neurons(
    positions,
    target,
    gradient,
    motility=torch.tensor([1.0, 0.5, 0.5, 1.0], dtype=dtype, device=device),
    domain_minimum=0.0,
    domain_maximum=1.0,
    config=MigrationConfig(dt=0.1, maximum_speed=0.2),
    noise=None,
)
```

Structural plasticity scores only an explicit candidate list. Candidate
generation belongs in a spatial index; never pass all `N*N` pairs.

```python
existing_src = torch.tensor([0, 2], dtype=torch.int64, device=device)
existing_dst = torch.tensor([1, 3], dtype=torch.int64, device=device)
existing_weight = torch.tensor([0.2, -0.2], dtype=dtype, device=device)

# Include existing edges only when they should be evaluated for pruning.
candidate_src = torch.tensor([0, 1, 2, 3], dtype=torch.int64, device=device)
candidate_dst = torch.tensor([1, 2, 3, 0], dtype=torch.int64, device=device)
affinity = torch.tensor([0.1, 0.95, 0.1, 0.9], dtype=dtype, device=device)
activity = torch.ones(4, dtype=dtype, device=device)  # values in [0, 1]
neuron_type = torch.tensor([0, 0, 1, 1], dtype=torch.int64, device=device)
source_sign = torch.tensor([1, 1, -1, -1], dtype=dtype, device=device)
compatibility = torch.ones((2, 2), dtype=torch.bool, device=device)

result = update_structural_connectome(
    positions,
    activity,
    existing_src,
    existing_dst,
    existing_weight,
    candidate_src,
    candidate_dst,
    affinity,
    neuron_type,
    source_sign,
    compatibility,
    config=StructuralPlasticityConfig(
        distance_weight=0.0,
        activity_weight=0.0,
        chemical_weight=1.0,
        formation_threshold=0.8,
        prune_threshold=0.3,
        maximum_new_edges_per_source=2,
        initial_weight_magnitude=0.05,
    ),
)
new_topology = SparseTopology.from_edges(
    positions.shape[0], result.src, result.dst, result.weight
)
print(result.n_formed, result.n_pruned, new_topology.n_edges)
```

`activity` `[N]` and `chemical_affinity` `[K]` must be finite in `[0,1]`.
Candidate and existing edge lists must contain no duplicates. New edge signs
come from the presynaptic `source_sign`, whose entries must be exactly `-1` or
`+1`. `maximum_new_edges_per_source` is a formation budget for one structural
epoch, not a cap on total source degree. Existing edges absent from the
candidate list are preserved.

Topology rebuilding is a slow structural boundary. Existing runtime objects do
not automatically adopt `new_topology`; use `MultiRateScheduler.replace_topology`
or construct/remap a runtime explicitly. The scheduler resets topology-aligned
eligibility and persistent current and can rebuild delayed communication through
`communication_rebuilder`; optimizer-specific state remains the caller's
responsibility.

## Related public helpers

The top-level package also exports smaller composable APIs:

| API | Purpose |
|---|---|
| `DenseOracle` | Small dense FP32/FP64 correctness oracle |
| `random_topology` | Seeded random sparse topology builder for tests/benchmarks |
| `DopaminePersistentRuntime` | Portable full-SpMM dopamine PELS reference |
| `update_eligibility_trace` | Standalone source-major `[M]` local trace update |
| `dopamine_gated_weight_update` | Standalone reduced three-factor weight update |
| `modulate_neuron_dynamics` | Standalone dopamine drive/gain modulation |
| `score_candidate_edges` | Score explicit structural candidates without updating |
| `RuntimeStats`, `DopamineStepStats`, `DopaminePersistentStats` | Results returned by `collect_stats=True` |
| `PlasticityUpdate`, `CandidateEdgeScores`, `ConnectomeUpdate` | Structured helper return values |

Use the integrated runtimes when their causal ownership matches the model. Use
the standalone helpers when an external scheduler owns traces, batching,
chemistry, communication, or structural epochs.

## Common errors and performance caveats

- `ValueError: ... must share a device/dtype`: construct every participating
  tensor on the state/topology device with the same floating dtype.
- `native dopamine runtime requires CUDA float32 state`: use CUDA FP32 for all
  neural, chemical, receptor, and weight floating tensors.
- `exact shared ... requires batch=1`: use `B=1` online learning or the exact
  offline batch API; do not silently average independent factors.
- `dt ... violates ... stability limit`: lower chemistry `dt` or diffusion and
  clearance rates. Inspect `field.maximum_stable_timestep`.
- `receptor tonic concentration must match ... baseline`: use the assigned
  region's tonic baseline for every neuron.
- `theta must be ...`: provide a finite non-negative scalar or `[N]` tensor.
- Delayed communication batch changes require `reset_history`.
- Safe validation, `collect_stats`, `.item()`, and Python `float(tensor)` cause
  CUDA synchronization. Keep them out of timed fast-path loops.
- Native matrix/event runtimes are fixed choices today. There is no automatic
  device-side policy, CUDA graph capture, integrated delayed communication, or
  optimizer-aware structural remap transaction yet.
- `DopamineMatrixRuntime.step` and all native dopamine steps are online
  state-mutating APIs under `torch.no_grad()`. Use the base PyTorch runtime for
  BPTT; native whole-rollout recurrent backward is not implemented.
- Core topology, dynamics, chemistry, receptor, and runtime objects are plain
  stateful Python objects: they have no `.to()` or complete `state_dict()`.
  Construct them directly on the final device and explicitly checkpoint every
  evolving tensor plus tick/accumulator/delay state needed for exact resume.
- Delayed communication is an `nn.Module`, but its live history ring is marked
  non-persistent and its cursor/tick are Python state. Its normal `state_dict()`
  is not an exact mid-sequence checkpoint; reset history or save that state in
  an owning simulation checkpoint.

## Prebuilt topology and inference memory

Random topology generation is for tests and small experiments. For a real-time
deployment, serialize a canonical topology once and load it without random
generation or dense allocation (validation is still performed):

```python
from neurosim_gpu import load_topology, save_topology

save_topology(topology, "connectome.pt")
loaded = load_topology("connectome.pt", device="cuda")
```

The validated loader is intended for correctness and moderate-scale startup.
At very large `M`, its safety checks and device copies can create a substantial
temporary peak; it is not yet a zero-copy 125M-edge loader.

For read-only native inference, load a single-weight destination-major view:

```python
view = load_topology("connectome.pt", device="cuda", single_weight=True)
```

Online learning requires the mutable `SparseTopology` with both graph views.
Loading a topology resets its runtime weight version by default; persistent
currents should be rebuilt from the loaded emitted state.

## Checkpoint, scheduler, and continual learning

`MultiRateScheduler` makes neural ticks, communication, migration, and
structural epochs explicit. `capture_runtime`/`restore_runtime` preserve
runtime tensors, topology weights/version, chemical accumulators, PELS state,
and optional delayed-communication history:

```python
from neurosim_gpu import (
    MultiRateScheduler,
    SchedulerConfig,
    capture_runtime,
    restore_runtime,
)

scheduler = MultiRateScheduler(
    runtime,
    config=SchedulerConfig(migration_interval=1000, structural_interval=10000),
)
for _ in range(100):
    state, info = scheduler.step(0.0, rpe=rpe)
checkpoint = scheduler.state_dict()
restore_runtime(runtime, checkpoint)
```

Structural callbacks must return a complete replacement topology. When a
communication module is attached, supply `communication_rebuilder`; otherwise
the scheduler rejects the replacement instead of silently using a stale graph.
Replacement resets topology-aligned eligibility and rebuilds persistent current.

For embodied continual learning, `ReplayBuffer` is a fixed-capacity ring with
checkpointable RNG state and `ContinualLearningController` couples it to the
scheduler:

```python
from neurosim_gpu import ContinualLearningController, ReplayBuffer

replay = ReplayBuffer(100000, device=runtime.state.device, seed=7)
controller = ContinualLearningController(scheduler, replay)
next_state = controller.step(0.0, rpe=rpe)
controller.observe(state, action, reward, next_state, terminal)
batch = controller.replay_batch(64)
```

The controller does not choose a loss or optimizer; it provides deterministic
simulation/replay state so those choices can be tested separately.

This checkpoint covers simulator/replay state only. Save environment state,
sensor streams, policy/optimizer state, migration candidate indices, and any
external RNG separately for exact episode replay. Replay RNG reproduction
assumes the same device family and a compatible PyTorch version.

## Real-time latency measurement

Throughput averages can hide control-loop stalls. Use the per-tick percentile
benchmark for embodied control deadlines:

```powershell
scripts\native_env.bat python benchmarks\realtime_latency.py --mode matrix --deadline-ms 5
scripts\native_env.bat python benchmarks\realtime_latency.py --mode event --theta 0.05 --deadline-ms 5
```

The benchmark synchronizes around every tick and reports min, p50, p95, p99,
max, mean, and deadline miss rate. This is intentionally slower than the
throughput benchmark and measures host-observed control latency. Use a saved
prebuilt topology for deployment-scale runs.

For a GPU-free theoretical screen of a configuration:

```powershell
python benchmarks\theoretical_cost.py --neurons 166000 --edges 125000000 --activity 0.05 --json
```

This reports work ratios and byte estimates, not wall-clock latency.

The default report is the fixed core plus chemistry. Add
`--communication --eligibility --plasticity --development` to include the full
embodied envelope. The model deliberately does not allocate tensors or touch a
GPU.

---

[← Documentation index](index.md)
