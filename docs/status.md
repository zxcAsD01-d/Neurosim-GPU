# Status and open work

The target is a neural simulation tick whose cost is comparable to a model with
the same `M` sparse parameters acting on an `N`-dimensional state.

## Current Scorecard

| Area | Target | Current result | Status |
|---|---:|---:|---|
| Fixed core forward, B=1 | neural/raw <= 1.05x | 0.742-0.821x at 10-25M edges | Achieved |
| Fixed core forward, B=8 | neural/raw <= 1.15x | 1.148x at 1M edges | Achieved at tested point |
| Training F+B | neural/RNN <= 1.05x | 1.075x at 1M edges, T=32 | 93% efficiency; 2.5 points remain |
| Target neuron count | N=166k | N=166k executed | Achieved |
| Target connection count | M=125M | M=49.95M executed | 40% scale validation |
| Read-only runtime topology | about 2 GB at 125M | 16M+8N schema/view implemented | 125M startup unverified |
| Online-learning topology | bounded practical memory | 24M+8N bytes currently | About 3.0 GB at 125M |
| Exact path | theta=0 forward/gradient | Reference tests pass | Achieved |
| Bounded event path | lower than matrix at low activity | 3-4.5x faster near 2.64% edges | Achieved in fixed event mode |
| Automatic hybrid | switch without CPU sync | Not implemented | Open |
| Native training adjoint | checkpointed full rollout | Pointwise analytic backward only | Partial |
| Regional dopamine | tonic/burst/dip, D1/D2, diffusion, uptake | Integrated and tested | Functional proxy achieved |
| DA-gated plasticity | causal three-factor rule | Matrix and PELS paths tested | Achieved |
| Neurochemical native path | retain matrix-like tick cost | Fused matrix/event, eligibility and weight kernels | Implemented; clean benchmark pending |
| Delayed communication | per-edge delay and channel | Causal ring and channel gains tested | Functional proxy achieved |
| Migration | bounded slow O(N) update | Position/chemotaxis module tested | Primitive achieved |
| Connectome development | no N^2 formation | Explicit candidates, pruning and top-k formation | Primitive achieved |
| Topology deployment | prebuilt load without random builder | Compact save/load and single-weight inference view | Achieved |
| Runtime checkpoint | episode/preemption resume | PELS, chemistry, delay history and topology version | Achieved |
| Continual embodied loop | replay + multi-rate boundaries | Scheduler and deterministic replay controller | Achieved |

The current regression suite contains 177 tests (165 pass plus 1 CUDA-only and
11 optional `transformers` skips on a base CPU install), including topology
serialization, runtime/communication checkpoints, scheduler boundaries,
replay-buffer resume determinism, the Gomoku environment/CSR-view checks, and
the Cortex-LM contract/dry-run tests.

## Gomoku learning probe

The first task-level comparison is now runnable through
`benchmarks/gomoku_learning.py`. It uses a deterministic 9x9 tactical
supervised set with the same three-plane observation, legal-action masking,
policy/value loss, AdamW schedule, and validation split for both models. The
CNN is a compact convolutional policy/value baseline; the brain row is a
fixed-topology sparse rate-network BPTT upper bound with a fixed sparse sensory
fanout projection and learned recurrent/readout weights. The runner also has
an optional legal `random-rollouts` data source with an immediate-win/block/
center teacher; tactical-lines remains the default.

The runner reports trainable parameters separately from fixed sensory/topology
storage. A fixed projection is real compute and memory, so the trainable
parameter ratio must not be presented as total-model parity. It also reports a
per-epoch sample-efficiency curve and optional p50/p95/p99 per-position
inference latency. The current task is intentionally a plumbing/architecture
smoke test, not self-play strength or a dopamine-learning result; those are
separate protocol tracks in `gomoku-protocol.md`.

The current low-load CUDA smoke (`N=64`, recurrent `M=480`, `F*fanout=972`
sensory edges, 64 positions, one epoch, batch 16, effective-parameter match)
is diagnostic only. It measured 35.8 CNN versus 79.2 sparse-brain training
positions/s, with p50 decision latency 1.317 ms versus 3.763 ms and p99 2.309
ms versus 6.328 ms; validation policy accuracy was 7.69% versus 0%. The
fixed-sensory brain used 6,910 effective scalars versus 6,892 CNN scalars.
These values are not a claim of learning superiority: the task is tiny, the
sparse CSR view is rebuilt per forward, and the random fixed reservoir has a
weaker spatial inductive bias than convolution. The learned-sensory flag and a
larger multi-seed protocol are required before drawing a model-level
conclusion.

The real-time path now has a p50/p95/p99/deadline benchmark, while the
continual-learning path has an explicit multi-rate scheduler, topology-boundary
transaction, runtime/communication checkpoint, and deterministic replay ring.

The neutral real-time dopamine profile can skip receptor refresh when dopamine
is not used as a neural modulator; this removes a repeated `BN` tensor phase
without changing the chemistry state. GPU latency readings on 2026-10-12 were
contaminated by an unrelated process using about 4.8 GiB and 58% GPU, so they
are diagnostic only and are not in the accepted results table.

The CPU-only theoretical cost model now separates core/event propagation from
optional communication, eligibility, plasticity, and development terms. At the
125M/166k example with 5% activity, propagation-only event work is about 5.1%
of full matrix propagation; when full eligibility and other embodied terms are
enabled, the modeled event/full ratio rises to about 68.6%, showing why low
activity alone does not guarantee end-to-end speedup.

## Task-level probes

Beyond the runtime benchmarks, two task-level suites exercise the stack end to
end on CPU: MNIST digit recognition and inverted-pendulum stabilisation (105 grid
runs, 2–3 seeds). They are functionality and parameter-efficiency probes, not GPU
benchmarks. Headline results: a fixed sparse brain with 4,585 trainable scalars
reaches 91.1% on MNIST (parameter-matched MLP 87.3%, logistic regression 91.8%),
and emitting only 21% of the edges costs 0.12 accuracy points. On the control
task both cloned policies hold the 200-step ceiling; failures require extreme
delay or noise, and online dopamine-gated plasticity showed no measurable
benefit because the learned recurrence is nearly dispensable. Details, including
the negative results, are in [experiments](experiments.md) and
[../results/REPORT.md](../results/REPORT.md).

## Interpretation

The fixed-topology core forward objective is no longer blocked by neural dynamics. On the same
custom sparse kernel, adding leak, gain, tanh and clipping costs only a few
percent, and the fused tick is faster than the cuSPARSE comparison layer at
batch one. Remaining performance risk is concentrated in training backward,
large-batch low-degree graphs, automatic event switching, and the unexecuted
125M-edge scale point.

The dopamine subsystem now includes non-negative tonic/burst/dip chemistry,
volume-weighted regional diffusion, uptake/decay, transport delay, optional
autoreceptor feedback, D1/D2 occupancy, multirate updates, TD-RPE input, local
eligibility, sign-preserving plasticity, and PELS current resynchronization.
Absolute performance remeasurement is pending because the measurement device was
shared with an unrelated GPU workload during the 2026-09-27 session; correctness
and native/reference comparisons passed under that contention.

The next milestone is not another forward micro-optimization. It is an actual
125M-edge prebuilt topology run plus a whole-rollout custom adjoint that moves
training parity from 1.075x to at most 1.05x.

---

[← Documentation index](index.md)
