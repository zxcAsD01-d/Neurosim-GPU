# Gomoku CNN vs Sparse Brain Protocol

This protocol defines a small, reproducible comparison between a convolutional
policy/value model and the project's sparse rate-network runtime. It is a
benchmark specification, not evidence that a rate model is biologically
equivalent to a brain.

## Scope

Use a 9x9 board with a five-in-a-row win condition for the first pass. Keep
the environment on CPU and use a small, isolated CUDA run only for neural
timing. A 15x15 board is a follow-up scale/generalization point; it must not be
used to extrapolate whole-brain real-time performance without a measured
topology and latency distribution.

The current `benchmarks/gomoku_learning.py` is a supervised tactical smoke
test. It supports deterministic line templates and a `random-rollouts` source
of legal environment positions, but it deliberately does not claim self-play
strength, Elo, or dopamine learning. Its sparse model is a differentiable
rate-network adapter trained by AdamW; it does not exercise
`DopamineMatrixRuntime` or PELS during training.

## Shared observation and labels

Every position is represented from the side-to-move perspective. The current
runner uses three binary planes (own stones, opponent stones, and empty
intersections) plus a separate legal-action mask. A richer embodied protocol
may add legal, last-move, turn, and normalized move-count planes, but the exact
channel count must be identical for both models.

Do not canonicalize board symmetries unless the corresponding action index is
transformed by the same rotation/reflection. The environment provides
`canonical_symmetry_index`, `transform_move`, and `canonical_move`; the first
runner still uses `symmetry=False` to keep replay-side transforms out of the
timed path.

Split by complete games or independently generated position seeds. Splitting
adjacent positions from the same game leaks nearly identical states into the
validation set. Use at least three, preferably five, independent seeds and
report a paired confidence interval.

## Model tracks

### Track A: architecture and compute upper bound

Train both models on the same policy/value targets with the same optimizer,
batching, update count, and learning-rate search budget. The loss is:

```text
cross_entropy(masked_policy_logits, action) + 0.25 * mse(value, outcome)
```

The CNN baseline can use a 3x3 stem, a small stack of residual/conv blocks,
and policy/value heads. Count every trainable convolution, linear, bias, and
normalization scalar.

The sparse brain adapter should expose fixed sparse sensory edges, signed
recurrent edge values, and policy/value readouts. The current runner uses a
fixed `sensory_fanout=4` per input feature, so sensory storage is
`O(F * fanout)` rather than a dense `F*N` matrix. For a larger fair pilot use
roughly `N=1024`, `M=70k`, and sparse policy readout (`k=16` edges per action),
then adjust `M` until trainable scalars are within 5% of the CNN. Frozen
sensory weights and index buffers are not trainable parameters: report them
separately as fixed storage and fixed compute. Matching only learned scalars is
not the same as matching total memory or bandwidth. The runner exposes
`--parameter-match trainable|effective`; use `effective` when comparing total
learned plus frozen scalar state, and report both counts in either case.

The runner's `--brain-learn-sensory` flag makes the sparse sensory amplitudes
trainable. Use it as a separate parameter-matched ablation; do not mix its
results with the default fixed-reservoir condition.

For a recurrent decision, report the number of neural ticks `K` explicitly.
The per-move work estimate is approximately:

```text
brain: K * (M + N) + F * fanout + policy/value_readout
CNN:   measured convolution/linear work for one policy/value forward
```

The estimate is only a bookkeeping aid. Sparse kernels, launch overhead,
memory traffic, and host synchronization require measured p50/p95/p99 latency.

Run the brain first with exact propagation (`theta=0` or dense/CSR BPTT). Add
PELS thresholds as a separate ablation (`theta=0.01`, `0.05`) and report state
error and active-edge fraction. PELS reduces propagation work only; a full
eligibility pass still scans all `M` edges.

### Track B: online brain learning

Use `DopamineMatrixRuntime` or `DopaminePersistentRuntime` with `B=1` and a
policy/value output population. A starting configuration is:

```text
track_eligibility=True
chemistry_interval=4 or 8
plasticity_interval=1
dopamine_delay_steps=0
validate_inputs=True during correctness, False only inside timed loops
```

At an action boundary, compute the TD signal

```text
RPE = reward + gamma * (1 - terminal) * V_next - V
```

and inject it on the next observation tick. The runtime first advances with
the concentration at entry (`C_t`) and only then advances chemistry toward
`C_(t+1)`, so injecting a reward before the action would leak future
information. Record the intentional one-tick pending-RPE delay.

The current generic eligibility rule is not action-specific: dopamine gates an
edge using its destination signal, but does not know which stochastic action
was selected. A global RPE over all policy output edges is therefore not a
valid actor-credit implementation by itself. Use one of these explicit
variants:

1. train the readout with an external policy-gradient/Adam update and label the
   system hybrid;
2. add a selected-action eligibility trace to the incoming edges of the chosen
   action neuron; or
3. report the unmodified runtime as a value-learning/representation ablation,
   not as a complete actor-critic result.

Compare CNN Adam, sparse-RNN Adam, and dopamine eligibility as separate
learning-rule conditions. Do not attribute a difference caused by the optimizer
to the network architecture.

## Required metrics

Report all metrics against the same seed and workload:

- trainable scalar count, fixed topology/index bytes, optimizer-state bytes;
- train/validation policy top-1 accuracy, legal-action rate, value MSE and
  signed-value accuracy;
- positions processed, update count, wall-clock milliseconds, samples/second;
- inference and complete decision p50/p95/p99 latency, deadline miss rate;
- peak allocated/reserved GPU memory and (when available) energy per move;
- for PELS: `theta`, active-neuron fraction, active-edge fraction, and state
  error against the exact reference;
- for online play: environment moves, network calls, win rate against random
  and tactical opponents, color-swapped paired Elo, and time/compute to reach a
  predefined win-rate target.

Use three horizontal axes for learning curves: environment moves (sample
efficiency), measured GPU milliseconds (compute efficiency), and wall-clock
time (deployment efficiency). An average throughput ratio alone is not a
real-time result; use the p99 deadline miss rate.

## Baselines and ablations

The minimum table is:

1. random and deterministic tactical opponents;
2. CNN policy/value;
3. sparse brain with exact CSR propagation and AdamW;
4. sparse brain with PELS thresholds;
5. sparse brain with dopamine/eligibility (B=1);
6. no-dopamine and no-eligibility controls;
7. optional shallow MLP or dense rate-RNN control.

If search is added, use the same search algorithm, number of simulations, and
network-call budget for both models. Do not give one model MCTS while the other
selects a raw policy.

## Reproducibility commands

The current smoke test is intentionally small:

```powershell
.venv\\Scripts\\python.exe benchmarks\\gomoku_learning.py --device cpu --samples 512 --epochs 3 --batch 32 --json
# Optional legal-position distribution shift:
.venv\\Scripts\\python.exe benchmarks\\gomoku_learning.py --device cpu --dataset-source random-rollouts --samples 512 --epochs 3 --batch 32 --json
```

An isolated small GPU run is explicit:

```powershell
scripts\\native_env.bat python benchmarks\\gomoku_learning.py --device cuda --samples 512 --epochs 3 --batch 32 --parameter-match effective --json
```

Record GPU process/utilization before and after the run. Do not publish a
latency number if another process is using the GPU. The runner's template and
random-rollout teacher tasks are useful for checking action masking and
optimization plumbing; a self-play wrapper is required before reporting game
strength or continual-learning claims.

---

[← Documentation index](index.md)
