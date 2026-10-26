# Functional Neurochemical Model

## Scope

This project targets a functional neurochemical whole-network model. It is more
expressive than a generic rate RNN, but it is not a molecular, compartmental,
or clinically validated brain simulation. Dopamine is modeled as one regional
chemical field and may encode reward prediction error, salience, novelty,
movement, or externally supplied release. Throughput is not evidence of
biological validity, consciousness, or human-equivalent behavior.

## Dopamine Semantics

Dopamine concentration is always non-negative. Positive and negative dopamine
mechanisms mean deviations from a tonic fixed point, not positive and negative
molecules:

```text
positive RPE -> release burst -> concentration above tonic
negative RPE -> suppressed tonic release -> concentration dip, still >= 0
```

For an optional temporal-difference critic:

```text
delta = clip(reward + gamma * (1-terminal) * next_value - value)
```

The runtime also accepts non-RPE source release. This is intentional because
real dopamine activity is not exclusively a reward-prediction signal.

## Regional Field

For region `r` with volume `V_r`, concentration `C_r`, clearance `k_r`, release
`q_r`, and symmetric edge conductance `G_rl`:

```text
dC_r/dt = q_r - k_r C_r + (1/V_r) sum_l G_rl (C_l - C_r)
```

The explicit step is accepted only when its positivity CFL condition holds.
With no release or clearance, `sum_r V_r C_r` is conserved. Tonic release is
derived from the same discrete operator so the configured tonic concentration
is an actual fixed point. Optional D2-like autoreceptor feedback attenuates only
the excess burst above tonic, preserving that fixed point.

Chemistry can advance every `K_c` neural ticks. RPEs are averaged into a release
rate over the interval, concentration and receptor modulation are held between
boundaries, and a delay ring models transport latency.

## Receptors And Neural Dynamics

Receptor occupancy uses a Hill response:

```text
O_q(C) = C^h / (K_q^h + C^h)
delta_O_q = O_q(C) - O_q(C_tonic)
```

Both D1 and D2 occupancy increase with concentration. Functional pathway signs
come from cell-specific coupling:

```text
D1 contribution = +rho_D1 * delta_O_D1
D2 contribution = -rho_D2 * delta_O_D2
net phasic signal = D1 contribution + D2 contribution
```

Thus a burst is positive for a pure D1-coupled cell and negative for a pure
D2-coupled cell; a dip reverses both. This direct/indirect interpretation is a
configurable basal-ganglia proxy, not a universal rule for every brain region.

The default runtime applies the signal as a bounded additive drive. Optional
multiplicative gain modulation is disabled by default and clamped when enabled,
because changing gain also changes the recurrent stability bound.

## Three-Factor Plasticity

Neuron pre/post traces and one bounded eligibility value per edge are updated
from local coactivity. At a chemical boundary:

```text
delta_|w_ij| = learning_rate * eligibility_ij * phasic_signal_i
w_ij' = sign(w_ij) * (|w_ij| * (1-decay) + delta_|w_ij|)
```

For positive eligibility, a burst produces D1-like potentiation and D2-like
depression; a dip reverses them. The integrated runtime updates synaptic
magnitude and reapplies the existing Dale sign, so LTD also weakens an
inhibitory edge rather than making it more negative. Projection preserves
configured weight bounds and optionally the existing sign. This is a functional
three-factor learning proxy, not a claim about one universal molecular LTP/LTD
law.

## Delayed Multi-Channel Communication

Each fixed edge carries an independent signed weight, integer conduction delay,
and transmitter-channel ID. A channel selects a global or destination-local
receptor gain; it does not determine excitatory/inhibitory sign. A causal ring
stores `D+1` neural states, so delay `d` reads exactly `s_(t-d)` and missing
pre-history is zero. The all-zero-delay case bypasses the ring.

Work is `O(BM + BN)` per tick and history storage is `O(BN(D+1))`. This is a
functional axonal-delay/transmitter model, not vesicle release, receptor kinetic
state, or detailed dendritic integration.

## Migration And Connectome Development

Slow developmental migration stores one 3-D position per neuron. Bounded target
attraction, an externally supplied chemotactic gradient, motility, optional
deterministic noise, maximum speed, and domain bounds define an `O(N)` update.

Structural formation never performs an all-pairs search. A spatial index must
provide at most `K` candidate edges. Formation scores combine distance,
activity and chemical affinity; hysteretic thresholds prune and form edges,
with deterministic per-source top-k limits, cell-type compatibility, and source
sign constraints. Canonical rebuilding is a slow `O(M log M + K log K)` step
amortized over a much longer structural interval.

## Causal Tick

The integrated runtime uses this order:

1. `W_t` and `C_t` produce `s_(t+1)`.
2. `s_t` and `s_(t+1)` update local traces and eligibility.
3. The outcome supplies RPE or other release signals.
4. The delay/multirate chemistry boundary produces `C_(t+1)`.
5. `C_(t+1)` gates a weight update to `W_(t+1)`.

The reward cannot change the transition that produced it. A changed weight can
affect only the next transition. In a persistent-current runtime, every weight
update is followed by an exact `current = W @ emitted_state` resynchronization.

Exact shared online plasticity currently requires `B=1`. The separate batch API
stores `[B,M]` eligibility and reduces the within-sample
`eligibility*dopamine` product. Multiplying two independently batch-averaged
signals is explicitly not treated as exact.

`validate_inputs=True` is the safe research mode and synchronously rejects
non-finite RPE or invalid release before state mutation. A benchmark or trusted
pipeline may select `validate_inputs=False` to keep the CUDA tick asynchronous;
invalid values in that mode are outside the runtime contract.

Integrated dopamine runtime steps use `torch.no_grad`: plasticity is the local
online three-factor rule described above. BPTT through regional chemistry,
receptor state, and online weight mutation is not implemented.

`dopamine_delay_steps` delays the RPE path. Explicit `source_release` is treated
as an immediate exogenous input and is not placed in that delay ring.

For real-time control experiments that do not use dopamine to modulate neural
drive, `refresh_receptors=False` skips receptor-cache refresh while preserving
the regional concentration trajectory. This is an explicit simplified profile,
not a biologically complete dopamine mode.

## Complexity

```text
neural propagation       O(BM)
neuron/receptor gather   O(BN)
regional chemistry       O(B(R + E_R + S R)) every K_c ticks
eligibility              O(BN) trace kernel + O(BM) edge kernel
weight projection        O(M) for exact B=1 online learning
```

With bounded-degree region graphs and small source count, amortized work is
`O(BN + BM + B(R + E_R + S R)/K_c)`. Structural development and migration
belong to a slower schedule and must never trigger all-pairs neuron searches.

`DopaminePersistentRuntime` is the portable reference and uses a full SpMM for
emitted deltas. `NativeDopamineEventRuntime` implements the sparse source-event
path with cached receptor modulation. Both resynchronize exactly after weight
changes; performance claims must distinguish these two classes.

The native event propagation itself is `O(BN + M_A)`. Default full eligibility
tracking remains `O(BM)` every tick, and a plasticity boundary performs `O(M)`
weight projection plus current reconstruction. It is not correct to describe
the whole plastic runtime as event-sparse under those settings.

Any PELS state-error certificate is tied to the weights and maximum gain used
to construct it. Plasticity or gain modulation requires recalibration or a
worst-case certificate before the old bound may be claimed.

## Validation Boundaries

The implementation tests non-negative concentration, tonic equilibrium,
volume-weighted diffusion conservation, batch isolation, burst/dip direction,
D1/D2 sign reversal, receptor saturation, TD termination, delay timing,
eligibility bounds, weight sign/bounds, causal no-future-leakage, theta-zero
matrix/persistent equivalence, and persistent-current resynchronization.
Delayed communication tests exact timing, ring wraparound, channels, signed
edges and batch isolation. Development tests migration bounds, determinism,
candidate-only formation/pruning, top-k budgets and type/sign constraints.

---

[← Documentation index](index.md)
