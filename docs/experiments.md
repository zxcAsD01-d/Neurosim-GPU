# Task-level experiments

`experiments/` contains two self-contained probes that exercise the runtime on
real tasks rather than microbenchmarks. Both need only PyTorch and the standard
library, so they run on a CPU-only install.

| File | Role |
|---|---|
| `common.py` | seeding, timing, percentile, latency sampling, MNIST download/cache, `RESULT` protocol |
| `mnist_experiments.py` | dense baselines vs the fixed sparse rate brain on handwritten digits |
| `cartpole_experiments.py` | inverted-pendulum stabilisation: LQR expert, behaviour cloning, robustness grid, online plasticity |
| `run_suite.py` | the experiment grid, subprocess driver, CSV writer |
| `analyze.py` | aggregates `results/raw/*.json` into `results/TABLES.md` |

Each script is a CLI that prints exactly one machine-readable line
`RESULT {json}`; the suite driver parses that line and writes the payload to
`results/raw/`. Every run is deterministic for a given `(config, seed)`:
accuracy and survival metrics reproduce bit-for-bit, only wall-clock latency
varies.

## What is compared

**MNIST.** A fixed sparse recurrent core (`N` neurons, `d` synapses per neuron,
`T` ticks, leaky-tanh dynamics), a fixed sparse fanout sensory projection and a
linear readout — trained by BPTT. Baselines: logistic regression, an MLP with
the same number of trainable scalars, a wider MLP, and a dense RNN with the same
`N` and `T`.

**Inverted pendulum.** Standard cart-pole physics (`dt=0.02`, 200 steps, 12°
termination), solved analytically by a discrete-LQR expert. Two policies are
behaviour-cloned from that expert: an MLP and the sparse brain. Evaluation is
closed-loop, not teacher-forced.

## Grid

| Suite | Runs | What varies |
|---|---:|---|
| `mnist_models` | 15 | model family (logreg, MLP ×2, sparse brain, dense RNN), 3 seeds |
| `mnist_degree` | 10 | synapses per neuron 2–32 |
| `mnist_neurons` | 8 | 64–512 neurons |
| `mnist_steps` | 10 | 1–16 recurrent ticks |
| `mnist_data` | 10 | 1k–20k training samples |
| `mnist_variants` | 12 | no recurrence, learned sensory fanout, wider fanout, leak, weight scale |
| `mnist_corruption` | 12 | Gaussian noise, occlusion, 2 px shift |
| `cartpole_expert` | 3 | LQR reference |
| `cartpole_bc` | 6 | MLP vs sparse brain, nominal protocol |
| `cartpole_hard` | 9 | same policies under the harder protocol |
| `cartpole_robustness` | 4 | noise, action delay, pole length/mass mismatch, impulse |
| `cartpole_online` | 6 | frozen vs dopamine-gated online plasticity after a plant change |

105 grid runs in total, plus roughly 77 exploratory runs kept under
`results/raw/tuning/` and excluded from the aggregates.

```powershell
python experiments/run_suite.py --list      # grid inventory
python experiments/run_suite.py --suite all # run it (~40 min on CPU)
python experiments/analyze.py               # results/TABLES.md
```

## Headline results

MNIST, 10k training digits, 10 epochs, 3 seeds (accuracy mean ± population std):

| Model | Test accuracy | Trainable scalars |
|---|---|---|
| Logistic regression | 91.77% ± 0.30 | 7,850 |
| MLP, matched to the brain | 87.27% ± 0.93 | 4,780 |
| MLP, wide baseline | 94.65% ± 0.04 | 50,890 |
| Sparse brain (N=256, d=8, T=8) | 91.10% ± 0.16 | 4,585 |
| Dense RNN (N=256, T=8) | 94.98% ± 0.32 | 269,066 |

- The sparse brain beats a parameter-matched MLP by 3.8 points but does not beat
  a linear classifier with 70% more parameters. Fixed topology buys parameter
  efficiency, not absolute accuracy.
- Emitting only the state deltas above a threshold keeps accuracy: at 21.1% of
  edges active the loss is 0.12 points; at 8.3% it is 0.55 points. This is the
  bounded-error persistent-current path measured on trained weights, forward
  pass only.
- The sparse brain is markedly more robust to input noise (72.9% vs 55.1% for
  the wide MLP at σ=0.3) but has no advantage under occlusion and is worse
  under a 2 px shift.

Inverted pendulum, hard protocol (initial angle ±0.15 rad, 6 N actuator, seeded
impulse):

- Both cloned policies are stable at the 200-step ceiling; behaviour-cloning
  accuracy (96.7% vs 88.7%) does not translate into a closed-loop difference.
- Failures require extreme conditions: action delay ≥6 steps (both collapse to
  ~30 steps) and sensor noise σ=0.2. Pole length ×4 collapses the MLP (38.8
  steps) while the sparse brain survives (198.7).
- Dopamine-gated online plasticity shows no measurable benefit (129.1 vs 130.3
  steps after a plant change). Zeroing every recurrent weight still leaves
  193.8/200 steps, so the learned recurrence is nearly dispensable and the
  plasticity rule has no behavioural leverage.

Full tables, methodology, failure log and caveats:
[../results/REPORT.md](../results/REPORT.md) and
[../results/TABLES.md](../results/TABLES.md).

## What these experiments do not establish

- They are **not GPU performance measurements**. All runs are CPU-only; the
  accepted GPU numbers are in [results](results.md).
- Behaviour cloning is **not reinforcement learning**. The policies imitate an
  LQR expert, so the ceiling is the expert itself.
- The MNIST runs use a 10k training subset, 2–3 seeds and no significance
  testing; treat relative ordering, not absolute percentages, as the finding.
- The online-plasticity result says the *testbed* gave the rule no leverage. It
  is not evidence for or against the dopamine rule itself.

---

[← Documentation index](index.md)
