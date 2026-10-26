# NeuroSim GPU

[![CI](https://github.com/zxcAsD01-d/Neurosim-GPU/actions/workflows/ci.yml/badge.svg)](https://github.com/zxcAsD01-d/Neurosim-GPU/actions/workflows/ci.yml)
[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

GPU-first sparse neural simulation: a fixed-topology neural core with
persistent-current event propagation, regional dopamine chemistry,
eligibility-based plasticity and a multi-rate scheduler.

For `N` neurons, `M` fixed synapses and batch `B`, one tick is a sparse matrix
multiplication plus a few vector operations:

```text
synaptic_current = W_csr @ state
state = clamp(alpha * state + (1 - alpha) * gain * tanh(current + drive))
```

That is `O(BM + BN)` time and `O(M + BN)` memory. Neurochemistry, delays,
eligibility and structural development have their own state, cadence and
documented cost — see [architecture](docs/architecture.md) and
[efficiency](docs/efficiency.md).

## Quick start

```powershell
python -m pip install -e .
python -m unittest discover -s tests -q     # 107 tests, 1 skipped without CUDA
```

A first tick (CPU is fine):

```python
import torch
from neurosim_gpu import Dynamics, MatrixRuntime, random_topology

topology = random_topology(1000, 20_000, device="cpu", seed=0)
dynamics = Dynamics(
    alpha=torch.full((1000,), 0.72),
    gain=torch.full((1000,), 1.05),
)
runtime = MatrixRuntime(topology, dynamics, torch.zeros(1, 1000))
for _ in range(10):
    state = runtime.step(drive=0.1)
print(topology.n_edges, float(state.mean()))
```

More: [getting started](docs/getting-started.md) ·
[runnable API workflows](docs/api.md).

## Runtimes

| Runtime | Path | Notes |
|---|---|---|
| `MatrixRuntime`, `DopamineMatrixRuntime` | portable PyTorch CSR | readable reference, portable |
| `HybridPELSRuntime`, `DopaminePersistentRuntime` | persistent synaptic current | emitted-delta propagation; delta scatter is still a full SpMM in the reference |
| `NativeCanonicalMatrixRuntime`, `NativeDopamineMatrixRuntime` | fused CUDA | destination-major spawn reduction + one weight array |
| `NativeEventPELSRuntime`, `NativeDopamineEventRuntime` | fused CUDA event path | `O(BN + M_A)` propagation at low activity |

The native paths need a CUDA build of PyTorch and compile on first use:
[native build](docs/native-build.md). With eligibility enabled, the dopamine
runtime still scans all `M` edges per tick — whole-tick event sparsity currently
requires eligibility disabled and no weight update on that tick.

## What has been measured

Accepted numbers with hardware, graph size and command are in
[results](docs/results.md); their interpretation is in
[efficiency](docs/efficiency.md). Highlights:

- fixed-core forward at batch 1: **0.74–0.82×** the cost of a same-parameter
  sparse kernel at 10–25M edges;
- low-activity event propagation: **3–4.5×** faster than the matrix path near
  2.6% active edges;
- emitted-event threshold on MNIST: **21% of edges active, −0.12 points**
  accuracy (forward pass only).

Task-level probes (MNIST digits, inverted pendulum) live in
[experiments](docs/experiments.md); the full report is
[results/REPORT.md](results/REPORT.md):

| Probe | Result |
|---|---|
| MNIST, sparse brain vs parameter-matched MLP | 91.1% vs 87.3% (4.6k trainable scalars) |
| MNIST, sparse brain vs logistic regression | 91.1% vs 91.8% — no accuracy win |
| Inverted pendulum, behaviour cloning | 200/200 steps for both MLP and sparse brain; failures only under extreme delay/noise |
| Online three-factor plasticity after a plant change | no measurable benefit; ablation shows the learned recurrence is nearly dispensable |

## Scope

This is a **research prototype**, not a validated brain model. It does not claim
biological realism, whole-brain scale or self-play strength, and its performance
numbers are bound to the hardware and graph size recorded with them. Read
[scope](docs/scope.md) before citing it.

## Documentation

Start at [docs/index.md](docs/index.md). The short version:

| Document | Contents |
|---|---|
| [getting-started](docs/getting-started.md) | install, first tick, troubleshooting |
| [api](docs/api.md) | public workflows, tensor contracts, errors |
| [design](docs/design.md) · [architecture](docs/architecture.md) | numerical contract, state ownership, runtime modes |
| [biology](docs/biology.md) | what the neurochemical model means and where the analogy stops |
| [efficiency](docs/efficiency.md) · [results](docs/results.md) · [validation](docs/validation.md) | cost model, accepted measurements, acceptance criteria |
| [experiments](docs/experiments.md) · [gomoku-protocol](docs/gomoku-protocol.md) | task-level probes and their protocols |
| [status](docs/status.md) · [verification](docs/verification.md) | milestone scorecard, release verification log |

中文简介见 [README.zh-CN.md](README.zh-CN.md)。

## Repository layout

```text
src/neurosim_gpu/   package: runtime, topology, chemistry, dopamine, scheduler, ...
csrc/               CUDA/C++ extension sources
benchmarks/         GPU, cost model, latency, training parity, Gomoku probes
experiments/        MNIST and inverted-pendulum experiment suite
tests/              regression suite
docs/               contracts, measurements and guides
scripts/            native build helper and documentation link check
results/            experiment report, aggregate tables, raw payloads
```

## Contributing and license

Contributions are welcome; start with [CONTRIBUTING.md](CONTRIBUTING.md) and the
acceptance criteria in [validation](docs/validation.md). The suite must stay
green and documentation links must resolve (`python scripts/check_links.py`).

Licensed under the Apache License 2.0 — see [LICENSE](LICENSE). Citation metadata
is in [CITATION.cff](CITATION.cff).
