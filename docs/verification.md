# Release verification log

What was executed, on which stack, and what passed — for version 0.1.0.

A verification entry is only meaningful with its environment. The CPU rows below
were executed on a Windows 11 workstation with Python 3.12 and
`torch 2.14.0+cpu`; the GPU row is inherited from the measurement session
recorded in [results](results.md) and cannot be reproduced without a CUDA device.

## v0.1.0 — 2026-10-26

| Check | Command | Result |
|---|---|---|
| Unit and integration tests | `python -m unittest discover -s tests -q` | **107 passed, 1 skipped** (`CUDA is not available`) |
| Documentation links | `python scripts/check_links.py` | all local links resolve |
| Task-level experiment grid | `python experiments/run_suite.py --suite all` | **105 runs** completed; payloads in `results/raw/` |
| Experiment determinism | re-run `--suite mnist_models` and compare | identical accuracy per `(config, seed)` |
| Cart-pole physics self-test | `python experiments/cartpole_experiments.py --self-test` | **7/7** checks passed (energy drift 1.62e-6 over 200 RK4 steps, LQR 200.0/200) |
| Cart-pole smoke | `python experiments/cartpole_experiments.py --smoke` | `all_passed` |
| Native CUDA validation | `python benchmarks/validate_native.py` | **PASS** on the reference workstation (RTX 5070 Laptop, CUDA 13.1); not runnable on a CPU-only install |

### What the experiment grid covered

Twelve suites, 105 runs: dense MLP / logistic regression / sparse brain / dense
RNN on MNIST (sparsity, neurons, ticks, data size, architecture variants,
corruption), and LQR / cloned policies on the inverted pendulum (nominal, hard,
robustness grid, online plasticity). The analysis, negative results and
limitations are in [experiments](experiments.md) and
[../results/REPORT.md](../results/REPORT.md).

### Reproduction notes

- Tests and experiments are CPU-only and need no extra packages.
- The MNIST suite downloads the official idx files on first use and caches them
  under `data/mnist/`; the directory is ignored by git.
- Experiment latency numbers depend on machine load; accuracy and survival
  metrics are seed-deterministic.
- The native extension builds lazily on first use; see
  [native-build](native-build.md).

## How to add an entry

Record the date, the exact command, the environment (device, torch build,
driver/toolkit when a GPU is involved) and the observed result. Link the
artifacts rather than pasting large outputs. If a check was skipped, say why.

---

[← Documentation index](index.md)
