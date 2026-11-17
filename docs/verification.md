# Release verification log

What was executed, on which stack, and what passed — for versions 0.1.0 and 0.2.0.

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
| CPU performance grid | `python benchmarks/performance_suite.py --device cpu --json results/performance_cpu.json` | 12 configurations × 6 runtime rows; payload committed, tables in [performance](performance.md) |
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
- The performance grid (`benchmarks/performance_suite.py`) runs on CPU or CUDA;
  its CPU payload is committed as `results/performance_cpu.json`. Latency numbers
  move with machine load and thread count, throughput ratios much less so.
- The MNIST suite downloads the official idx files on first use and caches them
  under `data/mnist/`; the directory is ignored by git.
- Experiment latency numbers depend on machine load; accuracy and survival
  metrics are seed-deterministic.
- The native extension builds lazily on first use; see
  [native-build](native-build.md).

## Optional compartment model — 2026-11-17

Windows workstation, Python 3.13.0, `torch 2.13.0.dev20260514+cu132`,
CUDA 13.2 as reported by PyTorch, NVIDIA GeForce RTX 5070
Laptop GPU. Commands used `.venv-str/Scripts/python.exe` with `PYTHONPATH=src`.
No native extension was rebuilt and no latency claim was measured.

| Check | Command | Result |
|---|---|---|
| Before-change baseline | `python -m unittest tests.test_runtime tests.test_checkpoint -q` | 17 passed |
| Compartment + existing runtime/checkpoint tests | `python -m unittest tests.test_compartmental tests.test_runtime tests.test_checkpoint -v` | 30 passed, including CPU and CUDA compartment parity |
| Full repository regression | `python -m unittest discover -s tests -q` | 208 passed |
| Documentation links | `python scripts/check_links.py` | All local links resolve across 95 Markdown files |
| Research document example | Executed its Python code block | Passed on CUDA, output shape `[2,128]` |
| Isolated-neuron response, CPU | `python experiments/compartmental_probe.py --device cpu --output results/compartmental_probe_cpu.json --plot results/compartmental_probe.png` | Completed; disabled model error exactly 0 |
| Same response, CUDA | `python experiments/compartmental_probe.py --device cuda --output results/compartmental_probe_cuda.json` | Completed; disabled model error exactly 0 |

Probe geometry is `N=32, M=0, B=2, FP64, 400 ticks`. The combined model's
sustained-response peak was 0.3731366856 and its end-of-step response was
0.2611566121 on both devices (rounding to 10 digits). These are synthetic
intrinsic-response observations, not accuracy improvements or biological fits.
The largest absolute difference across all CPU/CUDA probe state traces was
`2.220446049250313e-16`.
The test's connected sparse networks separately cover `N=12, B=2` with
48 candidate edges (coalesced by the topology builder), FP64.

Raw data: [CPU](../results/compartmental_probe_cpu.json),
[CUDA](../results/compartmental_probe_cuda.json),
[response figure](../results/compartmental_probe.png).
Equations and evidence boundaries: [compartment model](neuron-research-2026.md).

## 0.2.0 release validation — 2026-11-17

Release worktree starts at GitHub `main` (`f0c6c21`), with only the compartment
model and its optimization added. Unpublished local Cortex-LM/hippocampus and
affect experiments are not included. This explains the different test counts
from the earlier development-workspace entry above.

Stack: Windows, Python 3.13.0, PyTorch `2.13.0.dev20260514+cu132`, CUDA toolkit
13.1/runtime 13.2, MSVC 14.44, driver 616.92, RTX 5070 Laptop GPU 8 GiB.
Independent CPU-only checks used Python 3.12 and PyTorch `2.14.0+cpu`.

| Check | Command | Observed result |
|---|---|---|
| CUDA and native full suite | `NEUROSIM_TEST_NATIVE=1`, `scripts\native_env.bat python -m unittest discover -s tests -q` | 204 passed |
| CPU with optional HF dependencies | `CUDA_VISIBLE_DEVICES=-1`, `python -m unittest discover -s tests -q` | 204 discovered, 12 skipped, rest passed |
| CPU-only minimal install | CPU Python, same discovery command | 204 discovered, 23 skipped, rest passed |
| Previous native kernels | `scripts\native_env.bat python benchmarks/validate_native.py` | PASS |
| New kernel memory access | `compute-sanitizer --tool memcheck --error-exitcode 99 --target-processes application-only --launch-timeout 60 python -m unittest tests.test_native_compartmental -q` | 10 passed; ERROR SUMMARY: 0 errors |
| New kernel synchronization | Same command with `--tool synccheck` | 10 passed; ERROR SUMMARY: 0 errors |
| Wheel/sdist build | `uv build --out-dir dist` | Both 0.2.0 artifacts built |
| Distribution contents | `python scripts/check_distribution.py` | Version/module/source checks passed |
| Installed wheel CUDA smoke | Install wheel in isolated venv; build/load bundled sources and compare 30 matrix/event ticks with reference | PASS |
| Local Markdown links | `python scripts/check_links.py` | All 44 Markdown files resolve |

On this Windows host the virtualenv Python executable is a launcher. The first
sanitizer attempt using it with child-process tracking completed tests but hung
on exit, and an application-only retry could not attach. The successful check
used the actual Python interpreter with application-only tracking. Neither
failed attempt is counted as sanitizer success.

Same-model timings, kernel counts, allocation reduction and numerical limits
are recorded in [0.2.0 performance](performance-0.2.0.md), with raw samples in
[JSON](../results/compartmental_performance.json).

## How to add an entry

Record the date, the exact command, the environment (device, torch build,
driver/toolkit when a GPU is involved) and the observed result. Link the
artifacts rather than pasting large outputs. If a check was skipped, say why.

---

[← Documentation index](index.md)
