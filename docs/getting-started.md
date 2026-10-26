# Getting started

## Requirements

| Component | Requirement |
|---|---|
| Python | 3.10 or newer |
| PyTorch | 2.7 or newer (`pip install torch`; a CUDA build is only needed for the native kernels) |
| C++ toolchain | optional — only for the native CUDA/C++ extension, see [native-build](native-build.md) |

The core package has no other runtime dependencies.

## Install

From a checkout:

```powershell
python -m pip install -e .
```

`uv.lock` is provided for `uv` users (`uv sync`), but plain `pip` is enough.

## Verify the install

```powershell
python -m unittest discover -s tests -v
```

Expect `107 tests` with one skip when no CUDA device is visible (the skip is a
CUDA topology-transfer check). The suite takes a few seconds on CPU.

Two warnings are expected and harmless on a CPU-only install:

- `Failed to initialize NumPy` — PyTorch probes for NumPy; the project does not
  require it.
- `Sparse CSR tensor support is in beta state` — emitted whenever a CSR tensor
  is constructed.

## First tick

A fixed random graph of 1,000 neurons and 20,000 candidate synapses (duplicate
edges are coalesced, so the final count is slightly lower):

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

The state stays in `[0, 1]`; after ten ticks with a constant drive it settles
near `drive`.

To see the persistent-current/event variant, replace the runtime:

```python
from neurosim_gpu import HybridPELSRuntime

runtime = HybridPELSRuntime(
    topology, dynamics, torch.zeros(1, 1000),
    theta=0.05,          # emit only state deltas at or above this threshold
    propagation="matrix", # "matrix", "event" or "auto"
)
state, stats = runtime.step(drive=0.1, collect_stats=True)
print(stats.path, stats.active_edges, stats.active_neurons)
```

`collect_stats=True` reads scalars back to the host; leave it off in latency
measurements.

## Task-level demos

Both run on CPU in seconds and need no extra packages:

```powershell
python experiments/mnist_experiments.py --smoke
python experiments/cartpole_experiments.py --smoke
```

Full grids and the recorded results are described in [experiments](experiments.md).

## Benchmarks

A device-agnostic grid that runs on CPU or CUDA, and reports the tick cost
against a raw sparse multiplication:

```powershell
python benchmarks/performance_suite.py --device cpu --json results/performance_cpu.json
```

The CUDA-only harness and the accepted numbers live in
[results](results.md); the commands are in [efficiency](efficiency.md) and the
interpretation in [performance](performance.md). A short version:

```powershell
python benchmarks/benchmark.py --neurons 100000 --edges 5000000 --steps 100
```

The native fused kernels are optional and require a CUDA build of PyTorch:
see [native-build](native-build.md).

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `RuntimeError: native inference kernels do not implement autograd` | Native runtimes are inference-only; use `MatrixRuntime`/`DopamineMatrixRuntime` for training, or the pointwise `native_advance_training` path |
| `ValueError: native runtime requires CUDA float32 tensors` | The native runtimes only accept CUDA FP32 tensors |
| `torch.cuda.is_available()` is `False` | CPU-only PyTorch build; every CPU path still works, the native kernels do not |
| Edits to `src/` are not picked up | The package is installed editable; re-run `python -m pip install -e .` if you moved the checkout |
| Native extension rebuilds on every run | `torch.utils.cpp_extension.load` caches under the torch extensions directory; a changing source tree invalidates it |

---

[← Documentation index](index.md)
