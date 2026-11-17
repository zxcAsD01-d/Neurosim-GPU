# Building the native CUDA kernels

The Python package works without any compiled code. The optional extension
fuses the hot paths into a few kernels and is required only for the native
runtimes and the `--native` benchmark mode.

## What gets built

One PyTorch extension, `neurosim_gpu_native`, from three sources:

| Source | Contents |
|---|---|
| `csrc/bindings.cpp` | pybind11 module, argument checking, stream plumbing |
| `csrc/runtime.cu` | CUDA kernels and their launch configuration |
| `csrc/compartmental.cu` | fused dendritic matrix tick and event tick |

Exported entry points:

| Function | Role |
|---|---|
| `advance` | fused `clamp(alpha*s + (1-alpha)*gain*tanh(current+drive))` |
| `advance_train_forward` / `advance_train_backward` | the same update plus local derivatives for a custom autograd node |
| `canonical_spmm` | destination-major sparse matrix multiplication over one weight array |
| `canonical_step` | SpMM fused with the neural update |
| `canonical_modulated_step` | the same, with receptor (D1/D2) drive/gain modulation |
| `pels_event_step_` | persistent-current advance + event compaction + source-major scatter |
| `pels_dopamine_event_step_` | event step with receptor modulation |
| `update_eligibility_` | neuron traces + per-edge eligibility |
| `dopamine_weight_update_` | dopamine-gated canonical weight update |
| `dendritic_step` | sparse reduction and three compartment states in one launch |
| `dendritic_event` | compartment update/emission plus device event scatter |

The module is compiled lazily by `neurosim_gpu.native.load_native_extension()`
through `torch.utils.cpp_extension.load`, so the first call pays the build cost
and later calls reuse the cached binary.

Sources live in the repository's `csrc/` directory. Starting with 0.2.0, wheels
also include them in `neurosim_gpu/csrc/`. The loader accepts either location;
a local compiler and CUDA toolkit are still required (no precompiled GPU binary
is shipped). Missing sources produce an explicit error before invoking nvcc.

## Prerequisites

- A CUDA build of PyTorch (`torch.version.cuda` must not be `None`).
- A CUDA toolkit matching, or compatible with, the toolkit PyTorch was built
  against. `nvcc` must be reachable (`CUDA_HOME` or on `PATH`).
- A host C++ compiler:
  - **Windows**: Visual Studio 2019+ with the "Desktop development with C++"
    workload. 64-bit `cl.exe` must come from `vcvars64.bat`.
  - **Linux**: `gcc`/`g++` compatible with the toolkit.
- `ninja` is strongly recommended; `torch.utils.cpp_extension` uses it when
  available and falls back to slower build tooling otherwise.

## Build and validate

```powershell
python benchmarks/validate_native.py
```

For compartment kernels, set `NEUROSIM_TEST_NATIVE=1` and run
`python -m unittest tests.test_native_compartmental -v` in the same build
environment. The explicit opt-in keeps CPU installations and GPU hosts without
a compiler usable. On Windows, prefix the command with `scripts\native_env.bat`.

The script refuses to run without CUDA, builds the extension, then compares the
native matrix, canonical, event and dopamine paths against the Python reference
implementations with the tolerances recorded in [validation](validation.md). A
successful run prints `native CUDA validation: PASS`.

Once built, the native paths are used by:

```powershell
python benchmarks/benchmark.py --neurons 100000 --edges 5000000 --native
python benchmarks/dopamine_benchmark.py --native
```

## Windows: environment helper

MSVC is not on `PATH` in a normal shell, so the repository ships a wrapper that
locates Visual Studio through `vswhere`, picks the newest installed CUDA toolkit
when `CUDA_HOME` is unset, and then runs the command you pass:

```powershell
scripts\native_env.bat python benchmarks\validate_native.py
scripts\native_env.bat python benchmarks\benchmark.py --native
```

It contains no machine-specific paths; on Linux no wrapper is needed.

## Environment variables

| Variable | Effect |
|---|---|
| `CUDA_HOME` | Toolkit location; also used to locate `nvcc` |
| `TORCH_CUDA_ARCH_LIST` | Target architectures, for example `"8.9"` for Ada or `"12.0"` for Blackwell. Set it to skip architecture autodetection |
| `NEUROSIM_HOST_FLAGS` | Overrides the host compiler flags (default `-O3` on POSIX, `/O2` on Windows) |
| `NEUROSIM_CUDA_FLAGS` | Overrides the CUDA flags (default `-O3 --use_fast_math`) |
| `TORCH_EXTENSIONS_DIR` | Where torch caches compiled extensions; delete the `neurosim_gpu_native` subdirectory to force a rebuild |

## Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA is required` from `validate_native.py` | You are on a CPU-only PyTorch build; native kernels are unavailable |
| `nvcc: command not found` / `Cannot find nvcc` | Install a CUDA toolkit, or set `CUDA_HOME` |
| `torch.utils.cpp_extension` cannot find `cl.exe` | Run through `scripts\native_env.bat`, or launch from a "x64 Native Tools" prompt |
| Build fails with an unsupported architecture | Set `TORCH_CUDA_ARCH_LIST` to a compute capability your device and toolkit support |
| A stale build is reused after editing `csrc/` | torch keys the cache on source timestamps; if needed, delete `$env:TORCH_EXTENSIONS_DIR\neurosim_gpu_native` |
| Build takes minutes | Expected on first use; the `--use_fast_math` and `-O3` flags trade compile time for runtime |

---

[← Documentation index](index.md)
