# Getting started

## Install the current mainline

Python ≥3.10, PyTorch ≥2.7; no other core runtime dependencies. CPU needs no compiler.
CUDA native inference needs CUDA PyTorch plus a compatible C++/CUDA build toolchain.
See [native-build](native-build.md). The current mainline runner is newer than the
0.2.0 wheel, so install from a source checkout:

```powershell
python -m pip install -e .
python -m neurosim_gpu.simulate --steps 100 --output runs/mainline/cpu.json
```

This runs the recommended soma–dendrite–adaptation model, not the older point-neuron
baseline. The default is CPU FP32 matrix reference. It prints all three state
statistics and the actual graph size; the JSON is not a performance benchmark.

## Choose an execution path

- CPU / training / correctness: `DendriticMatrixRuntime`.
- CUDA FP32 inference starting point: `NativeDendriticMatrixRuntime`.
- Low activity, approximation allowed: benchmark `NativeDendriticPELSRuntime`
  against the matrix path on your workload; do not assume it is always faster.
- `theta=0` checks PELS parity. Nonzero theta is not a full trajectory-error bound.

```powershell
scripts\native_env.bat python -m neurosim_gpu.simulate --device cuda --backend native --steps 1000
scripts\native_env.bat python -m neurosim_gpu.simulate --device cuda --backend native --propagation pels --theta 0.02 --steps 1000
```

On Linux use Python directly with the configured compiler environment. Backend and
propagation are explicit. Native build failures are surfaced, never silently hidden.
For Python usage, copy the current [README example](../README.en.md); training uses
the API, while the CLI always uses inference mode. Full options and checkpoint
instructions: [mainline guide](mainline.md).

## Verify

```powershell
python -m unittest discover -s tests -v
python scripts/check_links.py
```

The 0.2.0 release discovered 204 tests. Current mainline adds runner tests. Native
checks require `NEUROSIM_TEST_NATIVE=1` and a configured toolchain; hardware and
optional HF dependency skips depend on the environment. Report executed/skipped
counts, rather than treating skips as passes. Sparse CSR beta warnings are expected.

```powershell
$env:NEUROSIM_TEST_NATIVE="1"
scripts\native_env.bat python -m unittest discover -s tests -p test_native_compartmental.py -v
scripts\native_env.bat python benchmarks/compartmental.py --large --profile --output runs/mainline/performance.json
```

[Measured performance](performance-0.2.0.md) · [Verification record](verification.md).
Older `benchmarks/performance_suite.py` and `benchmarks/benchmark.py` measure the
point-neuron baseline, not the current three-state core.

## Extensions, not automatic mainline features

Chemistry/plasticity, delay/development and Cortex-LM have separate entry points.
They have not all been integrated into the dendritic core runner. See
[architecture](architecture.md), [API](api.md) and [experiments](experiments.md).
Existing point-neuron examples and task probes remain useful compatibility baselines.

## Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `No module named neurosim_gpu.simulate` | Install the current checkout, not only the older 0.2.0 wheel; check which Python environment is active |
| Native build fails | Configure CUDA/C++ using [native-build](native-build.md), or explicitly choose `--backend reference` |
| `--backend native requires --device cuda` | Native compartment kernels are CUDA FP32 only |
| CUDA unavailable | CPU-only PyTorch or device/driver issue; use CPU reference |
| Native API reports `reference-autograd` | Expected gradient-preserving fallback; no native training speedup is claimed |
| Resume rejects parameters/type | Reuse the original graph/model/batch/runtime options; do not cross-load matrix/PELS or reference/native checkpoint classes |
| Edits are not visible | Check interpreter and import path; reinstall editable from this checkout |

[← Documentation index](index.md)
