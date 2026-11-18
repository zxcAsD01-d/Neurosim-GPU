# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Changed

- Promote the verified soma/dendrite/adaptation core to the recommended project
  mainline in both READMEs, onboarding and documentation navigation. Keep older
  point-neuron APIs and measurements as explicitly labeled baselines.
- Separate integrated core capabilities from chemistry/development extensions
  and Cortex-LM; remove stale test counts and completed wheel-packaging roadmap work.

### Added

- `python -m neurosim_gpu.simulate`: an explicit reference/native, matrix/PELS
  inference runner with complete runtime checkpoints and JSON diagnostics.
- Runner tests for three-state execution, zero-threshold parity, matrix/PELS
  checkpoint continuation, output records and invalid-option rejection.
- Six-panel native CUDA activity comparison for continued versus withdrawn
  input, with a reproduction script and measured traces.

## [0.2.0] - 2026-11-17

### Added

- Optional dendritic compartment and slow activity-adaptation dynamics, separate
  somatic/dendritic drives, heterogeneous time constants, matrix/PELS runtimes,
  complete state checkpointing and a dated 2025–2026 research review.
- `NativeDendriticMatrixRuntime`: one fused CUDA launch for CSR propagation and
  all three neuron states; 8/16/32-lane row reductions, strided/broadcast drives.
- `NativeDendriticPELSRuntime`: two CUDA launches for local state advance and
  event scatter, with no per-tick host count. Both native paths use the
  differentiable reference whenever gradients are requested.
- Reproducible same-model FP32 performance measurements through approximately
  25M edges, profiler kernel counts and transient allocation measurements.
- Regression coverage for long rollouts, input layouts, state retention,
  parameter/drive/synaptic gradients, non-default streams and checkpoint resume.

### Changed

- Wheels now include CUDA sources for optional local JIT compilation.
- Package and citation version are 0.2.0. Existing point-neuron equations and
  their published performance measurements retain their original meaning.

### Limits

- Accelerated compartment kernels are CUDA FP32 inference paths. Training
  falls back to PyTorch; this release does not claim training acceleration.
- Nonzero PELS thresholds can produce different emission schedules under FP32
  rounding. Reported empirical errors are not formal trajectory-error bounds.
- The neuron model remains an uncalibrated rate surrogate, not a biological fit.

## Changes already present before 0.2.0

### Added

- Cortex-LM pipeline: Qwen3-0.6B keeps its first 3 and last 2 blocks, the middle
  23 blocks are replaced by a 1024-neuron dense recurrent core initialised from
  the removed blocks' linearised residual maps. Includes JSON configs, streaming
  JSONL corpus loading, checkpoints, dry-run/training/evaluation CLIs and
  regression tests. A short training run on a MiniMind subset is included as an
  artifact, not as a language-capability claim.

### Changed

- Minimum Python is now **3.10**; CI covers 3.10 and 3.12.
- Updated `filelock` to 4.2.4 and `setuptools` to 84.2.1 (security advisories), and
  updated the CI actions to `actions/checkout@v4` and `actions/setup-python@v5`.

## [0.1.0] - 2026-10-26

Initial public prototype.

### Added

- Fixed-topology sparse core: canonical source-major edges, destination CSR
  view, weight versioning and duplicate-edge coalescing.
- Runtimes: `MatrixRuntime`, `HybridPELSRuntime` (persistent current),
  `DenseOracle`, plus native fused matrix and event runtimes.
- CUDA/C++ extension: fused advance, canonical SpMM and step, PELS event step,
  eligibility and dopamine-gated weight updates, pointwise training adjoint.
- Regional dopamine model: tonic/burst/dip chemistry, volume-weighted diffusion,
  uptake, transport delay, D1/D2 occupancy, TD-RPE and multirate boundaries.
- Three-factor plasticity: neuron traces, per-edge eligibility, sign-preserving
  dopamine-gated updates (exact for batch 1, `[B, M]` offline variant).
- Delayed multi-channel communication, bounded 3-D migration and
  candidate-only structural formation/pruning primitives.
- Multi-rate scheduler with a topology-boundary transaction, runtime and
  communication checkpoints, and a deterministic replay/continual-learning
  controller.
- Gomoku environment and a CNN versus sparse-brain supervised probe.
- Task-level experiment suite: MNIST digit recognition and inverted-pendulum
  stabilisation (105 grid runs), with a written report and aggregate tables.
- Documentation system under `docs/` (scope, contracts, measurements,
  validation, status, release verification) and a link checker in CI.

### Known limitations

- The 125M-edge prebuilt topology point has not been executed end to end.
- Automatic matrix/event switching has no device-side selector; the native paths
  are fixed choices.
- The native training adjoint is pointwise (analytic local backward), not a
  whole-rollout custom adjoint.
- Eligibility scanning keeps the event path from being end-to-end sparse.
- Structural change is a tested primitive, not a live developmental runtime.
- No biological validation; parameters are not calibrated to recordings.
