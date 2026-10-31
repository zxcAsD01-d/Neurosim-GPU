# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
