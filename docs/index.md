# Documentation

This directory holds the design, numerical contract and measurement record for
NeuroSim GPU. The code is the implementation; these documents state what it
promises, what has actually been measured, and what is explicitly out of scope.

## Reading paths

| I want to… | Read |
|---|---|
| Understand what this project is and is not | [scope](scope.md) → [biology](biology.md) |
| Install it and run a first tick | [getting-started](getting-started.md) |
| Use the public API | [api](api.md) |
| Run the recommended complete core | [mainline guide](mainline.md) → [research and model](neuron-research-2026.md) |
| Check the 0.2.0 CUDA optimization | [performance and validation](performance-0.2.0.md) |
| Understand the runtime and state ownership | [design](design.md) → [architecture](architecture.md) |
| Build the CUDA kernels | [native-build](native-build.md) |
| Run the task-level experiments | [experiments](experiments.md) |
| Run the optional NeuroBody servo experiment | [NeuroBody](../demo/README.md) → [failure review](../demo/docs/neurobody-dendritic-migration-20261120.md) |
| Build the brain-in-the-middle LM pipeline | [cortex-lm](../llm/docs/cortex-lm.md) |
| Check a performance number | [performance](performance.md) → [results](results.md) → [efficiency](efficiency.md) → [validation](validation.md) |
| Know what is claimed, and what would refute it | [claims](claims.md) |
| Know what is done and what is open | [status](status.md) |
| Contribute | [../CONTRIBUTING.md](../CONTRIBUTING.md) → [validation](validation.md) |

## Document map

| Document | Audience | Authority (what it owns) |
|---|---|---|
| [scope](scope.md) | everyone | Project scope, non-goals, claim boundaries |
| [getting-started](getting-started.md) | new users | Install, first run, troubleshooting |
| [mainline](mainline.md) | users, maintainers | Recommended three-state core, execution paths, checkpoint workflow and integration boundaries |
| [performance-0.2.0](performance-0.2.0.md) | performance work | Three-state model measurements, source hashes and numerical limits |
| [design](design.md) | core developers | Fixed-topology numerical and runtime contract |
| [architecture](architecture.md) | core developers | State ownership, runtime modes, multi-rate schedule, integration status |
| [biology](biology.md) | researchers | Functional biological semantics and where the analogy stops |
| [neuron-research-2026](neuron-research-2026.md) | researchers, users | Recent sources, compartment equations, API and validation boundaries |
| [api](api.md) | users | Runnable public workflows, tensor contracts, errors |
| [native-build](native-build.md) | users with a GPU | Building and validating the CUDA/C++ extension |
| [experiments](experiments.md) | researchers | Task-level probes (MNIST, inverted pendulum), how to run and read them |
| [cortex-lm](../llm/docs/cortex-lm.md) | researchers | Qwen3 3+2 split, neuron transplant, data and training |
| [gomoku-protocol](gomoku-protocol.md) | researchers | CNN vs sparse-brain task protocol and interpretation limits |
| [efficiency](efficiency.md) | performance work | Cost formulas, parity definitions, optimization analysis |
| [performance](performance.md) | everyone quoting numbers | Cost model, GPU parity and event numbers, CPU reference baseline, how to measure |
| [results](results.md) | everyone quoting numbers | Source of record for accepted measurements and their hardware |
| [validation](validation.md) | contributors, CI | Required tests, tolerances, benchmark protocol |
| [claims](claims.md) | reviewers, users | Claim ledger: status of each claim, evidence, falsifier |
| [status](status.md) | users, contributors | Milestone scorecard and open work |
| [verification](verification.md) | maintainers | Release verification log (what was run, on which stack, what passed) |

## Conventions

- **Numbers have one home per model.** `performance-0.2.0.md` owns the three-state
  mainline measurements; `results.md` owns historical point-neuron measurements.
  Do not transfer performance claims between models. `validation.md` defines
  acceptance and `verification.md` records actual checks.
- **Every measurement names its stack.** Hardware, CUDA/PyTorch versions, graph
  size, batch, dtype and command are part of the result, not an afterthought.
- **Claims are bounded.** `scope.md` and `biology.md` state which statements the
  project does *not* support (biological validation, whole-brain simulation,
  self-play strength, and so on).
- **Documents travel with the code.** If a change alters a contract or a
  measurement, update the owning document in the same change.
- **Links are checked.** `python scripts/check_links.py` verifies that local
  Markdown links resolve; CI runs it on every push.
