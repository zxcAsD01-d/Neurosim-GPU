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
| Understand the runtime and state ownership | [design](design.md) → [architecture](architecture.md) |
| Build the CUDA kernels | [native-build](native-build.md) |
| Run the task-level experiments | [experiments](experiments.md) |
| Check a performance number | [results](results.md) → [efficiency](efficiency.md) → [validation](validation.md) |
| Know what is done and what is open | [status](status.md) |
| Contribute | [../CONTRIBUTING.md](../CONTRIBUTING.md) → [validation](validation.md) |

## Document map

| Document | Audience | Authority (what it owns) |
|---|---|---|
| [scope](scope.md) | everyone | Project scope, non-goals, claim boundaries |
| [getting-started](getting-started.md) | new users | Install, first run, troubleshooting |
| [design](design.md) | core developers | Fixed-topology numerical and runtime contract |
| [architecture](architecture.md) | core developers | State ownership, runtime modes, multi-rate schedule, integration status |
| [biology](biology.md) | researchers | Functional biological semantics and where the analogy stops |
| [api](api.md) | users | Runnable public workflows, tensor contracts, errors |
| [native-build](native-build.md) | users with a GPU | Building and validating the CUDA/C++ extension |
| [experiments](experiments.md) | researchers | Task-level probes (MNIST, inverted pendulum), how to run and read them |
| [gomoku-protocol](gomoku-protocol.md) | researchers | CNN vs sparse-brain task protocol and interpretation limits |
| [efficiency](efficiency.md) | performance work | Cost formulas, parity definitions, optimization analysis |
| [results](results.md) | everyone quoting numbers | Source of record for accepted measurements and their hardware |
| [validation](validation.md) | contributors, CI | Required tests, tolerances, benchmark protocol |
| [status](status.md) | users, contributors | Milestone scorecard and open work |
| [verification](verification.md) | maintainers | Release verification log (what was run, on which stack, what passed) |

## Conventions

- **Numbers have one home.** `results.md` is the source of record for accepted
  measurements; `efficiency.md` interprets them; `validation.md` defines what
  counts as acceptance. If prose in another document disagrees, those three win.
- **Every measurement names its stack.** Hardware, CUDA/PyTorch versions, graph
  size, batch, dtype and command are part of the result, not an afterthought.
- **Claims are bounded.** `scope.md` and `biology.md` state which statements the
  project does *not* support (biological validation, whole-brain simulation,
  self-play strength, and so on).
- **Documents travel with the code.** If a change alters a contract or a
  measurement, update the owning document in the same change.
- **Links are checked.** `python scripts/check_links.py` verifies that local
  Markdown links resolve; CI runs it on every push.
