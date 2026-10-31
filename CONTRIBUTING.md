# Contributing

Thanks for taking the time to look at this. The project is a research prototype,
so the bar for a change is not "does it add features" but "does it keep the
documented contract and the measurements honest".

## Before you start

- Read [docs/scope.md](docs/scope.md) for what the project does and does not claim.
- Read [docs/architecture.md](docs/architecture.md) for state ownership and the
  multi-rate schedule; most design mistakes are ownership mistakes.
- For a new measurement or performance claim, read
  [docs/validation.md](docs/validation.md) first — the acceptance rules matter
  more than the number.

## Development setup

```powershell
python -m pip install -e .
python -m unittest discover -s tests -q
python scripts/check_links.py
```

The test suite is CPU-only and takes a few seconds. The native CUDA extension is
optional and built lazily; see [docs/native-build.md](docs/native-build.md).

## Invariants that must not regress

1. **No implicit all-pairs neuron operation.** A feature that would introduce
   one must be reformulated over explicit candidates.
2. **One owner per state.** Topology, activity, chemistry, eligibility, delay
   history and positions each have a single owning runtime.
3. **The fast path stays asynchronous.** No per-tick device-to-host
   synchronization unless statistics are explicitly requested.
4. **Eligibility is not free.** Enabling it costs a full `M`-edge scan per tick;
   do not describe event propagation as end-to-end sparse while it is on.
5. **Numbers name their stack.** Any result added to the docs must include
   hardware, graph size, batch, dtype and the command that produced it.

## Code style

- Python 3.10+, type hints on public functions, `from __future__ import annotations`.
- Four-space indentation, ~88 column soft limit, no formatter is enforced yet.
- Docstrings explain the contract and the cost, not the syntax.
- CUDA/C++ follows the existing file-local conventions; keep kernels launch-bound
  and avoid host synchronisation inside stepped paths.
- Do not add runtime dependencies. The package depends on PyTorch only; tests and
  experiments use the standard library.

## Tests

- New behaviour needs a regression test in `tests/`.
- Error paths are part of the contract: assert the exception type and a fragment
  of the message (`assertRaisesRegex`).
- If a test needs CUDA, guard it with `@unittest.skipUnless(torch.cuda.is_available(), ...)`
  so the CPU suite stays green.

## Documentation

- Every document owns a specific claim; see the table in
  [docs/index.md](docs/index.md). Put a number in `docs/results.md`, its
  interpretation in `docs/efficiency.md`, and the acceptance rule in
  `docs/validation.md`.
- Local links are checked by `python scripts/check_links.py`; CI fails on a
  broken link.
- Prefer tables and explicit units over prose for measurements.

## Submitting a change

1. Keep the change focused; separate refactors from behaviour changes.
2. Run the test suite and the link check.
3. If you changed a contract, update the owning document in the same commit.
4. If you changed a measurement, add a row to `docs/verification.md`.
5. Open a pull request describing: what changed, why, what you ran, and what you
   could not verify (for example "no CUDA device available").

## Reporting a problem

Use the issue templates. A useful performance report includes the exact command,
the graph size, batch, dtype, device, PyTorch/CUDA versions, and whether the
device was otherwise idle.
