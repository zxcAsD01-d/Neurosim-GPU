# Scope and claims

NeuroSim GPU is a research prototype of a GPU-first sparse neural simulator. It
treats a fixed-topology neural core as a sparse linear-algebra workload and adds
regional neurochemistry, eligibility traces, delayed communication and slow
structural processes only where their state, cadence and ownership are explicit.

This document states what the project claims and — more importantly — what it
does not.

## What it is

- A fixed-topology simulator: `N` neurons, `M` synapses, one sparse matrix
  multiplication plus a few vector operations per tick
  (`O(BM + BN)` time, `O(M + BN)` memory).
- A reference implementation with several interchangeable runtimes: portable
  matrix, persistent-current (PELS), native fused matrix, native event-driven.
- A functional model of dopamine-gated three-factor plasticity: regional
  concentration, D1/D2 modulation, TD-RPE input, eligibility traces and a
  sign-preserving edge update.
- Validated against dense oracles and reference paths with recorded tolerances,
  on hardware stated with every number.

## What it is not

- **Not a biologically validated model.** The dynamics are a functional proxy.
  Parameters are not calibrated to any recording, connectome or experiment.
- **Not a whole-brain or connectome simulation.** The topology is generic
  random sparse or user-supplied; structural plasticity is a tested primitive,
  not a live developmental whole-brain runtime.
- **Not a benchmark result about self-play or learned control.** The Gomoku and
  pendulum probes are supervised/behaviour-cloning upper-bound experiments, not
  AlphaZero-style self-play or reinforcement learning.
- **Not proof that sparse events are always faster.** Event sparsity pays off
  only at low activity and only when eligibility scanning and weight updates are
  not on the same tick. The documents say where this stops.
- **Not a production neuromorphic stack.** No deployment tooling, no
  multi-GPU/multi-node, no hard real-time guarantees.

## Evidence boundaries

Every performance number is bound to a specific hardware, topology, batch size,
precision and command. The binding contracts are:

| Question | Authoritative document |
|---|---|
| What was measured, on what hardware | [results](results.md) |
| How to interpret the numbers and their limits | [efficiency](efficiency.md) |
| What counts as passing | [validation](validation.md) |
| What the functional biology does and does not mean | [biology](biology.md) |

Numbers from different stacks are not interchangeable. In particular, CPU
results in [experiments](experiments.md) are about correctness and parameter
efficiency, not about throughput.

## Design commitments

1. No implicit all-pairs neuron operation. Any feature that would introduce one
   must be rejected or reformulated over explicit candidates.
2. State has one owner and a documented lifetime; see
   [architecture](architecture.md).
3. Fast paths stay asynchronous: no per-tick device-to-host synchronization in
   the latency path unless statistics are explicitly requested.
4. Chemistry and structural change run at their own cadence, never inside the
   neural tick.
5. A claim that cannot be reproduced from a documented command is not a claim.

## Status

Version 0.1.0, research prototype. The milestone scorecard, what is integrated
and what remains standalone are in [status](status.md). Release verification is
recorded in [verification](verification.md).

---

[← Documentation index](index.md)
