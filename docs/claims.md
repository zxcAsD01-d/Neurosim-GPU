# Claims, evidence, and what would refute them

Three different questions get conflated in project descriptions, so this file
separates them: what is **implemented**, what is **new**, and what would count as
**field-changing**. The third section is a claim ledger: every sentence the
project is willing to defend, its current status, and the experiment that would
falsify it.

## 1. What is implemented

| Capability | Code | Verified by |
|---|---|---|
| Fixed-topology sparse core with two index views and weight versioning | `src/neurosim_gpu/topology.py` | `tests/test_topology_io.py`, `test_runtime.py` |
| Portable matrix runtime, persistent-current (PELS) runtime | `runtime.py` | `test_runtime.py` (dense oracle, threshold zero equivalence) |
| Fused CUDA matrix and event kernels | `csrc/runtime.cu`, `native.py` | `benchmarks/validate_native.py` (CUDA only) |
| Regional dopamine chemistry, D1/D2 occupancy, TD-RPE | `chemistry.py`, `dopamine.py` | `test_chemistry.py`, `test_dopamine.py` |
| Eligibility traces and dopamine-gated three-factor updates | `dopamine.py`, `neurochem.py` | `test_dopamine.py`, `test_neurochem.py` |
| Multi-rate scheduler, topology-boundary transaction | `scheduler.py` | `test_scheduler.py` |
| Runtime/communication checkpoints, deterministic replay | `checkpoint.py`, `continual.py` | `test_checkpoint.py`, `test_continual.py` |
| Delayed communication, migration, structural plasticity primitives | `communication.py`, `development.py` | `test_communication.py`, `test_development.py` |
| Device-agnostic performance harness | `benchmarks/performance_suite.py` | `results/performance_cpu.json` |
| Task-level experiment grid (105 runs) with raw payloads | `experiments/` | `results/raw/`, `results/REPORT.md` |

"Implemented" means the code exists, is tested on CPU, and the CUDA paths have a
validation script. It does not mean the CUDA paths were validated on this machine
(there is no CUDA device here — see [verification](verification.md)).

## 2. What is new, and what is not

**New (defensible as a contribution):**

1. **A cost-normalized reporting protocol** with two denominators — the library
   SpMM call and a matching custom SpMM — plus explicit controls. Most simulator
   papers compare simulator against simulator; this isolates the neural update
   from the sparse math.
2. **Tick-parity measurements** from 1M to 50M synapses, including the
   decomposition "dynamics cost 2.8–6.6% of a matching SpMM at batch one".
3. **A bounded-error characterization on trained weights** rather than random
   ones: 56 trained MNIST networks, threshold sweep, accuracy change against the
   active-edge fraction.
4. **A negative result with its diagnosis**: dopamine-gated online plasticity did
   nothing in the control task, and an ablation showed the rule had no leverage
   (zeroing all recurrent weights still leaves 193.8/200 steps).
5. **An integrated stack with explicit cost accounting**: chemistry, eligibility,
   delays and structural change each carry a documented cadence and cost, and the
   documents state where event sparsity stops working.

**Not new — do not claim these:**

- Sparse spiking/rate simulation, event-driven propagation, three-factor
  plasticity, dopamine reward-prediction-error models, or CSR/SpMM kernels.
- Superiority over Brian 2, NEST, GeNN, Norse or snnTorch: no head-to-head
  comparison has been run.
- A faster SpMM kernel: against the *matching* custom SpMM the fused tick is
  0.92–1.07×, i.e. the update is cheap, not the SpMM.
- Biological realism: the chemistry is a functional proxy, not calibrated.

## 3. Claim ledger

Status legend: **supported** (measured, in scope) · **partial** (measured with a
caveat) · **unverified** (not measured) · **refuted** (our own data contradicts
the strong form).

| # | Claim | Status | Evidence | Would be refuted by |
|---|---|---|---|---|
| K1 | A complete neural tick costs about one sparse multiplication of the same size | supported | `docs/performance.md`, Tab. 1 of the report | a same-parameter benchmark where the update doubles the SpMM time |
| K2 | The neural update adds 2.8–6.6% over a matching custom SpMM | supported | `docs/results.md`, `results/performance_cpu.json` | a re-measurement outside that range on the same stack |
| K3 | The fused tick is 0.74–0.82× the PyTorch CSR SpMM call at batch one | partial | `docs/results.md` | — but see the denominator caveat below |
| K4 | Native event propagation is 3.0–4.5× faster at 2.64% active edges | partial | `docs/results.md`, event table | an activity sweep that shows no crossing point, or a crossing below the claimed activity |
| K5 | Emitting-event thresholds cost little accuracy on trained networks | partial (forward only, eligibility off) | `docs/experiments.md`, Tab. 2 | an accuracy collapse at 20% activity on a different task |
| K6 | Eligibility destroys whole-tick event sparsity | supported by design + cost model | `docs/efficiency.md`, `docs/architecture.md` | a sparse eligibility scheme that keeps the event path sparse |
| K7 | Memory is 24 B/synapse mutable and ~16 B/synapse read-only | supported (measured at ≤1M synapses) | `docs/efficiency.md`, `topology.storage_bytes()` | a different layout achieving materially fewer bytes |
| K8 | Training forward+backward is 1.075× a same-parameter sparse RNN | partial (one configuration) | `docs/results.md` | a second configuration or a fused rollout adjoint changing the ratio |
| K9 | Fixed sparse topology is parameter-efficient | supported | `results/REPORT.md` | — |
| K10 | Fixed sparse topology beats a linear classifier on MNIST | **refuted** | 91.1% vs 91.8% at 7.8k params | — |
| K11 | The sparse core is more robust to Gaussian input noise | partial (one corruption family, 3 seeds) | `results/REPORT.md` Fig. 3 | a different noise model or a tuned baseline closing the gap |
| K12 | Dopamine-gated online plasticity adapts a cloned controller | **refuted in our testbed** | 129.1 vs 130.3 steps; ablation 193.8/200 without recurrence | a testbed where the plastic synapses carry behaviour |
| K13 | The 125M-synapse configuration runs end to end | unverified (projection only) | `docs/results.md` | a successful run, or a run that does not fit |
| K14 | The simulator is faster than existing simulators | unverified | — | a head-to-head benchmark |

### The denominator caveat (K3)

`raw-spmm` is PyTorch's `torch.sparse.mm` over the topology's CSR arrays. PyTorch
CSR support is beta and may not select the fastest library routine. So K3 reads
as "faster than the library call we can express from PyTorch at that moment", not
"faster than cuSPARSE at its best". The claim that survives any reasonable
denominator is K2.

## 4. What would raise the claim

Three experiments, in order of value. None of them is a documentation task.

### E1 — Scale, not extrapolation

Run 1M → 10M → 25M → 50M → 125M synapses on one device, reporting per-tick
latency, effective throughput, peak device memory and achieved bandwidth. The
125M point should be a *prebuilt* topology, because random construction has a
larger temporary memory peak than the run itself.

**Convincing if:** tick latency stays within the bandwidth-bound envelope, peak
memory matches the `16M + 8N` / `24M + 8N` model, and nothing silently falls back
to a slower path.

### E2 — An activity sweep, not a single point

Sweep the active-edge fraction from 0.1% to 100% (not just θ values: also vary
the activity pattern, the batch and the degree distribution), measure both paths,
and report the crossing point. Then use it to replace the fixed matrix/event
choice with a policy that needs no host synchronisation.

**Convincing if:** the crossing point is predictable from the cost model, and an
automatic policy tracks it within a few percent of the best fixed choice.

### E3 — A task where the choice matters

Pick one reproducible network-dynamics task and compare against a mature
implementation (Brian 2, GeNN or Norse) on accuracy, latency, memory and training
cost. Check that the event path does not change the model's behaviour beyond the
stated error bound, and that long integrations do not drift.

**Convincing if:** the simulator matches the reference's accuracy at comparable
or lower cost, and the event path's error stays inside its contract.

## 5. One sentence to defend

> On a GPU, a fixed-topology sparse neural simulator can run large networks with
> a per-tick neural-update overhead of a few percent over a matching sparse
> multiplication, and can cut tick cost at low activity with native event
> propagation.

Everything else — "breakthrough", "faster than existing simulators", "brain-scale"
— is a hypothesis until E1–E3 are done.

---

[← Documentation index](index.md)
