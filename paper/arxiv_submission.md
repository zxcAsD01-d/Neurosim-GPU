# arXiv submission package

Everything needed to submit the report, plus the decisions that cannot be made
for you.

## Metadata

**Title**

```text
What Does a Neural Simulation Tick Cost? A Cost-Normalized Evaluation of a Sparse GPU Simulator
```

**Authors**

```text
zxcAsD01
```

> arXiv requires an author name and, for a first submission, usually an
> affiliation. A pseudonymous handle is not accepted as a legal name for new
> accounts; enter your real name here and keep the handle in the comments or the
> repository URL. Add `Independent researcher` as the affiliation if you have no
> institutional one.

**Abstract** (plain text, no LaTeX — paste as-is)

```text
Neural simulators are usually evaluated against other simulators, which mixes
algorithmic differences with implementation quality. We argue for a narrower and
more portable denominator: the cost of one sparse matrix multiplication over the
same synapses, on the same device, with the same batch. We present a
fixed-topology sparse simulator built around that accounting and report what the
denominator reveals. On a laptop GPU, a complete neural tick - sparse reduction,
tanh, leak, gain and clip - costs 0.74-0.82x a same-parameter cuSPARSE
multiplication at batch one and 2.8-6.6% more than a matching custom SpMM; the
fused kernel beats the library call only because it fuses work the library
cannot. Persistent-current event propagation is 3.0-4.5x faster than the matrix
path when 2.6% of edges emit, and on trained MNIST networks propagating one fifth
of the edges changes accuracy by 0.1 percentage points. We also report where the
approach does not pay: the reference event path is 5-81x slower than dense
propagation on CPU, eligibility scanning destroys whole-tick event sparsity, and a
dopamine-gated online plasticity rule produced no measurable behavioural change
in our control task - an ablation shows the rule had no leverage on the synapses
that mattered. We release the implementation, 105 experiment runs, raw payloads
and a device-agnostic performance harness so that every number can be regenerated
or contradicted.
```

**Categories**

| Field | Value | Why |
|---|---|---|
| Primary | `cs.NE` (Neural and Evolutionary Computing) | simulation of spiking networks |
| Cross-list | `cs.LG` | autodiff-based training, benchmark discipline |
| Cross-list (optional) | `cs.MS` | sparse kernels and performance methodology |

**Comments**

```text
10 pages, 5 figures, 4 tables. Code, data and reproduction scripts:
https://github.com/zxcAsD01-d/Neurosim-GPU
```

**License**

Choose `CC BY 4.0` if you want the widest reuse, or the default arXiv
non-exclusive licence if you plan to publish a journal version and want to keep
options open. CC BY is the usual choice for a technical report that you also
mirror on GitHub.

## Submission steps

1. Create an arXiv account if you do not have one; complete the author profile.
2. Check whether your account needs **endorsement** for `cs.NE` (first-time
   submitters usually do). If so, ask a colleague who has published in the
   category, or start with the cross-list category you can already post to.
3. Start a new submission: `TeX` source, upload `main.tex`, `references.bib`,
   `tables/*.tex`, `figures/*.tex`. arXiv compiles with TeX Live; the source is
   plain LaTeX and needs no shell escape.
4. Alternative: upload `main.pdf` only (arXiv accepts a PDF, but source is
   strongly preferred because it enables HTML rendering and long-term
   preservation).
5. Paste the metadata above; preview the PDF; fix any compile warnings.
6. Submit. Moderation typically takes one to three working days for a new
   account.

## After acceptance

- Add the arXiv ID to `CITATION.cff` (`preferred-citation`) and to the README.
- Archive the release on Zenodo to get a DOI, and cite the DOI in the paper's
  artifact paragraph.
- Keep `paper/main.tex` and the repository in sync: regenerate the assets and the
  PDF whenever a number changes.

## Do not submit if

- Any number in the abstract cannot be traced to a table or a committed payload.
- The author field contains a pseudonym (it will be rejected or later corrected
  publicly).
- You have not decided on the licence.
- The repository is private: the artifact claim in the abstract requires a public
  link.
