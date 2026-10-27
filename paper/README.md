# Technical report

The LaTeX source of the paper *"What Does a Neural Simulation Tick Cost? A
Cost-Normalized Evaluation of a Sparse GPU Simulator"*, plus the scripts that
generate every table and figure from the committed experiment results.

## Build

```powershell
# any TeX distribution works; tectonic downloads only what it needs
tectonic -X compile main.tex

# or, with a full TeX Live / MiKTeX install
latexmk -pdf main.tex
```

The compiled PDF is committed as `main.pdf` so the report is readable without a
TeX toolchain.

## Regenerate the assets

Every table and figure in the paper is generated from the repository, not typed
by hand:

```powershell
python paper/make_assets.py     # writes paper/tables/*.tex and paper/figures/*.tex
python paper/check_tex.py       # verifies inputs, citations and brace balance
```

Inputs:

- `results/raw/*.json` — the 105-run experiment grid (accuracy, survival,
  activity, corruption rows);
- `paper/data/gpu_core.csv`, `paper/data/gpu_event.csv` — the accepted GPU
  numbers transcribed from `docs/results.md`, with hardware recorded there.

If the paper and the repository disagree, the repository payload wins; regenerate
the assets and the PDF rather than editing the tables by hand.

## Files

| Path | Contents |
|---|---|
| `main.tex` | the paper |
| `references.bib` | bibliography (11 entries) |
| `tables/` | generated LaTeX tables |
| `figures/` | generated pgfplots figures |
| `data/` | transcribed GPU measurements |
| `make_assets.py` | table/figure generator |
| `check_tex.py` | structural check (inputs, cites, braces, non-ASCII) |
| `arxiv_submission.md` | metadata and submission steps for arXiv |
| `CHECKLIST.md` | pre-submission checklist |
