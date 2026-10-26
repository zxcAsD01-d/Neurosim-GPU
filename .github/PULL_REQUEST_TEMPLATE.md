## What changed

<!-- One paragraph. What behaviour or document changed, and why. -->

## Contract impact

- [ ] No contract change
- [ ] Numerical/runtime contract (link the document you updated):
- [ ] Performance/measurement claim (link the row you added):

## Verification

<!-- Commands you actually ran and what they printed. -->

```text
python -m unittest discover -s tests -q
python scripts/check_links.py
```

- [ ] Tests pass on CPU
- [ ] Documentation links resolve
- [ ] New behaviour has a regression test
- [ ] Any new number names its hardware, graph size, batch, dtype and command

## Not verified

<!-- Be explicit: no CUDA device, no 125M-edge graph, etc. -->
