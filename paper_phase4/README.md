# Phase 4 paper

`natural-language-deterministic-arithmetic.pdf` is the primary technical report
for the natural-language routing and same-prompt comparison experiment.
`main.tex` is its source; figures are generated from the frozen confirmation by
`scripts/analyze_phase4_confirmation.py`.

Compile from the repository root:

```bash
tectonic paper_phase4/main.tex --outdir paper_phase4/build
cp paper_phase4/build/main.pdf \
  paper_phase4/natural-language-deterministic-arithmetic.pdf
```
