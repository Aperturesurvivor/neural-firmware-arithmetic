# Phase 3 paper

`native-deterministic-transformer-unit.pdf` is the primary technical report
for the internal transformer-unit experiment. `main.tex` is its source;
figures are generated from the frozen study by `scripts/analyze_phase3.py`.

Compile from the repository root:

```bash
tectonic paper_phase3/main.tex --outdir paper_phase3/build
cp paper_phase3/build/main.pdf \
  paper_phase3/native-deterministic-transformer-unit.pdf
```
