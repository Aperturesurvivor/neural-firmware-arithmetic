# Phase 5 paper

`neural-firmware-versus-igc.pdf` is the primary technical report for the
three-seed typed neural-firmware versus IGC-style comparison. `main.tex` is
the source; figures are generated directly from the frozen confirmation by
`scripts/analyze_phase5_confirmation.py`.

Compile from the repository root:

```bash
tectonic paper_phase5/main.tex --outdir paper_phase5/build
cp paper_phase5/build/main.pdf \
  paper_phase5/neural-firmware-versus-igc.pdf
```
