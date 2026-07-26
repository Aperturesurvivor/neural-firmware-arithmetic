# Phase 5 paper

`neural-firmware-versus-igc.pdf` is the primary technical report for the
three-seed typed neural-firmware versus IGC-style comparison. `main.tex` is
the source; figures are generated directly from the frozen confirmation by
`scripts/analyze_phase5_confirmation.py`.

Compile from the repository root:

```bash
python3 /Users/josiahwilson/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py \
  "$PWD/paper_phase5/main.tex" \
  --output-directory "$PWD/paper_phase5/build"
cp paper_phase5/build/main.pdf \
  paper_phase5/neural-firmware-versus-igc.pdf
```
