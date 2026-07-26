# Phase 4 paper

`natural-language-deterministic-arithmetic.pdf` is the primary technical report
for the natural-language routing and same-prompt comparison experiment.
`main.tex` is its source; figures are generated from the frozen confirmation by
`scripts/analyze_phase4_confirmation.py`.

Compile from the repository root:

```bash
python3 /Users/josiahwilson/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py \
  "$PWD/paper_phase4/main.tex" \
  --output-directory "$PWD/paper_phase4/build"
cp paper_phase4/build/main.pdf \
  paper_phase4/natural-language-deterministic-arithmetic.pdf
```
