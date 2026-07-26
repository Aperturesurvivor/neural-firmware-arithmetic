# Phase 3 paper

`native-deterministic-transformer-unit.pdf` is the primary technical report
for the internal transformer-unit experiment. `main.tex` is its source;
figures are generated from the frozen study by `scripts/analyze_phase3.py`.

Compile from the repository root:

```bash
python3 /Users/josiahwilson/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py \
  "$PWD/paper_phase3/main.tex" \
  --output-directory "$PWD/paper_phase3/build"
cp paper_phase3/build/main.pdf \
  paper_phase3/native-deterministic-transformer-unit.pdf
```
