# Phase 2 paper

`neural-firmware-pretrained-llm.pdf` is the primary research report for the
pretrained-LLM experiment. Its source is `main.tex`; figures are generated from
the frozen study summaries by `scripts/analyze_phase2.py`.

Compile from the repository root with the bundled LaTeX workflow:

```bash
python3 /Users/josiahwilson/.codex/plugins/cache/openai-bundled/latex/0.2.4/scripts/compile_latex.py \
  "$PWD/paper_phase2/main.tex" \
  --output-directory "$PWD/paper_phase2/build"
cp paper_phase2/build/main.pdf paper_phase2/neural-firmware-pretrained-llm.pdf
```
