# arXiv metadata draft

Verified against arXiv's submission and metadata documentation on
2026-07-27.

## Title

Deterministic "Neurons" Inside an Existing Transformer MLP: A Three-Seed
Causal Study of Natural-Language Addition, Routing, Preservation, and
Registered State in Qwen2.5-0.5B

## Authors

Josiah Wilson (Independent Researcher)

OpenAI Codex is acknowledged as an assisting system, not listed as an author.

## Abstract

We test whether a pretrained language model can learn to use a deterministic
calculator occupying a small activation subspace inside an existing
transformer MLP, rather than invoking an external tool or replacing an answer
after generation. We modify frozen Qwen2.5-0.5B-Instruct at decoder layer 16.
Twenty-eight of its 4,864 MLP coordinates form a typed interface: sixteen
learned route, operand-role, and digit coordinates feed a frozen decimal
ripple-carry adder, and twelve deterministic result coordinates return through
the ordinary MLP down projection. The interface has 25,088 learned weights
(0.0051% of the 494-million-parameter base); the calculator has zero. In a
frozen final audit, three independently learned interfaces answered 173/180
natural-language additions exactly, including 90/90 word problems. They
recovered 174/180 exact operand registers, all of which remained stable and
produced exact calculator trajectories. Calculator-result ablation left 9/180
answers correct, a paired causal loss on 165 examples. All 180 adversarial
negative prompts remained route-off and token-identical to untouched Qwen. An
untouched-base comparison produced 1/60 exact numeral-only responses under the
same eight-token budget. The compound protocol nevertheless failed two of five
gates because of six learned operand-framing errors and one downstream decoding
error. The results support the narrow causal implant hypothesis while leaving
multi-call reasoning and residual-native state unresolved.

The abstract is ASCII-only and below arXiv's 1,920-character limit.

## Categories

- Primary: `cs.LG` (Machine Learning)
- Cross-list: `cs.CL` (Computation and Language)
- Optional second cross-list if arXiv permits and the submitter prefers:
  `cs.NE` (Neural and Evolutionary Computing)

## Comments

10 pages, 0 figures, 9 tables. Three-seed causal study with retained negative
results and a retrospective same-prompt untouched-base comparison.

Do not write "under review at NeurIPS" or name a target venue in this field.

## License

User decision required. The recommended choice for compatibility with a later
TMLR submission is `CC BY 4.0`, because TMLR uses that license throughout its
submission and publication process. The arXiv license choice is irrevocable.

## Optional identifiers

- ORCID: user input required (optional)
- Report number: leave blank
- Journal reference: leave blank
- DOI: leave blank

## Source

Upload `dist/deterministic-neurons-qwen-arxiv-source.zip`. The archive contains
one standalone `main.tex`; arXiv prefers TeX source when the PDF was generated
from TeX.
