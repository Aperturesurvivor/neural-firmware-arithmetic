# Phase 3 Lab Notebook

This is the chronological, replication-oriented record for the native
deterministic transformer-unit study. Times use America/Los_Angeles unless
otherwise noted. Phase 3 begins after the completed phase-2 reporting commit
`c16fbb3d66e8db41aec0281c98b56e40512dd476`.

## 2026-07-25 — Goal authorization

- Josiah explicitly instructed Codex to start the next persistent goal.
- This explicit instruction supersedes the earlier conditional instruction to
  start only if phase 2 met every success criterion. Phase 2's formal verdict
  remains unsuccessful and is not reinterpreted.
- Created the active goal:
  “Design, implement, run, and document a reproducible experiment that
  modifies a pretrained 0.5B-class transformer to include a genuinely internal
  frozen deterministic arithmetic unit inside or between repeated transformer
  blocks, with learned token/residual encoders and decoders, rigorous controls
  and preservation tests, and a full research report; text Josiah a
  notification-length success/failure summary when complete.”
- Completion notification by text remains explicitly authorized.

## 2026-07-25 — Initial architecture audit

- Read `FUTURE_ARCHITECTURE_GOAL.md`, the phase-2 bridge implementation, and the
  fixed ripple-carry implementation.
- Confirmed that phase 2's bridge is output-adjacent: it modifies the final
  normalized residual immediately before the tied language-model head.
- Inspected the installed Transformers 4.57.6 source for `Qwen2Model.forward`
  and `Qwen2DecoderLayer.forward`.
- Qwen's 24 decoder blocks are iterated directly from `model.layers`; each
  block accepts and returns a hidden-state tensor while sharing the attention
  cache. A wrapper that proxies the original block's `attention_type` and
  calling signature can therefore be installed as a real entry in the repeated
  `ModuleList`.
- Rejected the weakest candidate—moving the phase-2 symbol vector earlier
  while retaining externally parsed operands—as insufficient for the main
  phase-3 claim. It would test downstream transport but not learned
  residual-to-register translation.
- Selected a staged typed-register candidate for feasibility work:
  1. a controlled grammar with space-separated single digit tokens;
  2. a learned affine digit classifier over intermediate residual vectors;
  3. a frozen tensorized ripple-carry cell consuming predicted digits;
  4. a learned eleven-symbol residual codebook written after a selected block;
  5. all remaining native Qwen blocks and the original output head downstream.
- Wrote `PHASE3_ARCHITECTURE_DRAFT.md`. It is explicitly a pilot design, not a
  frozen confirmatory protocol.

## 2026-07-25 — Typed cell implementation and first caught defect

- Added `src/neural_firmware/internal_data.py` with the controlled prompt
  grammar, strict full-match eligibility, operand character-span location, and
  reproducible example generation.
- Added `src/neural_firmware/internal_firmware.py` with:
  - a zero-parameter tensorized ripple-carry cell;
  - a learned residual-to-digit classifier;
  - an eleven-symbol residual decoder;
  - gather, exact-plan, and residual-injection operations.
- Added randomized unit tests using 204 operand pairs up to 18 digits,
  explicitly including zero, unequal widths, and a twelve-position carry
  chain.
- The first test run failed: `9+1` produced `0` when evaluated in the same
  batch as longer operands. Cause: the global loop continued updating shorter
  rows' carry state through padding columns needed by longer rows.
- Corrective change: added a per-row active-width mask and froze each row's
  carry after its declared maximum operand width. This defect was found before
  any pretrained-model feasibility result was generated.
- Reran the focused tests after correction: 2/2 passed; Ruff passed.

## 2026-07-25 — Token-to-register structural feasibility

- Added offset-based token-position recovery to `internal_data.py`. It formats
  the request with Qwen's actual chat template, finds the user request exactly
  once, and maps only the structurally located operand character positions to
  tokenizer offsets.
- Added and ran `scripts/check_phase3_tokenization.py` with:
  - pinned Qwen revision
    `7ae557604adf67be50417f59c2c2f167def9a775`;
  - seed 31415;
  - 1,000 examples with independently sampled one- to twenty-digit operands.
- Result:
  - 1,000/1,000 operand digit sequences recovered at the correct token
    positions;
  - 1,000/1,000 quoted-command variants rejected by the strict full-match
    eligibility rule;
  - formatted chat length ranged from 50 to 126 tokens;
  - no failures.
- Compact output:
  `phase3_results/tokenizer_feasibility.json`.
- This check validates only structural position recovery and tokenizer
  stability. It does not yet validate the learned residual-to-digit encoder.
