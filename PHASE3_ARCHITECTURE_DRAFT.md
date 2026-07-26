# Phase 3 Architecture Draft: Native Deterministic Transformer Unit

Status: design draft for feasibility work. Nothing in this document is
confirmatory or frozen.

Origin: Josiah Wilson's proposal to add or replace an internal transformer
unit with a locked calculator-like process, plus translation between ordinary
tokens/residuals and the deterministic representation.

## Research question

Can a frozen exact arithmetic cell be installed as a real module inside the
repeated block stack of a pretrained 0.5B-class causal transformer, consume
operands decoded from intermediate residual states, write its result back into
those states, and remain exact after subsequent native transformer layers?

## Why this is distinct from phase 2

Phase 2 injected a deterministic output symbol after all 24 Qwen transformer
blocks and immediately before the tied language-model head. Phase 3 must:

1. replace one entry in Qwen's repeated `model.layers` stack with a wrapper
   containing the original block plus the deterministic unit;
2. obtain operand values from hidden states produced at that internal depth;
3. execute the fixed arithmetic transition inside the wrapped forward call;
4. return a modified residual tensor to later unmodified Qwen blocks;
5. generate through Qwen's ordinary final normalization and tied output head.

No post-generation replacement or direct output-logit correction is allowed in
the main condition.

## Candidate architecture

### Controlled input grammar

The first experiment will use an explicit internal-register command whose
operand digits are separated:

```text
Use the internal addition register. Return only the exact integer.
A = 1 2 3
B = 4 5
```

Qwen represents each bare digit as a stable single token. The fixed structural
locator may identify the A and B spans and their token positions, but it may
not read digit values. Digit values must be classified from the selected
intermediate residual vectors by the learned input encoder.

The locator must full-match the registered request instead of searching for an
eligible substring. Eligibility is latched once per sequence. Quoted,
negated, nested, or otherwise ineligible commands must stay off for the whole
generation.

### Learned token/residual-to-register encoder

At insertion depth \(k\), the output of Qwen block \(k\) contains one residual
vector for each operand digit position. A learned affine classifier

\[
q_k(h_i) \rightarrow \{0,\ldots,9\}
\]

maps each selected residual to a typed decimal digit. It will be trained
separately with supervised digit labels, then frozen before end-to-end decoder
training. Exact register-decoding accuracy is a required intermediate metric.

The deterministic core receives only the encoder's predicted digit IDs and
fixed length masks. It will not receive ground-truth operands at inference.

### Frozen typed arithmetic cell

The arithmetic cell is a zero-trainable-parameter `torch.nn.Module` containing
immutable sum and carry lookup buffers. It performs masked decimal ripple-carry
addition on typed digit tensors and emits a typed sequence containing digits
0--9 plus end-of-answer.

The cell is executed from the inserted layer wrapper's forward path. Unit tests
will verify it against Python integer addition, including long carry chains and
unequal operand lengths.

### Learned register-to-residual decoder

An eleven-entry residual codebook maps each exact output symbol to the Qwen
hidden width. At teacher-forced answer positions, the selected code vector is
added to the internal residual tensor after block \(k\). All later Qwen blocks,
the final RMS normalization, and the existing tied output head remain active.

Only the digit encoder and residual codebook/interface are trainable in the
main condition. The Qwen weights and deterministic cell remain frozen.

### Internal layer wrapper

The implementation will replace a selected Qwen decoder layer in the existing
`ModuleList` with:

```text
Qwen block k
    -> learned residual-to-digit encoder
    -> frozen typed addition cell
    -> learned symbol-to-residual decoder
    -> Qwen block k+1 ... block 24
    -> ordinary final norm and tied LM head
```

The wrapper will preserve Qwen's attention type, cache behavior, and calling
signature so ordinary cached autoregressive generation still works.

## Experimental ladder

1. Verify stable single-digit tokenization and structural position recovery.
2. Measure digit-classification accuracy at candidate depths after blocks
   6, 12, 18, and 22.
3. Train a residual decoder independently at each feasible depth.
4. Measure target-symbol decodability:
   - immediately after the deterministic unit;
   - after every remaining transformer block;
   - at the final output head.
5. Select an insertion depth using pilot-only seeds and examples.
6. Freeze a multi-seed protocol with:
   - frozen base;
   - internal unit disabled;
   - internal deterministic unit;
   - same-depth learned control without deterministic state;
   - phase-2 output-adjacent bridge as a location control;
   - direct deterministic logit control as a structural bound.
7. Evaluate in-range, length-OOD, carry-chain, routing-preservation, quoted
   commands, and internal causal interventions.

## Required causal tests

- **Unit off:** disabling the internal residual write must remove the arithmetic
  gain while leaving all base weights unchanged.
- **State permutation:** permuting deterministic symbols between batch examples
  must permute/corrupt outputs accordingly.
- **Wrong-state intervention:** replacing one internal symbol with a known
  wrong digit must causally change the corresponding output.
- **Downstream trace:** correct-symbol logit margins must be measured
  immediately after insertion and after each subsequent block.
- **No direct channel:** the main condition may not alter output logits or
  replace decoded text.

## Key risks

1. Early-layer residuals may not linearly expose individual digit identity.
2. A hard argmax digit encoder is not differentiable; staged training is
   necessary unless a straight-through estimator is justified.
3. Downstream layers may overwrite a code that is perfectly decodable
   immediately after insertion.
4. Teacher-forced answer positions may make training easier than free-running
   generation; all final accuracy must be autoregressive.
5. Mutable generation context can silently leak between examples. Context
   reset and cache alignment require explicit tests.
6. A controlled space-separated grammar narrows the claim. It is acceptable
   for the architectural proof but must not be described as general natural
   language arithmetic.

## Provisional feasibility decision

Proceed with the typed-register design. Do not treat “move the phase-2 symbol
embedding to an earlier layer” as sufficient: that would test downstream
transport but would still obtain operands from the external parser. Phase 3's
main condition must compute digit values from learned internal residual
encodings before the frozen cell runs.
