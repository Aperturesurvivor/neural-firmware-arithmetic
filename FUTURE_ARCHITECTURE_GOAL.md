# Follow-up Goal: A Native Deterministic Transformer Unit

Origin: Josiah Wilson, clarified 2026-07-25.

Start this as a new `/goal` after the phase 2 pretrained-LLM residual-bridge
experiment is complete, but only if phase 2 is successful.

## Josiah's intended architecture

The long-term idea is not merely a calculator tool called after generation and
not merely a correction applied to output logits. It is a redesigned
transformer architecture containing a native deterministic computational unit:

1. Select or add an internal unit within the repeated transformer computation.
2. Make that unit implement an exact calculator operation rather than a
   statistically learned approximation.
3. Add a learned input interface that translates token/residual
   representations into a typed representation the deterministic unit can
   consume.
4. Add a learned output interface that writes the exact result back into the
   transformer's residual stream.
5. Preserve the deterministic unit's weights or transition rules while
   training the interfaces and surrounding model.
6. Test whether later layers preserve or corrupt the exact internal result.

“Extra neuron” is conceptual shorthand. A practical implementation may need a
small group of fixed units, a recurrent state machine, a typed register, or a
compiled attention/MLP circuit because decimal addition requires digit state,
position, and carry.

## Required distinction from phase 2

Phase 2 injects a deterministic sidecar through a learned residual bridge
immediately before the existing output head. The follow-up must move the
deterministic computation into or between repeated transformer blocks and pass
its result through subsequent native layers. It should therefore test an
architectural change rather than an output-adjacent interface.

## Candidate experimental ladder

1. Insert a frozen typed register/carry cell between two transformer blocks.
2. Train only token-to-register and register-to-residual projections.
3. Compare injection at early, middle, and late blocks.
4. Measure exactness immediately after the deterministic unit and after every
   subsequent block.
5. Add an invariant-preserving residual channel or protected subspace if
   ordinary layers corrupt the value.
6. Attempt to compile the fixed cell into ordinary attention/MLP weights as a
   more ambitious final variant.

## Success criterion

An arithmetic result is computed by a fixed unit within the repeated
transformer stack, survives downstream transformer computation, and improves
end-to-end exact-match accuracy outside trained operand lengths without
post-generation correction.
