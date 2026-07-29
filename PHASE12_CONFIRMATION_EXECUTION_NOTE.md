# Phase 12 Confirmation Execution Note

The Phase 12 protocol and canonical prompt manifest were frozen before model
evaluation. Two execution attempts were later stopped because another
research process needed the shared MPS accelerator:

1. the first attempt had not completed a single prompt and wrote no result;
2. the second had generated conditions in memory but was stopped at the
   user's request and also wrote no result.

Neither attempt exposed aggregate metrics or gate outcomes. No prompt,
checkpoint, weight, threshold, decoding rule, metric, or gate was changed.

After those attempts, the evaluator gained operational checkpoint/resume
support. It atomically saves raw per-prompt outputs every ten completed rows
under the ignored `phase12_artifacts/` directory. A resumed process accepts
that progress only when the frozen manifest hash, canonical row hash, model
identity and revision, progress schema, and evaluator source hash all match.
It also verifies that every stage is a contiguous prefix of the frozen rows.

SIGINT or SIGTERM requests a pause after the current prompt and then releases
the model and MPS cache. Because generation is greedy, the model is frozen,
and every prompt is evaluated independently from a fresh prompt context,
process boundaries do not alter the experimental condition. The tracked final
result is still written only after every frozen condition is complete.

This is an execution-reliability amendment, not a protocol or analysis
amendment. The frozen protocol and manifest hashes remain unchanged.
