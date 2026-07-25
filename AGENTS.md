# Research Integrity Instructions

This repository is a controlled machine-learning experiment.

1. Preserve the preregistered primary outcome in `PROTOCOL.md`.
2. Never replace an unsuccessful run without retaining its configuration and
   result record.
3. Separate pilot-driven engineering changes from confirmatory study results.
4. Record every reported run's seed, configuration, environment, commit, and
   raw metrics.
5. Do not describe deterministic execution as end-to-end mathematical
   correctness. Parsing, routing, and decoding remain separate failure modes.
6. Do not claim novelty until the literature review establishes the boundary.
7. Attribute the originating research hypothesis to Josiah Wilson and the
   implementation/analysis assistance to OpenAI Codex.
8. Generated datasets must be reproducible from committed code and seeds.
9. Keep model checkpoints out of Git; keep compact metrics and manifests in
   Git.
10. Report negative and null results plainly.

