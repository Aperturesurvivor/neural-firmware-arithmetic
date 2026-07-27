# Publication package

This directory stages publication materials without making anything public.

## Current recommendation

1. Post the full Phase 7 report as an arXiv preprint after the author confirms
   metadata and the irrevocable license choice.
2. Submit the focused four-page paper to the non-archival MATH-AI workshop at
   NeurIPS 2026 by September 25, 2026.
3. Use the remaining time before that deadline for the clean second-model
   replication in `../PHASE8_SECOND_MODEL_REPLICATION_PROTOCOL_DRAFT.md`.
4. After the replication, expand the manuscript and submit it to TMLR for
   archival peer review.

## Contents

- `PREPRINT_METADATA.md`: copy-ready arXiv fields.
- `SUBMISSION_READINESS.md`: completed checks and user-controlled decisions.
- `VENUE_STRATEGY.md`: verified venue fit, policies, dates, and sequencing.
- `arxiv_source/`: minimal full-report source package.
- `mathai2026/`: anonymous MATH-AI workshop manuscript and official style.
- `dist/`: generated PDFs, upload archives, and checksums.

Run `python3 publication/build_packages.py` from the repository root to rebuild
the upload archives and checksum manifest.

No account login, external upload, license grant, or public repository change
is performed by the packaging script.
