# Venue and release strategy

Policy and deadline information was verified on 2026-07-27 from official venue
pages.

## Recommended sequence

### 1. arXiv preprint: ready after author-controlled fields

The full ten-page report is scientifically useful now as a dated technical
record. arXiv accepts TeX source, requires accurate non-anonymous authorship,
may require first-time or new-category endorsement, and requires an
irrevocable distribution-license choice.

Recommended category: `cs.LG`, cross-listed to `cs.CL`.

NeurIPS 2026 policy permits a non-anonymous preprint during double-blind review,
provided the public version does not say that it is under review and the work
is not aggressively advertised in a way that undermines anonymity.

Official sources:

- https://info.arxiv.org/help/submit/index.html
- https://info.arxiv.org/help/prep.html
- https://info.arxiv.org/help/endorsement.html
- https://info.arxiv.org/help/license/index.html
- https://neurips.cc/Conferences/2026/MainTrackHandbook

### 2. Primary near-term target: MATH-AI at NeurIPS 2026

This is the best current fit. The workshop explicitly seeks architectures,
training methods, tool interfaces, reliability, robustness, and efficiency for
mathematical agents.

- Submission deadline: September 25, 2026, Anywhere on Earth.
- Format: four content pages, unlimited references and supplementary material.
- Review: double-blind, at least three reviews, no rebuttal.
- Reciprocal reviewing: at least one author agrees to review.
- Status: non-archival; accepted papers appear online and may later be sent to
  archival venues.
- Presentation: in-person poster in Atlanta on December 12 or 13, 2026.

The current result is suitable as a concept-and-feasibility workshop paper. A
clean second-model replication before September 25 would materially strengthen
it but is not required to preserve the current finding.

Official sources:

- https://mathai-2026.github.io/
- https://mathai-2026.github.io/cfp/
- https://openreview.net/group?id=NeurIPS.cc%2F2026%2FWorkshop%2FMATH-AI

### 3. Archival target after replication: TMLR

TMLR is the best full-paper destination after a clean cross-model replication
and, ideally, a bounded multi-call controller.

- Submissions are rolling rather than tied to one annual deadline.
- Review is double-blind and open-ended, with revision and discussion.
- The official TMLR LaTeX template is mandatory.
- Preprints and prior non-archival workshop versions are allowed.
- TMLR licenses submissions and published papers under CC BY 4.0.
- LLMs may be assistive tools, cannot be authors, and humans remain responsible
  for the entire paper.

The current single-model study is likely too narrow for the strongest TMLR
case. The second-model replication is the cleanest next increment.

Official sources:

- https://jmlr.org/tmlr/submissions.html
- https://jmlr.org/tmlr/author-guide.html
- https://jmlr.org/tmlr/editorial-policies.html

## Targets not recommended now

- NeurIPS 2026 main track: the May 2026 submission deadline has passed.
- NeurIPS 2026 Position Paper track: the deadline has passed and its special
  policy requires the final paper to be substantially human-written, allowing
  AI only for peripheral copy-editing. That is not a clean fit for the current
  collaboration history.
- ICML 2026 Mechanistic Interpretability workshop: it already occurred in July
  2026.
- A full archival submission before the replication: possible, but it would
  weaken the paper's central generality claim and reduce the value of the next
  experimental result.

## Working calendar

- By July 31: finalize metadata, accounts, license choice, and public code
  licensing decision.
- August 1-14: port the implant and run channel census/smoke tests on the
  second model.
- August 15-31: complete development without consuming the frozen confirmation
  set; freeze protocol and checkpoints.
- September 1-14: run three-seed confirmation and untouched-base comparison.
- September 15-20: update the four-page paper and supplement.
- September 25: MATH-AI submission deadline.
- After workshop submission: prepare the expanded TMLR manuscript; add a
  bounded multi-call result if it can be completed without contaminating the
  clean replication.
