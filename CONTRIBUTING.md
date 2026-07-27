# Contributing

This repository is a research artifact. Contributions that improve
reproducibility, identify errors, add controlled replications, or clarify the
claim boundary are welcome.

Before opening a pull request:

1. Open an issue for changes that alter an experiment, metric, or
   interpretation.
2. Preserve preregistered outcomes and negative runs. Never replace an
   unfavorable result with a later run.
3. Separate exploratory work from confirmatory work and label it explicitly.
4. Record model revisions, seeds, configurations, commands, and environment
   details needed to reproduce a result.
5. Do not add model weights, credentials, private filesystem paths, personal
   data, or generated artifacts that are excluded by `.gitignore`.
6. Run the local checks:

   ```bash
   uv sync --frozen --extra dev
   uv run ruff check .
   uv run pytest
   ```

Keep pull requests focused and describe which protocol or result they affect.
Where a contribution changes a reported number, include both the machine-
readable output and the corresponding documentation update.

The repository does not yet grant an open-source license. Until a license and
contribution terms are posted, please use issues for replication reports and
discussion rather than submitting code.
