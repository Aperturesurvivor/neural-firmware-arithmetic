from __future__ import annotations

import hashlib
import json
from pathlib import Path

OUTPUT = Path("phase5_results/artifact_manifest.sha256.json")
TRAINING_MANIFEST = Path("phase5_results/training_v2/manifest.json")
EXPLICIT_PATHS = (
    Path("PHASE5_CONFIRMATORY_PROTOCOL.md"),
    Path("PHASE5_PILOT_LOG.md"),
    Path("PHASE5_LAB_NOTEBOOK.md"),
    Path("PHASE5_EXECUTIVE_SUMMARY.md"),
    Path("phase5_results/confirmation_analysis_v1.json"),
    Path("phase5_results/confirmation_summary_v1.csv"),
    Path("phase5_results/confirmation_comparisons_v1.csv"),
    Path("phase5_results/confirmation_by_family_v1.csv"),
    Path("paper_phase5/main.tex"),
    Path("paper_phase5/neural-firmware-versus-igc.pdf"),
    Path("scripts/run_phase5_confirmation.py"),
    Path("scripts/analyze_phase5_confirmation.py"),
)
DIRECTORIES = (
    Path("phase5_results/confirmation_raw_v1"),
    Path("phase5_results/training_v2"),
    Path("paper_phase5/figures"),
)


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def tracked(path: Path) -> bool:
    return "phase5_artifacts" not in path.parts


def main() -> None:
    training_manifest = json.loads(TRAINING_MANIFEST.read_text())
    checkpoint_paths = [
        Path(record["checkpoint"]) for record in training_manifest["records"]
    ]
    for record, checkpoint in zip(
        training_manifest["records"],
        checkpoint_paths,
        strict=True,
    ):
        observed = hash_file(checkpoint)
        expected = record["checkpoint_sha256"]
        if observed != expected:
            raise RuntimeError(
                f"checkpoint hash mismatch for {checkpoint}: "
                f"{observed} != {expected}"
            )
    paths = set(EXPLICIT_PATHS)
    paths.update(checkpoint_paths)
    for directory in DIRECTORIES:
        paths.update(path for path in directory.rglob("*") if path.is_file())
    rows = []
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
                "tracked": tracked(path),
            }
        )
    result = {
        "algorithm": "SHA-256",
        "files": len(rows),
        "bytes": sum(row["bytes"] for row in rows),
        "tracked_files": sum(row["tracked"] for row in rows),
        "local_checkpoint_files": sum(not row["tracked"] for row in rows),
        "entries": rows,
    }
    OUTPUT.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "files",
                    "bytes",
                    "tracked_files",
                    "local_checkpoint_files",
                )
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
