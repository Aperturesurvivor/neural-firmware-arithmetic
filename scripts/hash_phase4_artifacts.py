from __future__ import annotations

import hashlib
import json
from pathlib import Path

PATHS = (
    Path("PHASE4_CONFIRMATORY_PROTOCOL.md"),
    Path("PHASE4_LAB_NOTEBOOK.md"),
    Path("PHASE4_EXECUTIVE_SUMMARY.md"),
    Path("phase4_results/confirmation_raw.json"),
    Path("phase4_results/confirmation_analysis.json"),
    Path("phase4_results/confirmation_summary.csv"),
    Path("phase4_results/confirmation_by_family.csv"),
    Path("paper_phase4/main.tex"),
    Path("paper_phase4/natural-language-deterministic-arithmetic.pdf"),
    Path("phase4_artifacts/router_pilot_v2.pt"),
    Path("phase4_artifacts/semantic_pilot_v2/semantic_unit.pt"),
    Path("phase4_artifacts/semantic_pilot_v2/semantic_control.pt"),
)
OUTPUT = Path("phase4_results/artifact_manifest.sha256.json")


def hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    rows = []
    for path in PATHS:
        if not path.is_file():
            raise FileNotFoundError(path)
        rows.append(
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "sha256": hash_file(path),
                "tracked": "phase4_artifacts/" not in str(path),
            }
        )
    OUTPUT.write_text(json.dumps({"files": rows}, indent=2) + "\n")
    print(OUTPUT)
    for row in rows:
        print(f"{row['sha256']}  {row['path']}")


if __name__ == "__main__":
    main()
