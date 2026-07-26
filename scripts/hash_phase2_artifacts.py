from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("phase2_artifacts/confirmatory_v1"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "phase2_results/confirmatory_v1/artifact_manifest.sha256.json"
        ),
    )
    args = parser.parse_args()
    root = args.artifact_directory.resolve()
    files = sorted(path for path in root.rglob("*") if path.is_file())
    entries = [
        {
            "path": str(path.relative_to(root)),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in files
    ]
    result = {
        "algorithm": "SHA-256",
        "artifact_root": str(args.artifact_directory),
        "files": len(entries),
        "bytes": sum(entry["bytes"] for entry in entries),
        "entries": entries,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in ("files", "bytes")}, indent=2))


if __name__ == "__main__":
    main()
