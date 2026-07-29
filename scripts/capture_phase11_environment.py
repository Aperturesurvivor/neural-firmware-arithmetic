from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

import numpy
import scipy
import torch
import transformers

from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION

OUTPUT = Path("phase11_results/environment_audit.json")
UV_LOCK = Path("uv.lock")
PYPROJECT = Path("pyproject.toml")
MANIFEST = Path("phase11_results/frozen_prompt_manifest.json")
CONFIRMATION = Path("phase11_results/confirmation.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit() -> str:
    return subprocess.check_output(
        ("git", "rev-parse", "HEAD"),
        text=True,
    ).strip()


def main() -> None:
    manifest = json.loads(MANIFEST.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    checkpoints = {
        seed: {
            "source": {
                "path": values["source"]["path"],
                "sha256": sha256(Path(values["source"]["path"])),
                "bytes": Path(values["source"]["path"]).stat().st_size,
            },
            "candidate": {
                "path": values["candidate"]["path"],
                "sha256": sha256(Path(values["candidate"]["path"])),
                "bytes": Path(values["candidate"]["path"]).stat().st_size,
            },
        }
        for seed, values in confirmation["checkpoints"].items()
    }
    payload = {
        "status": "phase11_environment_provenance_audited_posthoc",
        "scope_note": (
            "Captured immediately after confirmatory evaluation from the same "
            "workspace and machine; this was not emitted contemporaneously by "
            "the evaluator."
        ),
        "audit_commit": git_commit(),
        "protocol_implementation_commit": manifest["implementation_commit"],
        "confirmation_implementation_commit": confirmation[
            "implementation_commit"
        ],
        "model": {
            "id": PHASE8_MODEL_ID,
            "revision": PHASE8_MODEL_REVISION,
        },
        "runtime": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "scipy": scipy.__version__,
            "mps_available": torch.backends.mps.is_available(),
            "execution_device": (
                "mps" if torch.backends.mps.is_available() else "cpu"
            ),
        },
        "dependency_provenance": {
            "uv_lock_sha256": sha256(UV_LOCK),
            "pyproject_sha256": sha256(PYPROJECT),
        },
        "artifacts": {
            "manifest_sha256": sha256(MANIFEST),
            "confirmation_sha256": sha256(CONFIRMATION),
            "checkpoints": checkpoints,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
