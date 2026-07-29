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

OUTPUT = Path("phase12_results/environment_audit.json")
UV_LOCK = Path("uv.lock")
PYPROJECT = Path("pyproject.toml")
MANIFEST = Path("phase12_results/frozen_prompt_manifest.json")
CONFIRMATION = Path("phase12_results/confirmation.json")
ANALYSIS = Path("phase12_results/analysis.json")
EVALUATOR = Path("scripts/evaluate_phase12_confirmation.py")


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
            condition: {
                "path": values[condition]["path"],
                "sha256": sha256(Path(values[condition]["path"])),
                "bytes": Path(values[condition]["path"]).stat().st_size,
            }
            for condition in ("phase11_control", "phase12_candidate")
        }
        for seed, values in confirmation["checkpoints"].items()
    }
    payload = {
        "status": "phase12_environment_provenance_audited_posthoc",
        "scope_note": (
            "Captured immediately after confirmation and analysis from the "
            "same workspace and machine. Restart-safe evaluator progress "
            "records the execution-source hash independently."
        ),
        "audit_commit": git_commit(),
        "protocol_implementation_commit": manifest[
            "implementation_commit"
        ],
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
            "analysis_sha256": sha256(ANALYSIS),
            "evaluator_sha256": sha256(EVALUATOR),
            "checkpoints": checkpoints,
        },
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
