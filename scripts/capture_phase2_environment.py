from __future__ import annotations

import json
import platform
import subprocess
import sys
from importlib.metadata import version
from pathlib import Path

import torch
from huggingface_hub import model_info


def command_output(command: list[str]) -> str:
    return subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def main() -> None:
    output_path = Path("phase2_records/environment.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = model_info("Qwen/Qwen2.5-0.5B-Instruct")
    payload = {
        "captured_at": "2026-07-25",
        "model": {
            "repo_id": info.id,
            "revision_sha": info.sha,
            "library_name": info.library_name,
        },
        "platform": {
            "python": sys.version,
            "platform": platform.platform(),
            "macos": command_output(["sw_vers"]),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "unified_memory_bytes": int(command_output(["sysctl", "-n", "hw.memsize"])),
        },
        "packages": {
            name: version(name)
            for name in (
                "accelerate",
                "huggingface-hub",
                "numpy",
                "safetensors",
                "torch",
                "transformers",
            )
        },
        "torch": {
            "mps_available": torch.backends.mps.is_available(),
            "mps_built": torch.backends.mps.is_built(),
        },
        "git": {
            "head": command_output(["git", "rev-parse", "HEAD"]),
            "dirty": bool(command_output(["git", "status", "--porcelain"])),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2) + "\n")
    print(output_path)


if __name__ == "__main__":
    main()
