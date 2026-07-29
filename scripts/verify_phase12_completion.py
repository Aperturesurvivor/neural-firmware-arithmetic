from __future__ import annotations

import hashlib
import json
from pathlib import Path

from neural_firmware.phase8_data import PHASE8_MODEL_ID, PHASE8_MODEL_REVISION

CORE_VERIFICATION = Path("phase12_results/verification.json")
CONFIRMATION = Path("phase12_results/confirmation.json")
ANALYSIS = Path("phase12_results/analysis.json")
ENVIRONMENT = Path("phase12_results/environment_audit.json")
SUMMARY = Path("PHASE12_EXECUTIVE_SUMMARY.md")
PROTOCOL = Path("PHASE12_MULTI_VIEW_ROUTING_PROTOCOL.md")
MANIFEST = Path("phase12_results/frozen_prompt_manifest.json")
DEVELOPMENT = Path("PHASE12_MULTI_VIEW_ROUTING_DEVELOPMENT.md")
AUDIT = Path("phase12_results/completion_audit.json")
UV_LOCK = Path("uv.lock")
PYPROJECT = Path("pyproject.toml")
SEEDS = (16_201, 16_202, 16_203)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check(name: str, value: bool, checks: list[dict[str, object]]) -> None:
    checks.append({"name": name, "passed": bool(value)})


def expected_summary_markers(
    confirmation: dict[str, object],
) -> list[str]:
    verdict = "Pass" if confirmation["gates"]["all_gates"] else "Fail"
    markers = [f"Compound verdict | **{verdict}**"]
    for seed in SEEDS:
        key = str(seed)
        control = confirmation["metrics"]["phase11_control"][key]["exact"]
        candidate = confirmation["metrics"]["phase12_candidate"][key][
            "exact"
        ]
        gain = confirmation["paired_exact_gains"][key]
        markers.append(
            f"seed {seed:,}: Phase 11 {control}/100, "
            f"Phase 12 {candidate}/100 (gain {gain:+d})"
        )
    return markers


def main() -> None:
    core = json.loads(CORE_VERIFICATION.read_text())
    confirmation = json.loads(CONFIRMATION.read_text())
    analysis = json.loads(ANALYSIS.read_text())
    environment = json.loads(ENVIRONMENT.read_text())
    summary = SUMMARY.read_text()
    manifest = json.loads(MANIFEST.read_text())
    checks: list[dict[str, object]] = []
    check(
        "core_verification",
        core["status"] == "phase12_confirmation_verification_passed"
        and core["failed"] == 0,
        checks,
    )
    check(
        "artifact_statuses",
        confirmation["status"] == "phase12_confirmation_complete"
        and analysis["status"] == "phase12_posthoc_analysis_complete"
        and environment["status"]
        == "phase12_environment_provenance_audited_posthoc"
        and manifest["status"]
        == "phase12_protocol_frozen_before_confirmation",
        checks,
    )
    check(
        "environment_model_and_dependency_provenance",
        environment["model"]["id"] == PHASE8_MODEL_ID
        and environment["model"]["revision"] == PHASE8_MODEL_REVISION
        and environment["dependency_provenance"]["uv_lock_sha256"]
        == sha256(UV_LOCK)
        and environment["dependency_provenance"]["pyproject_sha256"]
        == sha256(PYPROJECT),
        checks,
    )
    check(
        "environment_artifact_hashes",
        environment["artifacts"]["manifest_sha256"] == sha256(MANIFEST)
        and environment["artifacts"]["confirmation_sha256"]
        == sha256(CONFIRMATION)
        and environment["artifacts"]["analysis_sha256"] == sha256(ANALYSIS),
        checks,
    )
    check(
        "summary_is_data_derived",
        all(
            marker in summary
            for marker in expected_summary_markers(confirmation)
        ),
        checks,
    )
    check(
        "scope_limits_retained",
        "not general semantic understanding" in PROTOCOL.read_text()
        and "not general semantic understanding" in summary
        and "development evidence" in DEVELOPMENT.read_text(),
        checks,
    )
    check(
        "required_artifacts_nonempty",
        all(
            path.exists() and path.stat().st_size > 0
            for path in (
                CONFIRMATION,
                ANALYSIS,
                ENVIRONMENT,
                SUMMARY,
                PROTOCOL,
                MANIFEST,
                DEVELOPMENT,
            )
        ),
        checks,
    )
    failed = [item for item in checks if not item["passed"]]
    payload = {
        "status": (
            "phase12_completion_audit_passed"
            if not failed
            else "phase12_completion_audit_failed"
        ),
        "checks": checks,
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "failed_checks": [item["name"] for item in failed],
        "compound_verdict": confirmation["gates"]["all_gates"],
        "artifact_hashes": {
            str(path): sha256(path)
            for path in (
                CORE_VERIFICATION,
                CONFIRMATION,
                ANALYSIS,
                ENVIRONMENT,
                SUMMARY,
                PROTOCOL,
                MANIFEST,
                DEVELOPMENT,
            )
        },
    }
    AUDIT.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
