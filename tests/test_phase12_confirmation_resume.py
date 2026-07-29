import sys
from pathlib import Path

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))

import evaluate_phase12_confirmation as confirmation  # noqa: E402


class FakeExample:
    def __init__(self, prompt: str, route_label: bool) -> None:
        self.value = {
            "prompt": prompt,
            "route_label": route_label,
            "a": "1000",
            "b": "2000",
            "answer": "3000",
            "split": "test",
            "family_index": 0,
        }

    def to_dict(self) -> dict[str, object]:
        return dict(self.value)


def test_confirmation_progress_round_trip_and_identity_checks(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"canonical_rows_sha256":"canonical"}\n')
    progress = tmp_path / "progress.json"
    monkeypatch.setattr(confirmation, "MANIFEST_PATH", manifest)
    monkeypatch.setattr(confirmation, "PROGRESS_PATH", progress)
    monkeypatch.setattr(confirmation, "evaluator_sha256", lambda: "code")
    examples = [FakeExample("one", True), FakeExample("two", False)]
    rows, accumulated = confirmation.load_or_initialize_rows(
        examples,
        manifest_sha256="manifest",
        canonical_rows_sha256="canonical",
    )
    assert accumulated == 0.0
    rows[0]["base"] = {"latency_seconds": 1.0}
    confirmation.save_progress(
        rows,
        manifest_sha256="manifest",
        accumulated_before=3.0,
        invocation_started=confirmation.time.perf_counter(),
    )
    loaded, loaded_accumulated = confirmation.load_or_initialize_rows(
        examples,
        manifest_sha256="manifest",
        canonical_rows_sha256="canonical",
    )
    assert loaded[0]["base"] == {"latency_seconds": 1.0}
    assert loaded_accumulated >= 3.0
    with pytest.raises(ValueError, match="manifest_sha256"):
        confirmation.load_or_initialize_rows(
            examples,
            manifest_sha256="different",
            canonical_rows_sha256="canonical",
        )


def test_confirmation_progress_must_be_contiguous() -> None:
    rows = [
        {"base": {"done": True}, "conditions": {"candidate": {}}},
        {"base": None, "conditions": {"candidate": {}}},
        {"base": {"done": True}, "conditions": {"candidate": {}}},
    ]
    with pytest.raises(ValueError, match="non-contiguous"):
        confirmation.validate_contiguous_progress(rows, start=1)
