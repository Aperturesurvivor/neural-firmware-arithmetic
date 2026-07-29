from scripts.verify_phase12_completion import expected_summary_markers


def test_expected_summary_markers_are_derived_from_raw_verdict() -> None:
    confirmation = {
        "gates": {"all_gates": True},
        "metrics": {
            "phase11_control": {
                seed: {"exact": 70}
                for seed in ("16201", "16202", "16203")
            },
            "phase12_candidate": {
                seed: {"exact": 85}
                for seed in ("16201", "16202", "16203")
            },
        },
        "paired_exact_gains": {
            seed: 15 for seed in ("16201", "16202", "16203")
        },
    }
    assert expected_summary_markers(confirmation) == [
        "Compound verdict | **Pass**",
        "seed 16,201: Phase 11 70/100, Phase 12 85/100 (gain +15)",
        "seed 16,202: Phase 11 70/100, Phase 12 85/100 (gain +15)",
        "seed 16,203: Phase 11 70/100, Phase 12 85/100 (gain +15)",
    ]
