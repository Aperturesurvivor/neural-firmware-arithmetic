from __future__ import annotations

from neural_firmware.phase7_data import (
    PHASE7_AUDIT_ADDITION_FAMILIES,
    PHASE7_AUDIT_NEGATIVE_FAMILIES,
    PHASE7_AUDIT_WORD_FAMILIES,
    build_phase7_audit_examples,
    phase7_audit_family_sets,
)


def test_phase7_audit_families_are_disjoint_from_prior_families() -> None:
    prior_positive, prior_negative = phase7_audit_family_sets()
    current_positive = set(
        PHASE7_AUDIT_ADDITION_FAMILIES + PHASE7_AUDIT_WORD_FAMILIES
    )
    assert current_positive.isdisjoint(prior_positive)
    assert set(PHASE7_AUDIT_NEGATIVE_FAMILIES).isdisjoint(prior_negative)
    assert current_positive.isdisjoint(PHASE7_AUDIT_NEGATIVE_FAMILIES)


def test_phase7_audit_data_is_deterministic_and_partitioned() -> None:
    first = build_phase7_audit_examples(
        symbolic_count=3,
        word_count=4,
        negative_count=5,
    )
    second = build_phase7_audit_examples(
        symbolic_count=3,
        word_count=4,
        negative_count=5,
    )
    assert first == second
    assert [row.split for row in first].count("phase7_audit_symbolic") == 3
    assert [row.split for row in first].count("phase7_audit_word") == 4
    assert [row.split for row in first].count("phase7_audit_negative") == 5
    assert all(row.route_label for row in first[:7])
    assert all(not row.route_label for row in first[7:])
