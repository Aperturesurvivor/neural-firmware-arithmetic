from __future__ import annotations

from neural_firmware.phase5_data import (
    PHASE5_CONFIRMATORY_ADDITION_FAMILIES,
    PHASE5_CONFIRMATORY_NEGATIVE_FAMILIES,
    PHASE5_CONFIRMATORY_WORD_FAMILIES,
    PHASE5_TRAIN_ADDITION_FAMILIES,
    PHASE5_TRAIN_NEGATIVE_FAMILIES,
    build_phase5_confirmatory_negatives,
    build_phase5_confirmatory_positive_sets,
)


def test_confirmatory_families_are_disjoint_from_training() -> None:
    assert set(PHASE5_CONFIRMATORY_ADDITION_FAMILIES).isdisjoint(
        PHASE5_TRAIN_ADDITION_FAMILIES
    )
    assert set(PHASE5_CONFIRMATORY_WORD_FAMILIES).isdisjoint(
        PHASE5_TRAIN_ADDITION_FAMILIES
    )
    assert set(PHASE5_CONFIRMATORY_NEGATIVE_FAMILIES).isdisjoint(
        PHASE5_TRAIN_NEGATIVE_FAMILIES
    )


def test_confirmatory_data_is_deterministic_and_well_formed() -> None:
    first = build_phase5_confirmatory_positive_sets(count_per_split=3)
    second = build_phase5_confirmatory_positive_sets(count_per_split=3)
    assert first == second
    assert set(first) == {"id_1_4", "ood_5_8", "long_9_12", "word_5_8"}
    assert all(example.route_label for rows in first.values() for example in rows)
    negatives = build_phase5_confirmatory_negatives(count=7)
    assert all(not example.route_label for example in negatives)
