from __future__ import annotations

from neural_firmware.phase7_data import (
    PHASE7_AUDIT5_ADDITION_FAMILIES,
    PHASE7_AUDIT5_NEGATIVE_FAMILIES,
    PHASE7_AUDIT5_WORD_FAMILIES,
    phase7_audit5_prior_family_sets,
)
from neural_firmware.phase8_data import (
    PHASE8_CONFIRMATORY_DIRECT_FAMILIES,
    PHASE8_CONFIRMATORY_NEGATIVE_FAMILIES,
    PHASE8_CONFIRMATORY_WORD_FAMILIES,
    PHASE8_TRAINING_SEEDS,
    build_phase8_confirmatory_examples,
    build_phase8_training_and_development,
    operand_pairs,
)


def test_phase8_frozen_seed_count_is_three() -> None:
    assert PHASE8_TRAINING_SEEDS == (14_201, 14_202, 14_203)


def test_phase8_confirmatory_families_are_new() -> None:
    prior_positive, prior_negative = phase7_audit5_prior_family_sets()
    prior_positive.update(
        PHASE7_AUDIT5_ADDITION_FAMILIES + PHASE7_AUDIT5_WORD_FAMILIES
    )
    prior_negative.update(PHASE7_AUDIT5_NEGATIVE_FAMILIES)
    current_positive = set(
        PHASE8_CONFIRMATORY_DIRECT_FAMILIES
        + PHASE8_CONFIRMATORY_WORD_FAMILIES
    )
    current_negative = {
        family
        for families in PHASE8_CONFIRMATORY_NEGATIVE_FAMILIES.values()
        for family in families
    }
    assert current_positive.isdisjoint(prior_positive)
    assert current_negative.isdisjoint(prior_negative)
    assert current_positive.isdisjoint(current_negative)


def test_phase8_splits_are_deterministic_and_operand_disjoint() -> None:
    training, development = build_phase8_training_and_development()
    first = build_phase8_confirmatory_examples()
    second = build_phase8_confirmatory_examples()
    assert first == second
    assert len(training) == 2_400
    assert len(development) == 480
    assert len(first) == 120
    assert operand_pairs(training).isdisjoint(operand_pairs(development))
    assert operand_pairs(training + development).isdisjoint(operand_pairs(first))


def test_phase8_confirmatory_partition_is_balanced() -> None:
    examples = build_phase8_confirmatory_examples()
    assert sum(row.split == "phase8_confirmatory_direct" for row in examples) == 30
    assert sum(row.split == "phase8_confirmatory_word" for row in examples) == 30
    for category in PHASE8_CONFIRMATORY_NEGATIVE_FAMILIES:
        assert sum(
            row.split == f"phase8_confirmatory_negative_{category}"
            for row in examples
        ) == 12
    assert all(row.route_label for row in examples[:60])
    assert all(not row.route_label for row in examples[60:])
