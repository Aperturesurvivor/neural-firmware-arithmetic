from __future__ import annotations

from neural_firmware.phase6_data import (
    PHASE6_AUDIT_CHAIN_FAMILIES,
    PHASE6_AUDIT_NEGATIVE_FAMILIES,
    PHASE6_AUDIT_SINGLE_FAMILIES,
    PHASE6_CALIBRATION_CHAIN_FAMILIES,
    PHASE6_CALIBRATION_NEGATIVE_FAMILIES,
    PHASE6_CALIBRATION_SINGLE_FAMILIES,
    PHASE6_CONFIRMATORY_CHAIN_FAMILIES,
    PHASE6_CONFIRMATORY_NEGATIVE_FAMILIES,
    PHASE6_CONFIRMATORY_SINGLE_FAMILIES,
    PHASE6_DEVELOPMENT_CHAIN_FAMILIES,
    PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES,
    PHASE6_DEVELOPMENT_SINGLE_FAMILIES,
    PHASE6_GATE_CHAIN_FAMILIES,
    PHASE6_GATE_NEGATIVE_FAMILIES,
    PHASE6_GATE_SINGLE_FAMILIES,
    PHASE6_STRESS_CHAIN_FAMILIES,
    PHASE6_STRESS_NEGATIVE_FAMILIES,
    PHASE6_STRESS_SINGLE_FAMILIES,
    PHASE6_TRAIN_CHAIN_FAMILIES,
    PHASE6_TRAIN_NEGATIVE_FAMILIES,
    PHASE6_TRAIN_SINGLE_FAMILIES,
    PHASE6_VALIDATION_CHAIN_FAMILIES,
    PHASE6_VALIDATION_NEGATIVE_FAMILIES,
    PHASE6_VALIDATION_SINGLE_FAMILIES,
    build_phase6_audit_examples,
    build_phase6_calibration_examples,
    build_phase6_development_examples,
    build_phase6_gate_examples,
    build_phase6_stress_examples,
    build_phase6_training_examples,
    build_phase6_validation_examples,
)


def test_phase6_family_partitions_are_disjoint() -> None:
    positive_train = set(PHASE6_TRAIN_SINGLE_FAMILIES + PHASE6_TRAIN_CHAIN_FAMILIES)
    positive_development = set(
        PHASE6_DEVELOPMENT_SINGLE_FAMILIES
        + PHASE6_DEVELOPMENT_CHAIN_FAMILIES
    )
    positive_confirmation = set(
        PHASE6_CONFIRMATORY_SINGLE_FAMILIES
        + PHASE6_CONFIRMATORY_CHAIN_FAMILIES
    )
    positive_calibration = set(
        PHASE6_CALIBRATION_SINGLE_FAMILIES
        + PHASE6_CALIBRATION_CHAIN_FAMILIES
    )
    positive_validation = set(
        PHASE6_VALIDATION_SINGLE_FAMILIES
        + PHASE6_VALIDATION_CHAIN_FAMILIES
    )
    positive_audit = set(
        PHASE6_AUDIT_SINGLE_FAMILIES + PHASE6_AUDIT_CHAIN_FAMILIES
    )
    positive_stress = set(
        PHASE6_STRESS_SINGLE_FAMILIES + PHASE6_STRESS_CHAIN_FAMILIES
    )
    positive_gate = set(
        PHASE6_GATE_SINGLE_FAMILIES + PHASE6_GATE_CHAIN_FAMILIES
    )
    assert positive_train.isdisjoint(positive_development)
    assert positive_train.isdisjoint(positive_confirmation)
    assert positive_development.isdisjoint(positive_confirmation)
    assert positive_train.isdisjoint(positive_calibration)
    assert positive_train.isdisjoint(positive_validation)
    assert positive_calibration.isdisjoint(positive_validation)
    assert positive_calibration.isdisjoint(positive_confirmation)
    assert positive_validation.isdisjoint(positive_confirmation)
    for prior in (
        positive_train,
        positive_development,
        positive_calibration,
        positive_validation,
        positive_confirmation,
    ):
        assert positive_audit.isdisjoint(prior)
        assert positive_stress.isdisjoint(prior)
        assert positive_gate.isdisjoint(prior)
    assert positive_stress.isdisjoint(positive_audit)
    assert positive_gate.isdisjoint(positive_audit)
    assert positive_gate.isdisjoint(positive_stress)
    assert set(PHASE6_TRAIN_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES
    )
    assert set(PHASE6_TRAIN_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_CONFIRMATORY_NEGATIVE_FAMILIES
    )
    assert set(PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_CONFIRMATORY_NEGATIVE_FAMILIES
    )
    assert set(PHASE6_CALIBRATION_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_VALIDATION_NEGATIVE_FAMILIES
    )
    assert set(PHASE6_CALIBRATION_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_CONFIRMATORY_NEGATIVE_FAMILIES
    )
    assert set(PHASE6_VALIDATION_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_CONFIRMATORY_NEGATIVE_FAMILIES
    )
    for prior in (
        PHASE6_TRAIN_NEGATIVE_FAMILIES,
        PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES,
        PHASE6_CALIBRATION_NEGATIVE_FAMILIES,
        PHASE6_VALIDATION_NEGATIVE_FAMILIES,
        PHASE6_CONFIRMATORY_NEGATIVE_FAMILIES,
    ):
        assert set(PHASE6_AUDIT_NEGATIVE_FAMILIES).isdisjoint(prior)
        assert set(PHASE6_STRESS_NEGATIVE_FAMILIES).isdisjoint(prior)
        assert set(PHASE6_GATE_NEGATIVE_FAMILIES).isdisjoint(prior)
    assert set(PHASE6_STRESS_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_AUDIT_NEGATIVE_FAMILIES
    )
    assert set(PHASE6_GATE_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_AUDIT_NEGATIVE_FAMILIES
    )
    assert set(PHASE6_GATE_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_STRESS_NEGATIVE_FAMILIES
    )


def test_phase6_examples_are_deterministic_and_label_calls() -> None:
    first = build_phase6_training_examples(
        single_count=3,
        chain_count=3,
        negative_count=4,
    )
    second = build_phase6_training_examples(
        single_count=3,
        chain_count=3,
        negative_count=4,
    )
    assert first == second
    assert [example.call_count for example in first] == [1] * 3 + [2] * 3 + [0] * 4
    assert all(len(example.intermediate_answers) == example.call_count for example in first)
    assert all(
        example.answer == str(sum(map(int, example.operands)))
        for example in first
        if example.route_label
    )
    assert {example.controller_target for example in first[-4:]} <= {0, 3, 4}
    development = build_phase6_development_examples(
        single_count=2,
        chain_count=2,
        negative_count=2,
    )
    assert [example.call_count for example in development] == [1, 1, 2, 2, 0, 0]


def test_operand_register_targets_follow_textual_order() -> None:
    examples = build_phase6_development_examples(
        single_count=2,
        chain_count=0,
        negative_count=0,
    )
    reversed_family = examples[1]
    assert "By adding" in reversed_family.prompt
    first_position = reversed_family.prompt.index(reversed_family.operands[0])
    second_position = reversed_family.prompt.index(reversed_family.operands[1])
    assert first_position < second_position


def test_calibration_and_validation_builders_are_separate() -> None:
    calibration = build_phase6_calibration_examples(
        single_count=2,
        chain_count=2,
        negative_count=3,
    )
    validation = build_phase6_validation_examples(
        single_count=2,
        chain_count=2,
        negative_count=3,
    )
    assert {example.split for example in calibration}.isdisjoint(
        example.split for example in validation
    )
    assert [example.call_count for example in calibration] == [1, 1, 2, 2, 0, 0, 0]


def test_audit_builder_is_separate_and_deterministic() -> None:
    first = build_phase6_audit_examples(
        single_count=2,
        chain_count=2,
        negative_count=3,
    )
    second = build_phase6_audit_examples(
        single_count=2,
        chain_count=2,
        negative_count=3,
    )
    assert first == second
    assert [example.call_count for example in first] == [1, 1, 2, 2, 0, 0, 0]
    assert all(example.split.startswith("phase6_audit_") for example in first)


def test_stress_builder_is_separate_and_deterministic() -> None:
    first = build_phase6_stress_examples(
        single_count=2,
        chain_count=2,
        negative_count=3,
    )
    assert first == build_phase6_stress_examples(
        single_count=2,
        chain_count=2,
        negative_count=3,
    )
    assert [example.call_count for example in first] == [1, 1, 2, 2, 0, 0, 0]
    assert all(example.split.startswith("phase6_stress_") for example in first)


def test_gate_builder_is_separate_and_deterministic() -> None:
    first = build_phase6_gate_examples(
        single_count=2,
        chain_count=2,
        negative_count=3,
    )
    assert first == build_phase6_gate_examples(
        single_count=2,
        chain_count=2,
        negative_count=3,
    )
    assert [example.call_count for example in first] == [1, 1, 2, 2, 0, 0, 0]
    assert all(example.split.startswith("phase6_gate_") for example in first)
