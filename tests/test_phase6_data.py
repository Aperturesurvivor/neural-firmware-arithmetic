from __future__ import annotations

from neural_firmware.phase6_data import (
    PHASE6_CONFIRMATORY_CHAIN_FAMILIES,
    PHASE6_CONFIRMATORY_NEGATIVE_FAMILIES,
    PHASE6_CONFIRMATORY_SINGLE_FAMILIES,
    PHASE6_DEVELOPMENT_CHAIN_FAMILIES,
    PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES,
    PHASE6_DEVELOPMENT_SINGLE_FAMILIES,
    PHASE6_TRAIN_CHAIN_FAMILIES,
    PHASE6_TRAIN_NEGATIVE_FAMILIES,
    PHASE6_TRAIN_SINGLE_FAMILIES,
    build_phase6_development_examples,
    build_phase6_training_examples,
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
    assert positive_train.isdisjoint(positive_development)
    assert positive_train.isdisjoint(positive_confirmation)
    assert positive_development.isdisjoint(positive_confirmation)
    assert set(PHASE6_TRAIN_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES
    )
    assert set(PHASE6_TRAIN_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_CONFIRMATORY_NEGATIVE_FAMILIES
    )
    assert set(PHASE6_DEVELOPMENT_NEGATIVE_FAMILIES).isdisjoint(
        PHASE6_CONFIRMATORY_NEGATIVE_FAMILIES
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
