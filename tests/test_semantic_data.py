from __future__ import annotations

from neural_firmware.semantic_data import (
    AUGMENTED_TRAIN_ADDITION_FAMILIES,
    AUGMENTED_TRAIN_NEGATIVE_FAMILIES,
    CONFIRMATORY_ADDITION_FAMILIES,
    CONFIRMATORY_NEGATIVE_FAMILIES,
    CONFIRMATORY_WORD_PROBLEM_FAMILIES,
    DEVELOPMENT_ADDITION_FAMILIES,
    DEVELOPMENT_NEGATIVE_FAMILIES,
    HELDOUT_ADDITION_FAMILIES,
    HELDOUT_NEGATIVE_FAMILIES,
    TRAIN_ADDITION_FAMILIES,
    TRAIN_NEGATIVE_FAMILIES,
    WORD_PROBLEM_FAMILIES,
    exact_format_correct,
    last_integer,
    locate_two_decimal_operands,
    make_semantic_addition_examples,
    make_semantic_routing_negatives,
    mathematical_correct,
)


def test_wording_families_are_disjoint() -> None:
    positive_sets = [
        set(TRAIN_ADDITION_FAMILIES),
        set(HELDOUT_ADDITION_FAMILIES),
        set(WORD_PROBLEM_FAMILIES),
        set(AUGMENTED_TRAIN_ADDITION_FAMILIES),
        set(DEVELOPMENT_ADDITION_FAMILIES),
        set(CONFIRMATORY_ADDITION_FAMILIES),
        set(CONFIRMATORY_WORD_PROBLEM_FAMILIES),
    ]
    negative_sets = [
        set(TRAIN_NEGATIVE_FAMILIES),
        set(HELDOUT_NEGATIVE_FAMILIES),
        set(AUGMENTED_TRAIN_NEGATIVE_FAMILIES),
        set(DEVELOPMENT_NEGATIVE_FAMILIES),
        set(CONFIRMATORY_NEGATIVE_FAMILIES),
    ]
    all_sets = positive_sets + negative_sets
    for index, left in enumerate(all_sets):
        for right in all_sets[index + 1 :]:
            assert left.isdisjoint(right)


def test_every_semantic_prompt_has_exactly_two_candidate_operands() -> None:
    positives = make_semantic_addition_examples(
        count=100,
        min_digits=1,
        max_digits=12,
        seed=17,
        split="test",
        families=(
            TRAIN_ADDITION_FAMILIES
            + HELDOUT_ADDITION_FAMILIES
            + WORD_PROBLEM_FAMILIES
            + AUGMENTED_TRAIN_ADDITION_FAMILIES
            + DEVELOPMENT_ADDITION_FAMILIES
            + CONFIRMATORY_ADDITION_FAMILIES
            + CONFIRMATORY_WORD_PROBLEM_FAMILIES
        ),
    )
    negatives = make_semantic_routing_negatives(
        count=100,
        min_digits=1,
        max_digits=12,
        seed=18,
        split="test_negative",
        families=(
            TRAIN_NEGATIVE_FAMILIES
            + HELDOUT_NEGATIVE_FAMILIES
            + AUGMENTED_TRAIN_NEGATIVE_FAMILIES
            + DEVELOPMENT_NEGATIVE_FAMILIES
            + CONFIRMATORY_NEGATIVE_FAMILIES
        ),
    )
    for example in positives + negatives:
        assert locate_two_decimal_operands(example.prompt) == (example.a, example.b)


def test_mathematical_scoring_allows_prose_but_uses_final_integer() -> None:
    assert last_integer("The answer is 1,234.") == "1234"
    assert mathematical_correct("Adding 10 and 20 gives 30.", "30")
    assert not mathematical_correct("I tried 30, but the answer is 31.", "30")
    assert exact_format_correct("30", "30")
    assert not exact_format_correct("The answer is 30.", "30")
