from __future__ import annotations

from collections import Counter

from neural_firmware.phase8_data import operand_pairs
from neural_firmware.phase9_data import (
    PHASE9_CONFIRMATORY_NEGATIVE_FAMILIES,
    PHASE9_CONFIRMATORY_POSITIVE_FAMILIES,
    PHASE9_DEVELOPMENT_NEGATIVE_FAMILIES,
    PHASE9_DEVELOPMENT_POSITIVE_FAMILIES,
    PHASE9_HARD_NEGATIVE_FAMILIES,
    PHASE9_HARD_POSITIVE_FAMILIES,
)
from neural_firmware.phase10_data import (
    PHASE10_SOURCE_SEEDS,
    PHASE10_TRAINING_SEEDS,
    build_phase10_confirmatory_examples,
    phase10_family_set,
)


def test_phase10_seed_mapping_is_fixed() -> None:
    assert PHASE10_TRAINING_SEEDS == (16_201, 16_202, 16_203)
    assert PHASE10_SOURCE_SEEDS == {
        16_201: 14_201,
        16_202: 14_202,
        16_203: 14_203,
    }


def test_phase10_confirmation_is_balanced_and_unique() -> None:
    examples = build_phase10_confirmatory_examples()
    positives = [example for example in examples if example.route_label]
    negatives = [example for example in examples if not example.route_label]
    assert len(examples) == 300
    assert len(positives) == 100
    assert len(negatives) == 200
    assert len({example.prompt for example in examples}) == 300
    assert len(operand_pairs(examples)) == 300
    assert Counter(example.split for example in positives) == {
        "phase10_confirmatory_positive_direct": 50,
        "phase10_confirmatory_positive_word": 25,
        "phase10_confirmatory_positive_distractor": 25,
    }
    assert set(Counter(example.split for example in negatives).values()) == {20}


def test_phase10_families_are_disjoint_from_phase9() -> None:
    prior = {
        family.template
        for family in (
            PHASE9_HARD_POSITIVE_FAMILIES
            + PHASE9_HARD_NEGATIVE_FAMILIES
            + PHASE9_DEVELOPMENT_POSITIVE_FAMILIES
            + PHASE9_DEVELOPMENT_NEGATIVE_FAMILIES
            + PHASE9_CONFIRMATORY_POSITIVE_FAMILIES
            + PHASE9_CONFIRMATORY_NEGATIVE_FAMILIES
        )
    }
    assert phase10_family_set().isdisjoint(prior)


def test_phase10_distractors_render_three_numbers() -> None:
    for example in build_phase10_confirmatory_examples():
        expected = 3 if example.split.endswith("distractor") else 2
        import re

        assert len(re.findall(r"(?<![0-9])[0-9]+(?![0-9])", example.prompt)) == expected
