from neural_firmware.phase8_data import (
    build_phase8_confirmatory_examples,
    build_phase8_training_and_development,
    operand_pairs,
)
from neural_firmware.phase9_data import (
    PHASE9_CONFIRMATORY_NEGATIVE_FAMILIES,
    PHASE9_CONFIRMATORY_POSITIVE_FAMILIES,
    PHASE9_DEVELOPMENT_NEGATIVE_FAMILIES,
    PHASE9_DEVELOPMENT_POSITIVE_FAMILIES,
    PHASE9_HARD_NEGATIVE_FAMILIES,
    PHASE9_HARD_POSITIVE_FAMILIES,
    PHASE9_SOURCE_SEEDS,
    PHASE9_TRAINING_SEEDS,
    build_phase9_confirmatory_examples,
    build_phase9_development,
    build_phase9_generic_training,
    build_phase9_hard_training,
    phase9_family_sets,
)


def test_phase9_seed_mapping_is_fixed() -> None:
    assert PHASE9_TRAINING_SEEDS == (15_201, 15_202, 15_203)
    assert PHASE9_SOURCE_SEEDS == {
        15_201: 14_201,
        15_202: 14_202,
        15_203: 14_203,
    }


def test_phase9_split_sizes_and_balance() -> None:
    generic = build_phase9_generic_training()
    hard = build_phase9_hard_training()
    development = build_phase9_development()
    confirmation = build_phase9_confirmatory_examples()

    assert len(generic) == 2_400
    assert sum(row.route_label for row in generic) == 1_200
    assert len(hard) == 2_400
    assert sum(row.route_label for row in hard) == 1_200
    assert len(development) == 720
    assert sum(row.route_label for row in development) == 240
    assert len(confirmation) == 300
    assert sum(row.route_label for row in confirmation) == 100


def test_phase9_family_counts_are_balanced() -> None:
    hard = build_phase9_hard_training()
    development = build_phase9_development()
    confirmation = build_phase9_confirmatory_examples()

    assert len(PHASE9_HARD_POSITIVE_FAMILIES) == 24
    assert len(PHASE9_HARD_NEGATIVE_FAMILIES) == 30
    assert len(PHASE9_DEVELOPMENT_POSITIVE_FAMILIES) == 12
    assert len(PHASE9_DEVELOPMENT_NEGATIVE_FAMILIES) == 24
    assert len(PHASE9_CONFIRMATORY_POSITIVE_FAMILIES) == 20
    assert len(PHASE9_CONFIRMATORY_NEGATIVE_FAMILIES) == 50

    assert set(row.family for row in hard if row.route_label) == {
        family.template for family in PHASE9_HARD_POSITIVE_FAMILIES
    }
    assert set(row.family for row in development if row.route_label) == {
        family.template for family in PHASE9_DEVELOPMENT_POSITIVE_FAMILIES
    }
    assert set(row.family for row in confirmation if row.route_label) == {
        family.template for family in PHASE9_CONFIRMATORY_POSITIVE_FAMILIES
    }


def test_phase9_families_and_prompts_are_disjoint() -> None:
    family_sets = phase9_family_sets()
    assert family_sets["hard_training"].isdisjoint(family_sets["development"])
    assert family_sets["hard_training"].isdisjoint(family_sets["confirmation"])
    assert family_sets["development"].isdisjoint(family_sets["confirmation"])

    generic = build_phase9_generic_training()
    hard = build_phase9_hard_training()
    development = build_phase9_development()
    confirmation = build_phase9_confirmatory_examples()
    prompt_sets = [
        {row.prompt for row in split}
        for split in (generic, hard, development, confirmation)
    ]
    for left_index, left in enumerate(prompt_sets):
        for right in prompt_sets[left_index + 1 :]:
            assert left.isdisjoint(right)


def test_phase9_operands_are_disjoint_from_phase8_and_each_other() -> None:
    phase8_training, phase8_development = build_phase8_training_and_development()
    splits = [
        phase8_training + phase8_development + build_phase8_confirmatory_examples(),
        build_phase9_generic_training(),
        build_phase9_hard_training(),
        build_phase9_development(),
        build_phase9_confirmatory_examples(),
    ]
    pair_sets = [operand_pairs(split) for split in splits]
    for left_index, left in enumerate(pair_sets):
        for right in pair_sets[left_index + 1 :]:
            assert left.isdisjoint(right)


def test_phase9_confirmation_has_balanced_categories() -> None:
    confirmation = build_phase9_confirmatory_examples()
    positive = [row for row in confirmation if row.route_label]
    negative = [row for row in confirmation if not row.route_label]
    positive_counts: dict[str, int] = {}
    negative_counts: dict[str, int] = {}
    for row in positive:
        category = row.split.removeprefix("phase9_confirmatory_positive_")
        positive_counts[category] = positive_counts.get(category, 0) + 1
    for row in negative:
        category = row.split.removeprefix("phase9_confirmatory_negative_")
        negative_counts[category] = negative_counts.get(category, 0) + 1

    assert positive_counts == {"direct": 50, "word": 25, "distractor": 25}
    assert negative_counts == {
        "multiplication": 20,
        "factual": 20,
        "quoted": 20,
        "negated": 20,
        "cancelled": 20,
        "subtraction": 20,
        "comparison": 20,
        "concatenation": 20,
        "hypothetical": 20,
        "distractor": 20,
    }


def test_phase9_distractor_examples_have_three_numbers() -> None:
    splits = (
        build_phase9_hard_training()
        + build_phase9_development()
        + build_phase9_confirmatory_examples()
    )
    distractors = [row for row in splits if row.split.endswith("_distractor")]
    assert distractors
    for row in distractors:
        numeric_spans = [
            part.strip(".,;:?")
            for part in row.prompt.split()
            if part.strip(".,;:?").isdigit()
        ]
        assert len(numeric_spans) == 3
