from neural_firmware.phase10_data import phase10_family_set
from neural_firmware.phase11_data import (
    build_phase11_confirmatory_examples,
    phase11_family_set,
)
from neural_firmware.phase12_data import (
    PHASE12_CONFIRMATORY_NEGATIVE_FAMILIES,
    PHASE12_CONFIRMATORY_POSITIVE_FAMILIES,
    _all_pairs_before_phase12,
    build_phase12_confirmatory_examples,
    phase12_family_set,
)
from neural_firmware.phase8_data import operand_pairs
from neural_firmware.phase9_data import (
    PHASE9_CONFIRMATORY_NEGATIVE_FAMILIES,
    PHASE9_CONFIRMATORY_POSITIVE_FAMILIES,
    PHASE9_DEVELOPMENT_NEGATIVE_FAMILIES,
    PHASE9_DEVELOPMENT_POSITIVE_FAMILIES,
    PHASE9_HARD_NEGATIVE_FAMILIES,
    PHASE9_HARD_POSITIVE_FAMILIES,
)


def test_phase12_confirmation_is_balanced_and_unique() -> None:
    examples = build_phase12_confirmatory_examples()
    assert len(examples) == 300
    assert sum(example.route_label for example in examples) == 100
    assert len({example.prompt for example in examples}) == 300
    assert len(PHASE12_CONFIRMATORY_POSITIVE_FAMILIES) == 20
    assert len(PHASE12_CONFIRMATORY_NEGATIVE_FAMILIES) == 50


def test_phase12_families_are_exact_string_disjoint() -> None:
    phase9 = {
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
    assert phase12_family_set().isdisjoint(phase9)
    assert phase12_family_set().isdisjoint(phase10_family_set())
    assert phase12_family_set().isdisjoint(phase11_family_set())


def test_phase12_operand_pairs_are_disjoint() -> None:
    examples = build_phase12_confirmatory_examples()
    assert operand_pairs(examples).isdisjoint(_all_pairs_before_phase12())
    assert operand_pairs(examples).isdisjoint(
        operand_pairs(build_phase11_confirmatory_examples())
    )
