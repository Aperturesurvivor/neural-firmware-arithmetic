from collections import Counter

from neural_firmware.phase8_data import operand_pairs
from neural_firmware.phase9_data import phase9_family_sets
from neural_firmware.phase10_data import (
    build_phase10_confirmatory_examples,
    phase10_family_set,
)
from neural_firmware.phase11_data import (
    PHASE11_CONFIRMATORY_NEGATIVE_FAMILIES,
    PHASE11_CONFIRMATORY_POSITIVE_FAMILIES,
    _all_pairs_before_phase11,
    build_phase11_confirmatory_examples,
    phase11_family_set,
)


def test_phase11_confirmation_is_balanced_and_unique() -> None:
    rows = build_phase11_confirmatory_examples()
    positives = [row for row in rows if row.route_label]
    negatives = [row for row in rows if not row.route_label]
    assert len(rows) == 300
    assert len(positives) == 100
    assert len(negatives) == 200
    assert len({row.prompt for row in rows}) == len(rows)
    assert Counter(row.family for row in positives) == {
        family.template: 5
        for family in PHASE11_CONFIRMATORY_POSITIVE_FAMILIES
    }
    assert Counter(row.family for row in negatives) == {
        family.template: 4
        for family in PHASE11_CONFIRMATORY_NEGATIVE_FAMILIES
    }


def test_phase11_confirmation_is_disjoint_from_prior_phases() -> None:
    rows = build_phase11_confirmatory_examples()
    assert not (operand_pairs(rows) & _all_pairs_before_phase11())
    prior_families = phase10_family_set()
    for values in phase9_family_sets().values():
        prior_families.update(values)
    assert not (phase11_family_set() & prior_families)
    assert not (
        {row.prompt for row in rows}
        & {row.prompt for row in build_phase10_confirmatory_examples()}
    )
