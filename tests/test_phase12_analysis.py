import numpy as np

from scripts.analyze_phase12_confirmation import (
    paired_matrix,
    probability_summary,
    two_way_bootstrap_interval,
)


def test_paired_matrix_uses_only_positive_rows() -> None:
    rows = [
        {
            "route_label": True,
            "conditions": {
                "phase11_control": {
                    str(seed): {"format_exact": False}
                    for seed in (16_201, 16_202, 16_203)
                },
                "phase12_candidate": {
                    str(seed): {"format_exact": seed != 16_202}
                    for seed in (16_201, 16_202, 16_203)
                },
            },
        },
        {
            "route_label": False,
            "conditions": {
                "phase11_control": {},
                "phase12_candidate": {},
            },
        },
    ]
    assert paired_matrix(rows).tolist() == [[1], [0], [1]]


def test_two_way_bootstrap_is_reproducible() -> None:
    differences = np.asarray([[1, 0, 1], [1, -1, 0]], dtype=np.int8)
    first = two_way_bootstrap_interval(differences, seed=7, draws=100)
    second = two_way_bootstrap_interval(differences, seed=7, draws=100)
    assert first == second
    assert first["mean_paired_difference"] == differences.mean()


def test_probability_summary_reports_bounds_and_median() -> None:
    assert probability_summary([0.1, 0.9, 0.5]) == {
        "count": 3,
        "minimum": 0.1,
        "median": 0.5,
        "maximum": 0.9,
    }
