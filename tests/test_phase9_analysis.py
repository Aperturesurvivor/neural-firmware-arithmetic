from scripts.analyze_phase9_confirmation import (
    canonical_rows_sha256,
    failure_stage,
    two_sided_sign_test,
)


def test_canonical_rows_hash_ignores_evaluation_fields() -> None:
    row = {
        "prompt": "Add 2 and 3.",
        "a": "2",
        "b": "3",
        "answer": "5",
        "route_label": True,
        "family": "Add {x} and {y}.",
        "family_index": 0,
        "split": "test",
    }
    enriched = {**row, "base": {"generated_text": "five"}}

    assert canonical_rows_sha256([row]) == canonical_rows_sha256([enriched])


def test_two_sided_sign_test_handles_ties_and_direction() -> None:
    assert two_sided_sign_test(0, 0) == 1.0
    assert two_sided_sign_test(1, 0) == 1.0
    assert two_sided_sign_test(9, 1) == two_sided_sign_test(1, 9)
    assert two_sided_sign_test(10, 0) < 0.01


def test_failure_stage_is_mutually_ordered() -> None:
    exact = {
        "format_exact": True,
        "first_route": True,
        "first_route_active": True,
        "operands_exact": True,
        "trajectory_exact": True,
    }
    assert failure_stage(exact) == "exact"
    assert failure_stage({**exact, "format_exact": False}) == (
        "downstream_decode"
    )
    assert failure_stage(
        {
            **exact,
            "format_exact": False,
            "trajectory_exact": False,
        }
    ) == "calculator_trajectory"
    assert failure_stage(
        {
            **exact,
            "format_exact": False,
            "operands_exact": False,
        }
    ) == "operand_content"
    assert failure_stage(
        {
            **exact,
            "format_exact": False,
            "first_route_active": False,
        }
    ) == "typed_handshake_inactive"
    assert failure_stage(
        {
            **exact,
            "format_exact": False,
            "first_route": False,
        }
    ) == "route_off"
