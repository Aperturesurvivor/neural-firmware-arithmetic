from scripts.verify_phase12_confirmation import (
    recompute_gates,
    recompute_metrics,
)


def _record(
    *,
    exact: bool,
    route: bool,
    preserved: bool,
) -> dict[str, object]:
    return {
        "format_exact": exact,
        "first_route": route,
        "first_route_active": route,
        "operands_exact": exact,
        "trajectory_exact": exact,
        "token_preserved": preserved,
        "oracle_route": {"format_exact": exact},
        "route_off": {"format_exact": False},
    }


def test_recompute_metrics_uses_rows_as_authority() -> None:
    rows = [
        {
            "route_label": True,
            "conditions": {
                "phase12_candidate": {"16201": _record(
                    exact=True,
                    route=True,
                    preserved=False,
                )}
            },
        },
        {
            "route_label": False,
            "conditions": {
                "phase12_candidate": {"16201": _record(
                    exact=False,
                    route=False,
                    preserved=True,
                )}
            },
        },
    ]
    metrics = recompute_metrics(rows, "phase12_candidate", 16_201)
    assert metrics["exact"] == 1
    assert metrics["positive_routes"] == 1
    assert metrics["false_routes"] == 0
    assert metrics["token_preserved"] == 1
    assert metrics["oracle_exact"] == 1
    assert metrics["paired_route_off_losses"] == 1


def test_recompute_gates_applies_frozen_thresholds() -> None:
    candidate = {
        str(seed): {
            "exact": 85,
            "positive_routes": 90,
            "false_routes": 4,
            "token_preserved": 196,
            "oracle_exact": 90,
            "conditional_exact": 80,
            "conditional_examples": 80,
            "conditional_trajectories_exact": 80,
            "paired_route_off_losses": 85,
            "route_off_exact": 0,
        }
        for seed in (16_201, 16_202, 16_203)
    }
    control = {
        str(seed): {"exact": 70}
        for seed in (16_201, 16_202, 16_203)
    }
    checkpoints = {
        str(seed): {
            "phase12_candidate": {"request_router_parameters": 131_104},
            "inheritance": {
                "all_inherited_tensors_bit_identical": True
            },
        }
        for seed in (16_201, 16_202, 16_203)
    }
    gates, gains = recompute_gates(
        {
            "phase11_control": control,
            "phase12_candidate": candidate,
        },
        checkpoints,
    )
    assert gains == {"16201": 15, "16202": 15, "16203": 15}
    assert all(gates.values())
