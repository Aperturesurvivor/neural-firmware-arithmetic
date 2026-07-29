from scripts.build_phase12_summary import render_summary


def _payload(*, passed: bool) -> tuple[dict[str, object], dict[str, object]]:
    seeds = ("16201", "16202", "16203")
    candidate = {
        seed: {
            "exact": 85,
            "positive_routes": 90,
            "false_routes": 2,
            "token_preserved": 198,
            "oracle_exact": 90,
        }
        for seed in seeds
    }
    control = {seed: {"exact": 70} for seed in seeds}
    gates = {
        "autonomous_exactness": passed,
        "paired_improvement": passed,
        "route_recognition": passed,
        "preservation": passed,
        "operand_access": passed,
        "conditional_mechanism": passed,
        "causal_routing": passed,
        "checkpoint_integrity": passed,
        "all_gates": passed,
    }
    confirmation = {
        "metrics": {
            "phase12_candidate": candidate,
            "phase11_control": control,
        },
        "paired_exact_gains": {seed: 15 for seed in seeds},
        "gates": gates,
    }
    categories = {
        name: {
            "positive": True,
            "prompts": prompts,
            "phase11_control": {"routes": 0, "exact": 1},
            "phase12_candidate": {"routes": 0, "exact": 2},
        }
        for name, prompts in (
            ("positive_direct", 50),
            ("positive_word", 25),
            ("positive_distractor", 25),
        )
    }
    categories["negative_multiplication"] = {
        "positive": False,
        "prompts": 20,
        "phase11_control": {"routes": 1, "exact": 0},
        "phase12_candidate": {"routes": 2, "exact": 0},
    }
    analysis = {
        "paired_phase12_vs_phase11": {
            "phase11_control_exact": 210,
            "phase12_candidate_exact": 255,
            "two_way_bootstrap": {
                "mean_paired_difference": 0.15,
                "percentile_95_lower": 0.05,
                "percentile_95_upper": 0.25,
            },
        },
        "category_summary": categories,
    }
    return confirmation, analysis


def test_summary_pass_verdict_is_data_driven() -> None:
    confirmation, analysis = _payload(passed=True)
    summary = render_summary(confirmation, analysis)
    assert "Compound verdict | **Pass**" in summary
    assert "No frozen gate failed." in summary


def test_summary_failure_names_failed_gates() -> None:
    confirmation, analysis = _payload(passed=False)
    summary = render_summary(confirmation, analysis)
    assert "Compound verdict | **Fail**" in summary
    assert "Failed gates: Autonomous exactness" in summary
