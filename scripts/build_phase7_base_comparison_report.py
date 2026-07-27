from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent

RESULT_PATH = Path("phase7_results/base_vs_implant_audit5.json")
OUTPUT_DIRECTORY = Path("phase7_base_comparison")
ARTIFACT_PATH = OUTPUT_DIRECTORY / "artifact.json"
SEEDS = (13_201, 13_202, 13_203)
SOURCE_ID = "phase7_base_comparison"


def exact_label(record: dict[str, object]) -> str:
    prefix = "Exact" if record["format_correct"] else "Not exact"
    return f"{prefix}: {record['generated_text']}"


def extended_label(record: dict[str, object]) -> str:
    if record["format_correct"]:
        prefix = "Exact"
    elif record["mathematical_correct"]:
        prefix = "Last number correct; format wrong"
    else:
        prefix = "Wrong"
    return f"{prefix}: {record['generated_text']}"


def source() -> dict[str, object]:
    return {
        "id": SOURCE_ID,
        "label": "Phase 7 base-comparison dataset queries",
        "path": "phase7_base_comparison/source.sql",
    }


def main() -> None:
    result = json.loads(RESULT_PATH.read_text())
    if result["status"] != "complete":
        raise ValueError("base comparison evaluation is incomplete")
    rows = result["rows"]
    summary = result["summary"]
    positives = [row for row in rows if row["route_label"]]
    negatives = [row for row in rows if not row["route_label"]]
    base_format = summary["base"]["format_exact"]
    base_loose = summary["base"]["mathematical_exact"]
    base_extended = summary["base_extended_sensitivity"]["mathematical_exact"]
    implant_exact = summary["implants"]["pooled_mathematical_exact"]

    additions = []
    for row in positives:
        record = {
            "index": row["row_index"] + 1,
            "type": (
                "Direct"
                if row["split"] == "phase7_audit5_symbolic"
                else "Word problem"
            ),
            "prompt": row["prompt"],
            "expected": row["answer"],
            "base_8": exact_label(row["base"]),
            "base_64": extended_label(row["base_extended"]),
        }
        for seed in SEEDS:
            record[f"seed_{seed}"] = exact_label(
                row["implants"][str(seed)]
            )
        additions.append(record)

    negative_rows = [
        {
            "index": row["row_index"] + 1,
            "prompt": row["prompt"],
            "base_output": row["base"]["generated_text"],
            "implant_output": row["implants"]["13202"]["generated_text"],
            "all_three_token_identical": (
                "Yes"
                if all(
                    row["implants"][str(seed)]["token_identical_to_base"]
                    for seed in SEEDS
                )
                else "No"
            ),
        }
        for row in negatives
    ]

    primary_comparison = [
        {
            "condition": "Untouched Qwen",
            "accuracy": base_format / 60,
            "correct": base_format,
            "attempts": 60,
            "metric": "Exact requested response",
        },
        *[
            {
                "condition": f"Implant seed {seed:,}",
                "accuracy": summary["implants"]["per_seed"][str(seed)][
                    "format_exact"
                ]
                / 60,
                "correct": summary["implants"]["per_seed"][str(seed)][
                    "format_exact"
                ],
                "attempts": 60,
                "metric": "Exact requested response",
            }
            for seed in SEEDS
        ],
    ]
    sensitivity = [
        {
            "condition": "Base, 8-token last-number score",
            "accuracy": base_loose / 60,
            "correct": base_loose,
            "attempts": 60,
        },
        {
            "condition": "Base, 64-token last-number score",
            "accuracy": base_extended / 60,
            "correct": base_extended,
            "attempts": 60,
        },
        {
            "condition": "Implant, pooled exact response",
            "accuracy": implant_exact / 180,
            "correct": implant_exact,
            "attempts": 180,
        },
    ]
    headline = [
        {
            "base_exact": base_format / 60,
            "base_extended": base_extended / 60,
            "implant_exact": implant_exact / 180,
            "absolute_gain": implant_exact / 180 - base_format / 60,
            "negative_preservation": (
                summary["negative_preservation"][
                    "pooled_token_identical_to_base"
                ]
                / 180
            ),
        }
    ]

    generated_at = datetime.now(UTC).isoformat()
    report_source = source()
    artifact = {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "Calculator-Neuron Implant vs Untouched Qwen",
            "description": (
                "A paired problem-by-problem comparison on the complete frozen "
                "Phase 7 Audit 5 prompt set."
            ),
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "base_exact_card",
                    "description": (
                        "Untouched Qwen returned exactly the requested numeral "
                        "on one of sixty additions."
                    ),
                    "dataset": "headline",
                    "sourceId": SOURCE_ID,
                    "metrics": [
                        {
                            "label": "Base exact response",
                            "field": "base_exact",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "base_extended_card",
                    "description": (
                        "With 64 tokens, the final integer was correct on 27 "
                        "prompts, although only one output followed the requested format."
                    ),
                    "dataset": "headline",
                    "sourceId": SOURCE_ID,
                    "metrics": [
                        {
                            "label": "Base 64-token arithmetic recovery",
                            "field": "base_extended",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "implant_exact_card",
                    "description": (
                        "Exact numeral-only answers across all three learned "
                        "implant seeds."
                    ),
                    "dataset": "headline",
                    "sourceId": SOURCE_ID,
                    "metrics": [
                        {
                            "label": "Implant exact response",
                            "field": "implant_exact",
                            "format": "percent",
                        }
                    ],
                },
                {
                    "id": "preservation_card",
                    "description": (
                        "Every implanted negative-prompt output was token-identical "
                        "to untouched Qwen."
                    ),
                    "dataset": "headline",
                    "sourceId": SOURCE_ID,
                    "metrics": [
                        {
                            "label": "Negative preservation",
                            "field": "negative_preservation",
                            "format": "percent",
                        }
                    ],
                },
            ],
            "charts": [
                {
                    "id": "exact_accuracy_chart",
                    "title": "Exact response accuracy by model condition",
                    "subtitle": (
                        "Same 60 frozen additions, eight-token greedy generation"
                    ),
                    "showDescription": True,
                    "type": "horizontalBar",
                    "dataset": "primary_comparison",
                    "sourceId": SOURCE_ID,
                    "encodings": {
                        "x": {
                            "field": "condition",
                            "type": "nominal",
                        },
                        "y": {
                            "field": "accuracy",
                            "type": "quantitative",
                            "format": "percent",
                        },
                        "tooltip": [
                            {"field": "correct", "type": "quantitative"},
                            {"field": "attempts", "type": "quantitative"},
                            {"field": "metric", "type": "text"},
                        ],
                    },
                    "xAxisTitle": "Exact response accuracy",
                    "valueFormat": "percent",
                    "maxRows": 4,
                },
                {
                    "id": "sensitivity_chart",
                    "title": "Arithmetic recovery under token-budget sensitivity",
                    "subtitle": (
                        "Base last-number recovery is a looser metric; the implant "
                        "bar remains exact numeral-only output"
                    ),
                    "showDescription": True,
                    "type": "horizontalBar",
                    "dataset": "sensitivity",
                    "sourceId": SOURCE_ID,
                    "encodings": {
                        "x": {
                            "field": "condition",
                            "type": "nominal",
                        },
                        "y": {
                            "field": "accuracy",
                            "type": "quantitative",
                            "format": "percent",
                        },
                        "tooltip": [
                            {"field": "correct", "type": "quantitative"},
                            {"field": "attempts", "type": "quantitative"},
                        ],
                    },
                    "xAxisTitle": "Arithmetic recovery",
                    "valueFormat": "percent",
                    "maxRows": 3,
                },
            ],
            "tables": [
                {
                    "id": "addition_rows_table",
                    "title": "All 60 addition prompts and outputs",
                    "subtitle": (
                        "Exact means the full response is the requested numeral; "
                        "the 64-token base column separately labels loose last-number matches"
                    ),
                    "showDescription": True,
                    "dataset": "additions",
                    "sourceId": SOURCE_ID,
                    "density": "compact",
                    "defaultSort": {"field": "index", "direction": "asc"},
                    "columns": [
                        {"field": "index", "label": "#", "type": "number"},
                        {"field": "type", "label": "Type", "type": "text"},
                        {"field": "prompt", "label": "Prompt", "type": "text"},
                        {"field": "expected", "label": "Expected", "type": "text"},
                        {
                            "field": "base_8",
                            "label": "Untouched Qwen, 8 tokens",
                            "type": "text",
                        },
                        {
                            "field": "base_64",
                            "label": "Untouched Qwen, 64 tokens",
                            "type": "text",
                        },
                        {
                            "field": "seed_13201",
                            "label": "Implant seed 13,201",
                            "type": "text",
                        },
                        {
                            "field": "seed_13202",
                            "label": "Implant seed 13,202",
                            "type": "text",
                        },
                        {
                            "field": "seed_13203",
                            "label": "Implant seed 13,203",
                            "type": "text",
                        },
                    ],
                },
                {
                    "id": "negative_rows_table",
                    "title": "All 60 adversarial negative prompts",
                    "subtitle": (
                        "The displayed implant output is seed 13,202; all three "
                        "seeds matched untouched Qwen token-for-token"
                    ),
                    "showDescription": True,
                    "dataset": "negatives",
                    "sourceId": SOURCE_ID,
                    "density": "compact",
                    "defaultSort": {"field": "index", "direction": "asc"},
                    "columns": [
                        {"field": "index", "label": "#", "type": "number"},
                        {"field": "prompt", "label": "Prompt", "type": "text"},
                        {
                            "field": "base_output",
                            "label": "Untouched Qwen output",
                            "type": "text",
                        },
                        {
                            "field": "implant_output",
                            "label": "Implant output",
                            "type": "text",
                        },
                        {
                            "field": "all_three_token_identical",
                            "label": "All three preserved?",
                            "type": "text",
                        },
                    ],
                },
            ],
            "sources": [report_source],
            "blocks": [
                {
                    "id": "title",
                    "type": "markdown",
                    "body": "# Calculator-Neuron Implant vs Untouched Qwen",
                },
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": dedent(
                        f"""
                        ## Technical summary

                        **The implant is dramatically better on these exact prompts.**
                        Untouched Qwen returned the exact requested numeral on
                        **{base_format}/60 (1.67%)** additions. The three implant
                        seeds returned **57/60, 58/60, and 58/60**, or
                        **{implant_exact}/180 (96.11%)** pooled. That is a
                        **94.44-percentage-point exact-response gain**.

                        A generous 64-token base-only run recovered the correct sum
                        as its final integer on **{base_extended}/60 (45.0%)**, but
                        still followed the requested numeral-only format on just
                        **1/60**. The implant therefore improves both arithmetic
                        reliability and instruction-following, not formatting alone.
                        On all 60 adversarial non-addition prompts, every implant
                        seed remained token-identical to untouched Qwen.
                        """
                    ).strip(),
                },
                {
                    "id": "headline_metrics",
                    "type": "metric-strip",
                    "cardIds": [
                        "base_exact_card",
                        "base_extended_card",
                        "implant_exact_card",
                        "preservation_card",
                    ],
                },
                {
                    "id": "exact_result",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": dedent(
                        """
                        ## Exact requested answers separate the systems cleanly

                        Under the identical frozen evaluation—same prompts, Qwen
                        revision, chat formatting, greedy decoding, and eight-token
                        budget—untouched Qwen usually began an explanation or
                        restated an expression. The implant usually emitted the
                        requested numeral immediately. Seed 13,201 gained 56 exact
                        prompt-level wins over base with no base-only exact win;
                        seeds 13,202 and 13,203 each gained 57 with no base-only
                        exact win.
                        """
                    ).strip(),
                },
                {
                    "id": "exact_chart_block",
                    "type": "chart",
                    "chartId": "exact_accuracy_chart",
                },
                {
                    "id": "sensitivity_result",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": dedent(
                        """
                        ## Extra generation time helps base, but leaves a large gap

                        Eight tokens are exact experimental parity but can truncate
                        a verbose base response. Allowing untouched Qwen 64 tokens
                        raises loose last-number arithmetic recovery from 2/60 to
                        27/60. The implant remains 173/180 exact numeral-only
                        responses. On direct additions, extended base recovered
                        13/30 versus 27–28/30 for the implants; on word problems it
                        recovered 14/30 versus 30/30 for every implant seed. Seeds
                        13,202 and 13,203 solved all 27 prompts that extended base
                        solved plus 31 additional prompts, with no base-only win;
                        both systems missed the same two systematic operand-framing
                        prompts.
                        """
                    ).strip(),
                },
                {
                    "id": "sensitivity_chart_block",
                    "type": "chart",
                    "chartId": "sensitivity_chart",
                },
                {
                    "id": "addition_detail",
                    "type": "markdown",
                    "body": dedent(
                        """
                        ## Every addition output is available for inspection

                        The table preserves every prompt, expected sum, untouched
                        Qwen output at both token budgets, and all three implant
                        outputs. This makes systematic failures visible rather than
                        reducing the comparison to a single aggregate score.
                        """
                    ).strip(),
                },
                {
                    "id": "addition_table_block",
                    "type": "table",
                    "tableId": "addition_rows_table",
                },
                {
                    "id": "negative_result",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": dedent(
                        """
                        ## The gain did not alter the evaluated negative behavior

                        The negative set contains number-bearing but non-addition
                        instructions, quoted or canceled additions, other
                        operations, labels, and comparisons. All 180 seed-prompt
                        comparisons were token-identical to untouched Qwen. These
                        rows test preservation, not task correctness, because the
                        negative prompts do not have a single reference answer.
                        """
                    ).strip(),
                },
                {
                    "id": "negative_table_block",
                    "type": "table",
                    "tableId": "negative_rows_table",
                },
                {
                    "id": "scope_definitions",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": dedent(
                        """
                        ## Scope and metric definitions

                        The comparison population is the complete frozen Audit 5
                        set: 30 direct additions, 30 addition word problems, and
                        60 adversarial negatives. “Exact response” requires the
                        entire stripped output to equal the expected numeral.
                        “Last-number recovery” is deliberately looser: the final
                        integer appearing anywhere in the response must equal the
                        expected sum. Implant scores are reported separately by
                        seed and pooled across 180 seed-prompt attempts.
                        """
                    ).strip(),
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": dedent(
                        """
                        ## Paired methodology

                        Untouched Qwen used the exact frozen model revision and
                        generation path from Audit 5. For each base generation, the
                        layer-16 wrapper was temporarily removed, restoring the
                        original Qwen MLP. The primary comparison uses the audit's
                        eight-token greedy budget. The 64-token run is a base-only
                        sensitivity analysis. No prompts, outputs, or seeds were
                        discarded.
                        """
                    ).strip(),
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": SOURCE_ID,
                    "body": dedent(
                        """
                        ## Limitations and robustness boundary

                        This base benchmark is retrospective: the prompt set and
                        implant outputs were already frozen, but the explicit base
                        comparison was not preregistered. The eight-token condition
                        is controlled parity but penalizes verbose outputs; the
                        64-token sensitivity removes that truncation concern while
                        using a looser metric. The result covers one Qwen revision,
                        one-to-four-digit nonnegative addition, and the evaluated
                        prompt families. It does not establish arbitrary-language
                        robustness or multi-call reasoning.
                        """
                    ).strip(),
                },
                {
                    "id": "next_steps",
                    "type": "markdown",
                    "body": dedent(
                        """
                        ## Recommended next steps

                        1. Make this untouched-base row set a standard comparator in
                           future frozen audits.
                        2. Add a base condition with an explicitly optimized
                           arithmetic instruction while keeping the user-visible
                           prompts unchanged.
                        3. Compare latency and energy after implementing cached,
                           residual-native register state.
                        4. Repeat the paired benchmark on a second model family and
                           on bounded multi-call problems.
                        """
                    ).strip(),
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": dedent(
                        """
                        ## Further questions

                        How much of the remaining 3.89% implant error is removable
                        with a stronger neural operand interface? Can recurrent
                        calculator state remain inside reserved activations while
                        preserving the zero-false-route behavior? Does the same
                        relative gain hold after a base model is specifically
                        fine-tuned for concise arithmetic output?
                        """
                    ).strip(),
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": headline,
                "primary_comparison": primary_comparison,
                "sensitivity": sensitivity,
                "additions": additions,
                "negatives": negative_rows,
            },
            "accessIssues": [],
        },
        "sources": [report_source],
    }
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    ARTIFACT_PATH.write_text(json.dumps(artifact, indent=2) + "\n")
    print(ARTIFACT_PATH)


if __name__ == "__main__":
    main()
