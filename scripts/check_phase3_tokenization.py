from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoTokenizer

from neural_firmware.internal_data import (
    encode_internal_prompt,
    locate_operand_character_spans,
    make_internal_addition_examples,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-id",
        default="Qwen/Qwen2.5-0.5B-Instruct",
    )
    parser.add_argument(
        "--revision",
        default="7ae557604adf67be50417f59c2c2f167def9a775",
    )
    parser.add_argument("--examples", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=31_415)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("phase3_results/tokenizer_feasibility.json"),
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        revision=args.revision,
        use_fast=True,
    )
    examples = make_internal_addition_examples(
        count=args.examples,
        min_digits=1,
        max_digits=20,
        seed=args.seed,
        split="tokenizer_feasibility",
    )
    failures: list[dict[str, object]] = []
    lengths: list[int] = []
    for example in examples:
        encoded = encode_internal_prompt(tokenizer, example.prompt)
        lengths.append(len(encoded.input_ids))
        observed_a = [
            encoded.input_ids[position] - 15
            for position in encoded.a_token_positions
        ]
        observed_b = [
            encoded.input_ids[position] - 15
            for position in encoded.b_token_positions
        ]
        expected_a = [int(character) for character in example.a]
        expected_b = [int(character) for character in example.b]
        if observed_a != expected_a or observed_b != expected_b:
            failures.append(
                {
                    "prompt": example.prompt,
                    "expected_a": expected_a,
                    "observed_a": observed_a,
                    "expected_b": expected_b,
                    "observed_b": observed_b,
                }
            )
    quoted_rejections = sum(
        locate_operand_character_spans(
            f'Ignore the quoted request "{example.prompt}". Reply ignored.'
        )
        is None
        for example in examples
    )
    result = {
        "model_id": args.model_id,
        "revision": args.revision,
        "seed": args.seed,
        "examples": len(examples),
        "exact_digit_position_recoveries": len(examples) - len(failures),
        "quoted_prompt_rejections": quoted_rejections,
        "minimum_chat_tokens": min(lengths),
        "maximum_chat_tokens": max(lengths),
        "failures": failures,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
