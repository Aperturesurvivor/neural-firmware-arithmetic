from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from neural_firmware.data import AdditionExample
from neural_firmware.model import CausalArithmeticTransformer
from neural_firmware.tokenizer import ArithmeticTokenizer


@dataclass(frozen=True)
class Prediction:
    model: str
    seed: int
    split: str
    a: int
    b: int
    expected: str
    predicted: str
    correct: bool
    max_operand_digits: int


def _generate_bucket(
    model: CausalArithmeticTransformer,
    examples: list[AdditionExample],
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
    max_new_tokens: int,
) -> list[str]:
    prompts = [
        tokenizer.encode_expression(example.a, example.b, include_answer=False)
        for example in examples
    ]
    prompt_length = len(prompts[0])
    if any(len(prompt) != prompt_length for prompt in prompts):
        raise ValueError("Generation bucket contains unequal prompt lengths")
    input_ids = torch.tensor(prompts, dtype=torch.long, device=device)
    finished = torch.zeros(len(examples), dtype=torch.bool, device=device)
    generated: list[list[int]] = [[] for _ in examples]
    available_steps = model.config.max_sequence_length - prompt_length + 1
    max_new_tokens = min(max_new_tokens, available_steps)

    with torch.inference_mode():
        for _ in range(max_new_tokens):
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            logits = model(input_ids, attention_mask)
            next_ids = logits[:, -1, :].argmax(dim=-1)
            next_ids = torch.where(
                finished,
                torch.full_like(next_ids, tokenizer.eos_id),
                next_ids,
            )
            for row, token_id in enumerate(next_ids.detach().cpu().tolist()):
                if not finished[row]:
                    generated[row].append(token_id)
            finished = finished | (next_ids == tokenizer.eos_id)
            input_ids = torch.cat([input_ids, next_ids.unsqueeze(1)], dim=1)
            if bool(finished.all()):
                break
    return [tokenizer.decode_answer(token_ids) for token_ids in generated]


def evaluate_model(
    model: CausalArithmeticTransformer,
    seed: int,
    splits: dict[str, list[AdditionExample]],
    device: torch.device,
    batch_size: int = 128,
) -> tuple[list[Prediction], dict[str, dict[str, float | int]]]:
    tokenizer = model.tokenizer
    predictions: list[Prediction] = []

    for split, examples in splits.items():
        buckets: dict[tuple[int, int], list[AdditionExample]] = defaultdict(list)
        for example in examples:
            buckets[(len(str(example.a)), len(str(example.b)))].append(example)

        for bucket in buckets.values():
            for start in range(0, len(bucket), batch_size):
                chunk = bucket[start : start + batch_size]
                max_answer_digits = max(len(example.answer) for example in chunk)
                outputs = _generate_bucket(
                    model,
                    chunk,
                    tokenizer,
                    device,
                    max_new_tokens=max_answer_digits + 2,
                )
                for example, output in zip(chunk, outputs, strict=True):
                    predictions.append(
                        Prediction(
                            model=model.mode,
                            seed=seed,
                            split=split,
                            a=example.a,
                            b=example.b,
                            expected=example.answer,
                            predicted=output,
                            correct=output == example.answer,
                            max_operand_digits=example.max_operand_digits,
                        )
                    )

    metrics: dict[str, dict[str, float | int]] = {}
    for split in splits:
        split_predictions = [prediction for prediction in predictions if prediction.split == split]
        correct = sum(prediction.correct for prediction in split_predictions)
        metrics[split] = {
            "examples": len(split_predictions),
            "correct": correct,
            "exact_match_accuracy": correct / len(split_predictions),
        }
    return predictions, metrics


def save_evaluation(
    run_dir: Path,
    predictions: list[Prediction],
    metrics: dict[str, dict[str, float | int]],
    eval_hash: str,
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "evaluation.json").write_text(
        json.dumps({"eval_sha256": eval_hash, "metrics": metrics}, indent=2) + "\n"
    )
    with (run_dir / "predictions.jsonl").open("w") as handle:
        for prediction in predictions:
            handle.write(json.dumps(asdict(prediction), sort_keys=True) + "\n")
