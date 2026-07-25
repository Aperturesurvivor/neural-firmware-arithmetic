from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch

from neural_firmware.tokenizer import ArithmeticTokenizer


@dataclass(frozen=True)
class AdditionExample:
    a: int
    b: int
    split: str

    @property
    def answer(self) -> str:
        return str(self.a + self.b)

    @property
    def max_operand_digits(self) -> int:
        return max(len(str(self.a)), len(str(self.b)))


def _sample_integer(rng: np.random.Generator, digits: int) -> int:
    if digits == 1:
        return int(rng.integers(0, 10))
    leading = str(int(rng.integers(1, 10)))
    remainder = "".join(str(int(value)) for value in rng.integers(0, 10, size=digits - 1))
    return int(leading + remainder)


def sample_examples(
    rng: np.random.Generator,
    count: int,
    min_digits: int,
    max_digits: int,
    split: str,
) -> list[AdditionExample]:
    examples: list[AdditionExample] = []
    for _ in range(count):
        a_digits = int(rng.integers(min_digits, max_digits + 1))
        b_digits = int(rng.integers(min_digits, max_digits + 1))
        examples.append(
            AdditionExample(
                a=_sample_integer(rng, a_digits),
                b=_sample_integer(rng, b_digits),
                split=split,
            )
        )
    return examples


def make_carry_examples(count: int, seed: int = 240725) -> list[AdditionExample]:
    """Create unique cases whose least-significant suffix forces long carry propagation."""

    rng = np.random.default_rng(seed)
    examples: list[AdditionExample] = []
    seen: set[tuple[int, int]] = set()
    while len(examples) < count:
        total_digits = int(rng.integers(7, 21))
        carry_length = int(rng.integers(max(2, total_digits // 2), total_digits + 1))
        prefix_digits = total_digits - carry_length
        if prefix_digits:
            prefix = _sample_integer(rng, prefix_digits)
            a = prefix * (10**carry_length) + (10**carry_length - 1)
        else:
            a = 10**carry_length - 1
        b = int(rng.integers(1, 10))
        if (a, b) not in seen:
            seen.add((a, b))
            examples.append(AdditionExample(a=a, b=b, split="carry_chain"))
    return examples


def make_eval_splits(
    eval_examples: int,
    carry_examples: int,
    seed: int = 260725,
) -> dict[str, list[AdditionExample]]:
    rng = np.random.default_rng(seed)
    return {
        "id_random": sample_examples(rng, eval_examples, 1, 6, "id_random"),
        "ood_primary": sample_examples(rng, eval_examples, 7, 12, "ood_primary"),
        "ood_long": sample_examples(rng, eval_examples, 13, 20, "ood_long"),
        "carry_chain": make_carry_examples(carry_examples, seed + 1),
    }


def eval_splits_hash(splits: dict[str, list[AdditionExample]]) -> str:
    payload = {
        split: [asdict(example) for example in examples]
        for split, examples in sorted(splits.items())
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save_eval_splits(path: Path, splits: dict[str, list[AdditionExample]]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sha256": eval_splits_hash(splits),
        "splits": {
            split: [asdict(example) for example in examples]
            for split, examples in splits.items()
        },
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return payload["sha256"]


def load_eval_splits(path: Path) -> tuple[dict[str, list[AdditionExample]], str]:
    payload = json.loads(path.read_text())
    splits = {
        split: [AdditionExample(**row) for row in rows]
        for split, rows in payload["splits"].items()
    }
    actual_hash = eval_splits_hash(splits)
    if actual_hash != payload["sha256"]:
        raise ValueError(f"Evaluation set hash mismatch: {actual_hash} != {payload['sha256']}")
    return splits, actual_hash


def collate_full_sequences(
    examples: Iterable[AdditionExample],
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    sequences = [tokenizer.encode_expression(example.a, example.b) for example in examples]
    max_length = max(len(sequence) for sequence in sequences)
    batch_size = len(sequences)

    input_ids = torch.full(
        (batch_size, max_length - 1), tokenizer.pad_id, dtype=torch.long, device=device
    )
    labels = torch.full_like(input_ids, -100)
    attention_mask = torch.zeros_like(input_ids, dtype=torch.bool)

    for row, sequence in enumerate(sequences):
        input_sequence = sequence[:-1]
        target_sequence = sequence[1:]
        length = len(input_sequence)
        input_ids[row, :length] = torch.tensor(input_sequence, dtype=torch.long, device=device)
        attention_mask[row, :length] = True
        equals_index = sequence.index(tokenizer.equals_id)
        labels[row, equals_index:length] = torch.tensor(
            target_sequence[equals_index:length], dtype=torch.long, device=device
        )

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }


def sample_training_batch(
    rng: np.random.Generator,
    batch_size: int,
    min_digits: int,
    max_digits: int,
    tokenizer: ArithmeticTokenizer,
    device: torch.device,
) -> tuple[list[AdditionExample], dict[str, torch.Tensor]]:
    examples = sample_examples(
        rng,
        batch_size,
        min_digits,
        max_digits,
        split="train",
    )
    return examples, collate_full_sequences(examples, tokenizer, device)
