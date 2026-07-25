import numpy as np
import torch

from neural_firmware.data import (
    AdditionExample,
    collate_full_sequences,
    eval_splits_hash,
    make_eval_splits,
)
from neural_firmware.tokenizer import ArithmeticTokenizer


def test_tokenizer_round_trip_answer() -> None:
    tokenizer = ArithmeticTokenizer()
    sequence = tokenizer.encode_expression(483, 927)
    equals_index = sequence.index(tokenizer.equals_id)
    assert tokenizer.decode_answer(sequence[equals_index + 1 :]) == "1410"


def test_loss_mask_starts_after_equals() -> None:
    tokenizer = ArithmeticTokenizer()
    batch = collate_full_sequences(
        [AdditionExample(a=12, b=9, split="test")],
        tokenizer,
        torch.device("cpu"),
    )
    sequence = tokenizer.encode_expression(12, 9)
    equals_index = sequence.index(tokenizer.equals_id)
    assert torch.all(batch["labels"][0, :equals_index] == -100)
    assert batch["labels"][0, equals_index].item() == tokenizer.digit_id(2)


def test_eval_splits_are_reproducible() -> None:
    first = make_eval_splits(20, 10, seed=7)
    second = make_eval_splits(20, 10, seed=7)
    assert eval_splits_hash(first) == eval_splits_hash(second)
    assert len(first["carry_chain"]) == 10
    assert np.mean([example.max_operand_digits for example in first["ood_long"]]) >= 13

