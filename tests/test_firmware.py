import random

import torch

from neural_firmware.firmware import FrozenRippleCarryAdder
from neural_firmware.tokenizer import ArithmeticTokenizer


def _digit_ids(tokenizer: ArithmeticTokenizer, value: int) -> list[int]:
    return [tokenizer.digit_id(int(char)) for char in str(value)]


def test_ripple_carry_matches_python_for_random_integers() -> None:
    tokenizer = ArithmeticTokenizer()
    firmware = FrozenRippleCarryAdder(tokenizer, d_model=32)
    rng = random.Random(17)
    for _ in range(2_000):
        a = rng.randrange(0, 10**20)
        b = rng.randrange(0, 10**20)
        result = firmware.add_digit_ids(_digit_ids(tokenizer, a), _digit_ids(tokenizer, b))
        assert tokenizer.decode_answer(result) == str(a + b)


def test_next_token_targets_align_with_autoregressive_positions() -> None:
    tokenizer = ArithmeticTokenizer()
    firmware = FrozenRippleCarryAdder(tokenizer, d_model=32)
    full = tokenizer.encode_expression(999, 1)
    input_ids = torch.tensor([full[:-1]], dtype=torch.long)
    targets, valid = firmware.next_token_targets(input_ids)
    equals_index = full.index(tokenizer.equals_id)
    predicted = targets[0, valid[0]].tolist()
    assert predicted == [
        tokenizer.digit_id(1),
        tokenizer.digit_id(0),
        tokenizer.digit_id(0),
        tokenizer.digit_id(0),
        tokenizer.eos_id,
    ]
    assert valid[0, equals_index].item()


def test_firmware_has_no_trainable_parameters() -> None:
    firmware = FrozenRippleCarryAdder(ArithmeticTokenizer(), d_model=32)
    assert sum(parameter.numel() for parameter in firmware.parameters()) == 0

