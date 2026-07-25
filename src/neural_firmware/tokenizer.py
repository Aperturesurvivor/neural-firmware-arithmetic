from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ArithmeticTokenizer:
    """A deliberately transparent character tokenizer for the controlled grammar."""

    tokens: tuple[str, ...] = (
        "<pad>",
        "<bos>",
        "<eos>",
        "0",
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "+",
        "=",
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "token_to_id", {token: i for i, token in enumerate(self.tokens)})

    @property
    def vocab_size(self) -> int:
        return len(self.tokens)

    @property
    def pad_id(self) -> int:
        return self.token_to_id["<pad>"]

    @property
    def bos_id(self) -> int:
        return self.token_to_id["<bos>"]

    @property
    def eos_id(self) -> int:
        return self.token_to_id["<eos>"]

    @property
    def plus_id(self) -> int:
        return self.token_to_id["+"]

    @property
    def equals_id(self) -> int:
        return self.token_to_id["="]

    @property
    def zero_id(self) -> int:
        return self.token_to_id["0"]

    def digit_id(self, digit: int) -> int:
        if not 0 <= digit <= 9:
            raise ValueError(f"Not a decimal digit: {digit}")
        return self.zero_id + digit

    def id_to_digit(self, token_id: int) -> int:
        digit = token_id - self.zero_id
        if not 0 <= digit <= 9:
            raise ValueError(f"Token is not a decimal digit: {token_id}")
        return digit

    def encode_expression(self, a: int, b: int, *, include_answer: bool = True) -> list[int]:
        prompt = [self.bos_id]
        prompt.extend(self.digit_id(int(char)) for char in str(a))
        prompt.append(self.plus_id)
        prompt.extend(self.digit_id(int(char)) for char in str(b))
        prompt.append(self.equals_id)
        if include_answer:
            prompt.extend(self.digit_id(int(char)) for char in str(a + b))
            prompt.append(self.eos_id)
        return prompt

    def decode_answer(self, token_ids: list[int]) -> str:
        chars: list[str] = []
        for token_id in token_ids:
            if token_id == self.eos_id:
                break
            if self.zero_id <= token_id <= self.digit_id(9):
                chars.append(str(self.id_to_digit(token_id)))
            else:
                chars.append(f"<{self.tokens[token_id]}>")
        return "".join(chars)

