from __future__ import annotations

import re
from dataclasses import dataclass

SUM_TABLE = tuple(
    tuple((a + b + carry) % 10 for carry in range(2))
    for b in range(10)
    for a in range(10)
)
CARRY_TABLE = tuple(
    tuple((a + b + carry) // 10 for carry in range(2))
    for b in range(10)
    for a in range(10)
)

_PATTERNS = (
    re.compile(r"Calculate exactly\..*?(\d+)\s*\+\s*(\d+)\s*=", re.DOTALL),
    re.compile(r"What is\s+(\d+)\s+plus\s+(\d+)\?", re.IGNORECASE),
    re.compile(r"Add\s+(\d+)\s+and\s+(\d+)\.", re.IGNORECASE),
)


def _transition(a_digit: int, b_digit: int, carry: int) -> tuple[int, int]:
    index = a_digit + (10 * b_digit)
    return SUM_TABLE[index][carry], CARRY_TABLE[index][carry]


def add_decimal_strings(a: str, b: str) -> str:
    if not a.isdigit() or not b.isdigit():
        raise ValueError("operands must contain decimal digits only")
    width = max(len(a), len(b))
    a_padded = a.zfill(width)
    b_padded = b.zfill(width)
    carry = 0
    output_reversed: list[str] = []
    for a_char, b_char in zip(reversed(a_padded), reversed(b_padded), strict=True):
        digit, carry = _transition(ord(a_char) - 48, ord(b_char) - 48, carry)
        output_reversed.append(chr(48 + digit))
    if carry:
        output_reversed.append("1")
    return "".join(reversed(output_reversed)).lstrip("0") or "0"


@dataclass(frozen=True)
class FirmwarePlan:
    a: str
    b: str
    answer: str


class FrozenDecimalFirmware:
    """Immutable parser plus ripple-carry decimal addition.

    The parser deliberately accepts only the registered prompt templates.
    """

    trainable_parameters = 0

    def parse(self, text: str) -> FirmwarePlan | None:
        for pattern in _PATTERNS:
            match = pattern.search(text)
            if match:
                a, b = match.groups()
                return FirmwarePlan(a=a, b=b, answer=add_decimal_strings(a, b))
        return None

    def symbols(self, text: str) -> list[int] | None:
        plan = self.parse(text)
        if plan is None:
            return None
        return [ord(char) - 48 for char in plan.answer] + [10]
