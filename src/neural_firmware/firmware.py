from __future__ import annotations

import torch
from torch import nn

from neural_firmware.tokenizer import ArithmeticTokenizer


class FrozenRippleCarryAdder(nn.Module):
    """An immutable finite-state decimal adder represented by transition tables.

    The module has no trainable parameters. Its only persistent state is the
    truth table for one-column addition and a fixed token codebook.
    """

    def __init__(self, tokenizer: ArithmeticTokenizer, d_model: int) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        sum_table = torch.empty((10, 10, 2), dtype=torch.long)
        carry_table = torch.empty((10, 10, 2), dtype=torch.long)
        for a in range(10):
            for b in range(10):
                for carry in range(2):
                    total = a + b + carry
                    sum_table[a, b, carry] = total % 10
                    carry_table[a, b, carry] = total // 10
        self.register_buffer("sum_table", sum_table, persistent=True)
        self.register_buffer("carry_table", carry_table, persistent=True)
        self._sum_lookup = sum_table.tolist()
        self._carry_lookup = carry_table.tolist()

        generator = torch.Generator(device="cpu")
        generator.manual_seed(314159)
        codebook = torch.randn(tokenizer.vocab_size, d_model, generator=generator)
        codebook = torch.nn.functional.normalize(codebook, dim=-1)
        codebook[tokenizer.pad_id].zero_()
        self.register_buffer("codebook", codebook, persistent=True)

    def add_digit_ids(self, a_ids: list[int], b_ids: list[int]) -> list[int]:
        a_digits = [self.tokenizer.id_to_digit(token_id) for token_id in reversed(a_ids)]
        b_digits = [self.tokenizer.id_to_digit(token_id) for token_id in reversed(b_ids)]

        output_reversed: list[int] = []
        carry = 0
        for index in range(max(len(a_digits), len(b_digits))):
            a = a_digits[index] if index < len(a_digits) else 0
            b = b_digits[index] if index < len(b_digits) else 0
            output_reversed.append(self._sum_lookup[a][b][carry])
            carry = self._carry_lookup[a][b][carry]
        if carry:
            output_reversed.append(carry)
        return [
            self.tokenizer.digit_id(digit)
            for digit in reversed(output_reversed or [0])
        ]

    def _parse_operands(self, row: list[int]) -> tuple[list[int], list[int], int] | None:
        try:
            plus_index = row.index(self.tokenizer.plus_id)
            equals_index = row.index(self.tokenizer.equals_id)
        except ValueError:
            return None
        if not (1 < plus_index < equals_index - 1):
            return None
        a_ids = row[1:plus_index]
        b_ids = row[plus_index + 1 : equals_index]
        digit_min = self.tokenizer.zero_id
        digit_max = self.tokenizer.digit_id(9)
        if not all(digit_min <= token_id <= digit_max for token_id in a_ids + b_ids):
            return None
        return a_ids, b_ids, equals_index

    @torch.no_grad()
    def next_token_targets(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return exact next-token targets and positions where firmware is valid."""

        cpu_rows = input_ids.detach().cpu().tolist()
        targets = torch.full_like(input_ids, self.tokenizer.pad_id)
        valid = torch.zeros_like(input_ids, dtype=torch.bool)

        for batch_index, row in enumerate(cpu_rows):
            if self.tokenizer.pad_id in row:
                row = row[: row.index(self.tokenizer.pad_id)]
            parsed = self._parse_operands(row)
            if parsed is None:
                continue
            a_ids, b_ids, equals_index = parsed
            answer_ids = self.add_digit_ids(a_ids, b_ids)
            target_sequence = answer_ids + [self.tokenizer.eos_id]
            for offset, target_id in enumerate(target_sequence):
                position = equals_index + offset
                if position >= input_ids.shape[1]:
                    break
                targets[batch_index, position] = target_id
                valid[batch_index, position] = True
        return targets, valid

    def latent_signal(
        self,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        targets, valid = self.next_token_targets(input_ids)
        signal = torch.nn.functional.embedding(targets, self.codebook)
        signal = signal * valid.unsqueeze(-1)
        return signal, targets, valid
