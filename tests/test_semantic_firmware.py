from __future__ import annotations

import torch
from torch import nn

from neural_firmware.semantic_data import (
    TRAIN_ADDITION_FAMILIES,
    make_semantic_addition_examples,
)
from neural_firmware.semantic_firmware import (
    SemanticControlContext,
    SemanticFirmwareContext,
    SemanticInternalArithmeticUnit,
    SemanticInternalFirmwareLayer,
    SemanticLearnedControlLayer,
    SemanticMatchedResidualAdapter,
    SemanticRouter,
)
from neural_firmware.semantic_training import make_semantic_training_batch


class _IdentityLayer(nn.Module):
    attention_type = "full_attention"

    def forward(
        self,
        hidden_states: torch.Tensor,
        *args: object,
        **kwargs: object,
    ) -> torch.Tensor:
        return hidden_states


def _digits(value: str, width: int) -> torch.Tensor:
    result = torch.zeros((1, width), dtype=torch.long)
    result[0, : len(value)] = torch.tensor([int(character) for character in value])
    return result


def test_semantic_unit_parameter_count_and_exact_plan() -> None:
    unit = SemanticInternalArithmeticUnit(hidden_size=896, strength=64.0)
    assert unit.interface_parameter_count == 24_225
    symbols, mask = unit.plan_from_digits(
        _digits("999", 3),
        torch.tensor([3]),
        _digits("1", 3),
        torch.tensor([1]),
    )
    assert symbols[0][mask[0]].tolist() == [1, 0, 0, 0, 10]


def test_semantic_internal_route_off_is_identity_and_route_on_injects() -> None:
    unit = SemanticInternalArithmeticUnit(hidden_size=10, strength=3.0)
    wrapper = SemanticInternalFirmwareLayer(
        _IdentityLayer(),
        unit,
        depth_after_blocks=6,
    )
    hidden = torch.randn((1, 5, 10))
    off = SemanticFirmwareContext(
        a_digits=_digits("12", 2),
        a_lengths=torch.tensor([2]),
        b_digits=_digits("3", 2),
        b_lengths=torch.tensor([1]),
        output_positions=torch.tensor([[2, 3, 4, 0]]),
        route_mode="force_off",
    )
    wrapper.set_context(off)
    assert torch.equal(wrapper(hidden), hidden)
    on = SemanticFirmwareContext(
        a_digits=_digits("12", 2),
        a_lengths=torch.tensor([2]),
        b_digits=_digits("3", 2),
        b_lengths=torch.tensor([1]),
        output_positions=torch.tensor([[2, 3, 4, 0]]),
        route_mode="force_on",
    )
    wrapper.set_context(on)
    assert not torch.equal(wrapper(hidden), hidden)
    assert on.diagnostics["planned_symbols"].tolist() == [[1, 5, 10, -1]]


def test_semantic_control_exactly_matches_internal_interface_count() -> None:
    internal = SemanticInternalArithmeticUnit(hidden_size=896, strength=64.0)
    router = SemanticRouter(896, hidden_width=16)
    adapter = SemanticMatchedResidualAdapter(896, rank=5)
    control = SemanticLearnedControlLayer(
        _IdentityLayer(),
        router,
        adapter,
        depth_after_blocks=6,
    )
    assert sum(parameter.numel() for parameter in adapter.parameters()) == 9_856
    assert control.interface_parameter_count == internal.interface_parameter_count
    hidden = torch.randn((1, 3, 896))
    control.set_context(
        SemanticControlContext(generation=True, route_mode="force_off")
    )
    assert torch.equal(control(hidden), hidden)


class _Tokenizer:
    pad_token_id = 0
    eos_token_id = 99

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> list[int]:
        assert tokenize and add_generation_prompt
        return list(range(1, len(messages[0]["content"]) + 2))

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        assert not add_special_tokens
        return [int(text) + 10]


class _Bundle:
    tokenizer = _Tokenizer()
    device = torch.device("cpu")


def test_training_batch_reserves_cell_capacity_without_carry() -> None:
    examples = make_semantic_addition_examples(
        count=2,
        min_digits=2,
        max_digits=2,
        seed=41,
        split="test",
        families=TRAIN_ADDITION_FAMILIES[:1],
    )
    examples = [
        type(examples[0])(
            prompt=examples[index].prompt,
            a="10",
            b="10",
            answer="20",
            route_label=True,
            family=examples[index].family,
            family_index=examples[index].family_index,
            split="test",
        )
        for index in range(2)
    ]
    batch = make_semantic_training_batch(_Bundle(), examples)
    assert batch.internal_context.output_positions.shape == (2, 4)
    assert batch.target_mask.sum(dim=1).tolist() == [3, 3]
