from __future__ import annotations

import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from neural_firmware.phase7_implant import (
    ImplantRuntimeContext,
    NeuronImplantLayout,
    NeuronImplantMLP,
)
from neural_firmware.pretrained_data import answer_token_ids, chat_prompt_ids
from neural_firmware.pretrained_training import ModelBundle
from neural_firmware.semantic_data import SemanticPromptExample


def set_phase7_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def padded_batch(
    sequences: list[list[int]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(len(sequence) for sequence in sequences)
    input_ids = torch.full(
        (len(sequences), maximum),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        length = len(sequence)
        input_ids[row, :length] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, :length] = 1
    return input_ids, attention_mask


@dataclass(frozen=True)
class ChannelCensus:
    layer_index: int
    prompt_count: int
    token_count: int
    mean_abs: torch.Tensor
    root_mean_square: torch.Tensor
    active_fraction: torch.Tensor
    down_column_norm: torch.Tensor
    contribution_score: torch.Tensor

    def state_dict(self) -> dict[str, object]:
        return {
            "layer_index": self.layer_index,
            "prompt_count": self.prompt_count,
            "token_count": self.token_count,
            "mean_abs": self.mean_abs,
            "root_mean_square": self.root_mean_square,
            "active_fraction": self.active_fraction,
            "down_column_norm": self.down_column_norm,
            "contribution_score": self.contribution_score,
        }


@torch.inference_mode()
def collect_channel_census(
    bundle: ModelBundle,
    prompts: list[str],
    *,
    layer_index: int,
    batch_size: int = 4,
    activity_threshold: float = 1e-3,
) -> ChannelCensus:
    layer = bundle.model.model.layers[layer_index]
    mlp = layer.mlp
    if isinstance(mlp, NeuronImplantMLP):
        raise ValueError("census requires an unmodified Qwen MLP")
    width = mlp.up_proj.out_features
    absolute_sum = torch.zeros(width, dtype=torch.float64)
    square_sum = torch.zeros(width, dtype=torch.float64)
    active_sum = torch.zeros(width, dtype=torch.float64)
    token_count = 0
    captured: list[torch.Tensor] = []

    def capture_input(
        _module: nn.Module,
        arguments: tuple[torch.Tensor, ...],
    ) -> None:
        captured.append(arguments[0].detach())

    handle = mlp.register_forward_pre_hook(capture_input)
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            sequences = [chat_prompt_ids(bundle.tokenizer, prompt) for prompt in batch]
            input_ids, attention_mask = padded_batch(
                sequences,
                pad_token_id=bundle.tokenizer.pad_token_id,
                device=bundle.device,
            )
            captured.clear()
            bundle.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            if len(captured) != 1:
                raise RuntimeError("expected one MLP input capture per forward")
            hidden = captured[0]
            intermediate = mlp.act_fn(mlp.gate_proj(hidden)) * mlp.up_proj(hidden)
            valid = intermediate[attention_mask.to(torch.bool)].float().cpu()
            absolute_sum += valid.abs().sum(dim=0, dtype=torch.float64)
            square_sum += valid.square().sum(dim=0, dtype=torch.float64)
            active_sum += (valid.abs() > activity_threshold).sum(
                dim=0,
                dtype=torch.float64,
            )
            token_count += valid.shape[0]
    finally:
        handle.remove()

    if token_count == 0:
        raise ValueError("census received no valid tokens")
    mean_abs = (absolute_sum / token_count).to(torch.float32)
    root_mean_square = (square_sum / token_count).sqrt().to(torch.float32)
    active_fraction = (active_sum / token_count).to(torch.float32)
    down_column_norm = mlp.down_proj.weight.detach().float().cpu().norm(dim=0)
    contribution = mean_abs * down_column_norm
    return ChannelCensus(
        layer_index=layer_index,
        prompt_count=len(prompts),
        token_count=token_count,
        mean_abs=mean_abs,
        root_mean_square=root_mean_square,
        active_fraction=active_fraction,
        down_column_norm=down_column_norm,
        contribution_score=contribution,
    )


def select_low_impact_channels(
    censuses: list[ChannelCensus],
    *,
    width: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if not censuses:
        raise ValueError("at least one census is required")
    layer = censuses[0].layer_index
    channel_width = len(censuses[0].contribution_score)
    if any(census.layer_index != layer for census in censuses):
        raise ValueError("all censuses must refer to the same layer")
    if any(len(census.contribution_score) != channel_width for census in censuses):
        raise ValueError("census widths differ")
    if not 0 < width <= channel_width:
        raise ValueError("requested selection width is invalid")

    normalized: list[torch.Tensor] = []
    for census in censuses:
        score = census.contribution_score
        scale = score.median().clamp_min(torch.finfo(score.dtype).eps)
        normalized.append(score / scale)
    conservative_score = torch.stack(normalized).amax(dim=0)
    selected = conservative_score.argsort()[:width]
    return selected, conservative_score


class ChannelAblationMLP(nn.Module):
    def __init__(self, base_mlp: nn.Module, selected_indices: torch.Tensor) -> None:
        super().__init__()
        self.base_mlp = base_mlp
        self.register_buffer(
            "selected_indices",
            selected_indices.to(dtype=torch.long),
            persistent=False,
        )

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        intermediate = self.base_mlp.act_fn(
            self.base_mlp.gate_proj(hidden)
        ) * self.base_mlp.up_proj(hidden)
        intermediate = intermediate.clone()
        intermediate.index_fill_(-1, self.selected_indices, 0)
        return self.base_mlp.down_proj(intermediate)


@torch.inference_mode()
def evaluate_channel_ablation(
    bundle: ModelBundle,
    prompts: list[str],
    *,
    layer_index: int,
    selected_indices: torch.Tensor,
    batch_size: int = 4,
) -> dict[str, float | int]:
    base_mlp = bundle.model.model.layers[layer_index].mlp
    if isinstance(base_mlp, (NeuronImplantMLP, ChannelAblationMLP)):
        raise ValueError("ablation evaluation requires an unmodified Qwen MLP")
    base_logits: list[torch.Tensor] = []
    ablated_logits: list[torch.Tensor] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        sequences = [chat_prompt_ids(bundle.tokenizer, prompt) for prompt in batch]
        input_ids, attention_mask = padded_batch(
            sequences,
            pad_token_id=bundle.tokenizer.pad_token_id,
            device=bundle.device,
        )
        positions = attention_mask.sum(dim=1) - 1
        rows = torch.arange(len(batch), device=bundle.device)
        hidden = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        logits = bundle.model.lm_head(hidden[rows, positions])
        base_logits.append(logits.float().cpu())

        wrapper = ChannelAblationMLP(base_mlp, selected_indices)
        wrapper.to(bundle.device)
        bundle.model.model.layers[layer_index].mlp = wrapper
        try:
            hidden = bundle.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state
            logits = bundle.model.lm_head(hidden[rows, positions])
            ablated_logits.append(logits.float().cpu())
        finally:
            bundle.model.model.layers[layer_index].mlp = base_mlp

    base = torch.cat(base_logits)
    ablated = torch.cat(ablated_logits)
    base_probability = base.softmax(dim=-1)
    log_probability = base.log_softmax(dim=-1)
    ablated_log_probability = ablated.log_softmax(dim=-1)
    kl = (base_probability * (log_probability - ablated_log_probability)).sum(dim=-1)
    return {
        "prompts": len(prompts),
        "channels": len(selected_indices),
        "top1_agreement": float(
            (base.argmax(dim=-1) == ablated.argmax(dim=-1)).float().mean().item()
        ),
        "mean_kl_divergence": float(kl.mean().item()),
        "maximum_kl_divergence": float(kl.max().item()),
        "mean_logit_rms": float((base - ablated).square().mean(dim=-1).sqrt().mean()),
    }


@dataclass(frozen=True)
class ImplantFeatureSet:
    hidden: torch.Tensor
    route_targets: torch.Tensor
    a_digit_targets: torch.Tensor
    b_digit_targets: torch.Tensor
    step_targets: torch.Tensor
    positive_mask: torch.Tensor

    @property
    def examples(self) -> int:
        return self.hidden.shape[0]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return asdict(self)

    @classmethod
    def load_state_dict(cls, state: dict[str, torch.Tensor]) -> ImplantFeatureSet:
        return cls(**state)


def encode_digit_target(value: str, layout: NeuronImplantLayout) -> torch.Tensor:
    if len(value) > layout.max_digits:
        raise ValueError(f"operand {value} exceeds {layout.max_digits} digits")
    target = torch.full(
        (layout.max_digits,),
        layout.pad_digit,
        dtype=torch.long,
    )
    target[: len(value)] = torch.tensor([int(character) for character in value])
    return target


@torch.inference_mode()
def collect_implant_features(
    bundle: ModelBundle,
    examples: list[SemanticPromptExample],
    *,
    layer_index: int,
    layout: NeuronImplantLayout,
    batch_size: int = 4,
) -> ImplantFeatureSet:
    mlp = bundle.model.model.layers[layer_index].mlp
    if isinstance(mlp, NeuronImplantMLP):
        raise ValueError("feature collection requires an unmodified Qwen MLP")
    hidden_rows: list[torch.Tensor] = []
    route_targets: list[int] = []
    a_targets: list[torch.Tensor] = []
    b_targets: list[torch.Tensor] = []
    step_targets: list[int] = []
    positive_mask: list[bool] = []
    captured: list[torch.Tensor] = []

    def capture_input(
        _module: nn.Module,
        arguments: tuple[torch.Tensor, ...],
    ) -> None:
        captured.append(arguments[0].detach())

    handle = mlp.register_forward_pre_hook(capture_input)
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            sequences: list[list[int]] = []
            prediction_positions: list[list[int]] = []
            per_example_steps: list[list[int]] = []
            for example in batch:
                prompt_ids = chat_prompt_ids(bundle.tokenizer, example.prompt)
                if example.route_label:
                    if example.answer is None:
                        raise ValueError("positive example is missing an answer")
                    targets = answer_token_ids(bundle.tokenizer, example.answer)
                    if len(targets) > layout.step_width:
                        raise ValueError("answer exceeds implant step width")
                    sequences.append(prompt_ids + targets[:-1])
                    prediction_positions.append(
                        list(
                            range(
                                len(prompt_ids) - 1,
                                len(prompt_ids) - 1 + len(targets),
                            )
                        )
                    )
                    per_example_steps.append(list(range(len(targets))))
                else:
                    sequences.append(prompt_ids)
                    prediction_positions.append([len(prompt_ids) - 1])
                    per_example_steps.append([0])

            input_ids, attention_mask = padded_batch(
                sequences,
                pad_token_id=bundle.tokenizer.pad_token_id,
                device=bundle.device,
            )
            captured.clear()
            bundle.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            if len(captured) != 1:
                raise RuntimeError("expected one MLP input capture per forward")
            hidden = captured[0].float().cpu()
            for row, example in enumerate(batch):
                a_target = encode_digit_target(example.a, layout)
                b_target = encode_digit_target(example.b, layout)
                for position, step in zip(
                    prediction_positions[row],
                    per_example_steps[row],
                    strict=True,
                ):
                    hidden_rows.append(hidden[row, position])
                    route_targets.append(int(example.route_label))
                    a_targets.append(a_target)
                    b_targets.append(b_target)
                    step_targets.append(step)
                    positive_mask.append(example.route_label)
            captured.clear()
            del hidden, input_ids, attention_mask
            if bundle.device.type == "mps" and (start // batch_size) % 16 == 15:
                torch.mps.empty_cache()
    finally:
        handle.remove()

    return ImplantFeatureSet(
        hidden=torch.stack(hidden_rows),
        route_targets=torch.tensor(route_targets, dtype=torch.long),
        a_digit_targets=torch.stack(a_targets),
        b_digit_targets=torch.stack(b_targets),
        step_targets=torch.tensor(step_targets, dtype=torch.long),
        positive_mask=torch.tensor(positive_mask, dtype=torch.bool),
    )


@dataclass(frozen=True)
class InterfaceTrainConfig:
    seed: int = 12_701
    steps: int = 1_500
    batch_size: int = 256
    learning_rate: float = 0.003
    route_loss_weight: float = 1.0
    digit_loss_weight: float = 1.0
    step_loss_weight: float = 0.5


def evaluate_implant_interface(
    implant: NeuronImplantMLP,
    features: ImplantFeatureSet,
    *,
    device: torch.device,
) -> dict[str, object]:
    with torch.inference_mode():
        interface = implant.interface_logits(features.hidden.to(device))
        hard = implant.hard_interface(interface)
        positive = features.positive_mask.to(device)
        route_target = features.route_targets.to(device)
        a_target = features.a_digit_targets.to(device)
        b_target = features.b_digit_targets.to(device)
        step_target = features.step_targets.to(device)
        a_exact = (hard.a_digits == a_target).all(dim=-1)
        b_exact = (hard.b_digits == b_target).all(dim=-1)
        operand_exact = a_exact & b_exact
        return {
            "rows": features.examples,
            "positive_rows": int(positive.sum().item()),
            "negative_rows": int((~positive).sum().item()),
            "route_accuracy": float((hard.route == route_target).float().mean().item()),
            "route_true_positive_rate": float(
                (hard.route[positive] == 1).float().mean().item()
            ),
            "route_false_positive_rate": float(
                (hard.route[~positive] == 1).float().mean().item()
            ),
            "operand_exact_rate": float(operand_exact[positive].float().mean().item()),
            "a_exact_rate": float(a_exact[positive].float().mean().item()),
            "b_exact_rate": float(b_exact[positive].float().mean().item()),
            "digit_slot_accuracy": float(
                torch.cat(
                    (
                        hard.a_digits[positive] == a_target[positive],
                        hard.b_digits[positive] == b_target[positive],
                    ),
                    dim=-1,
                )
                .float()
                .mean()
                .item()
            ),
            "step_accuracy": float(
                (hard.step[positive] == step_target[positive]).float().mean().item()
            ),
            "route_probabilities": hard.route_probability.detach().cpu(),
            "route_targets": features.route_targets.clone(),
        }


def select_route_threshold(
    route_probabilities: torch.Tensor,
    route_targets: torch.Tensor,
    *,
    maximum_false_positive_rate: float = 0.01,
) -> dict[str, float]:
    positive = route_targets.to(torch.bool)
    candidates = torch.unique(
        torch.cat(
            (
                torch.tensor([0.5, 0.9999]),
                route_probabilities.float(),
            )
        )
    ).sort().values
    choices: list[tuple[float, float, float]] = []
    for threshold in candidates:
        route = route_probabilities >= threshold
        true_positive = float(route[positive].float().mean().item())
        false_positive = float(route[~positive].float().mean().item())
        choices.append((float(threshold), true_positive, false_positive))
    feasible = [
        choice
        for choice in choices
        if choice[2] <= maximum_false_positive_rate
    ]
    if feasible:
        selected = max(feasible, key=lambda value: (value[1], -value[2], -value[0]))
    else:
        selected = max(
            choices,
            key=lambda value: (value[1] - value[2], value[1], -value[2]),
        )
    return {
        "threshold": selected[0],
        "true_positive_rate": selected[1],
        "false_positive_rate": selected[2],
    }


def train_implant_interface(
    implant: NeuronImplantMLP,
    train: ImplantFeatureSet,
    development: ImplantFeatureSet,
    *,
    device: torch.device,
    config: InterfaceTrainConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    set_phase7_seed(config.seed)
    implant.input_rows.requires_grad_(True)
    implant.result_columns.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [implant.input_rows],
        lr=config.learning_rate,
    )
    generator = torch.Generator().manual_seed(config.seed)
    positive_indices = torch.where(train.positive_mask)[0]
    negative_indices = torch.where(~train.positive_mask)[0]
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()

    for step in range(config.steps):
        half = max(1, config.batch_size // 2)
        positive_sample = positive_indices[
            torch.randint(len(positive_indices), (half,), generator=generator)
        ]
        negative_sample = negative_indices[
            torch.randint(
                len(negative_indices),
                (config.batch_size - half,),
                generator=generator,
            )
        ]
        indices = torch.cat((positive_sample, negative_sample))
        hidden = train.hidden[indices].to(device)
        route_target = train.route_targets[indices].to(device)
        local_positive = train.positive_mask[indices].to(device)
        interface = implant.interface_logits(hidden)
        route_loss = nn.functional.cross_entropy(
            interface.route_logits,
            route_target,
        )
        digit_loss = (
            nn.functional.cross_entropy(
                interface.a_digit_logits[local_positive].reshape(
                    -1,
                    implant.layout.digit_classes,
                ),
                train.a_digit_targets[indices][local_positive.cpu()]
                .reshape(-1)
                .to(device),
            )
            + nn.functional.cross_entropy(
                interface.b_digit_logits[local_positive].reshape(
                    -1,
                    implant.layout.digit_classes,
                ),
                train.b_digit_targets[indices][local_positive.cpu()]
                .reshape(-1)
                .to(device),
            )
        ) / 2
        step_loss = nn.functional.cross_entropy(
            interface.step_logits[local_positive],
            train.step_targets[indices][local_positive.cpu()].to(device),
        )
        loss = (
            config.route_loss_weight * route_loss
            + config.digit_loss_weight * digit_loss
            + config.step_loss_weight * step_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        final_loss = float(loss.detach().cpu())

    development_metrics = evaluate_implant_interface(
        implant,
        development,
        device=device,
    )
    threshold = select_route_threshold(
        development_metrics["route_probabilities"],
        development_metrics["route_targets"],
    )
    implant.route_threshold = float(threshold["threshold"])
    development_metrics = evaluate_implant_interface(
        implant,
        development,
        device=device,
    )
    compact_metrics = {
        key: value
        for key, value in development_metrics.items()
        if key not in {"route_probabilities", "route_targets"}
    }
    training = {
        "config": asdict(config),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "wall_time_seconds": time.perf_counter() - started,
        "trainable_parameters": implant.input_rows.numel(),
        "selected_route_threshold": threshold,
    }
    return training, compact_metrics


@dataclass(frozen=True)
class OutputTrainConfig:
    seed: int = 12_701
    steps: int = 300
    batch_size: int = 1
    learning_rate: float = 0.01


def teacher_context(
    examples: list[SemanticPromptExample],
    *,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    tokenizer: object,
    layout: NeuronImplantLayout,
    device: torch.device,
) -> tuple[ImplantRuntimeContext, list[list[int]], list[list[int]]]:
    maximum = max(len(sequence) for sequence in sequences)
    batch = len(examples)
    eligible = torch.zeros((batch, maximum), dtype=torch.bool, device=device)
    route = torch.full((batch, maximum), -1, dtype=torch.long, device=device)
    a_digits = torch.full(
        (batch, maximum, layout.max_digits),
        -1,
        dtype=torch.long,
        device=device,
    )
    b_digits = torch.full_like(a_digits, -1)
    steps = torch.full((batch, maximum), -1, dtype=torch.long, device=device)
    prediction_positions: list[list[int]] = []
    targets_by_row: list[list[int]] = []
    for row, (example, prompt_length) in enumerate(
        zip(examples, prompt_lengths, strict=True)
    ):
        if example.answer is None:
            raise ValueError("output training requires positive examples")
        targets = answer_token_ids(tokenizer, example.answer)
        positions = list(
            range(
                prompt_length - 1,
                prompt_length - 1 + len(targets),
            )
        )
        prediction_positions.append(positions)
        targets_by_row.append(targets)
        target_a = encode_digit_target(example.a, layout).to(device)
        target_b = encode_digit_target(example.b, layout).to(device)
        eligible[row, positions] = True
        route[row, positions] = 1
        a_digits[row, positions] = target_a
        b_digits[row, positions] = target_b
        steps[row, positions] = torch.arange(len(targets), device=device)
    return (
        ImplantRuntimeContext(
            eligible_mask=eligible,
            teacher_route=route,
            teacher_a_digits=a_digits,
            teacher_b_digits=b_digits,
            teacher_step=steps,
        ),
        prediction_positions,
        targets_by_row,
    )


def train_implant_output(
    bundle: ModelBundle,
    implant: NeuronImplantMLP,
    examples: list[SemanticPromptExample],
    *,
    config: OutputTrainConfig,
) -> dict[str, object]:
    positives = [example for example in examples if example.route_label]
    if not positives:
        raise ValueError("output training requires positive examples")
    set_phase7_seed(config.seed)
    implant.input_rows.requires_grad_(False)
    implant.result_columns.requires_grad_(True)
    optimizer = torch.optim.AdamW(
        [implant.result_columns],
        lr=config.learning_rate,
    )
    rng = random.Random(config.seed)
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()

    for step in range(config.steps):
        batch = [rng.choice(positives) for _ in range(config.batch_size)]
        prompt_ids = [
            chat_prompt_ids(bundle.tokenizer, example.prompt) for example in batch
        ]
        target_ids = [
            answer_token_ids(bundle.tokenizer, example.answer or "")
            for example in batch
        ]
        sequences = [
            prompt + targets[:-1]
            for prompt, targets in zip(prompt_ids, target_ids, strict=True)
        ]
        input_ids, attention_mask = padded_batch(
            sequences,
            pad_token_id=bundle.tokenizer.pad_token_id,
            device=bundle.device,
        )
        context, positions_by_row, targets_by_row = teacher_context(
            batch,
            sequences=sequences,
            prompt_lengths=[len(prompt) for prompt in prompt_ids],
            tokenizer=bundle.tokenizer,
            layout=implant.layout,
            device=bundle.device,
        )
        implant.set_context(context)
        hidden = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state
        selected_logits: list[torch.Tensor] = []
        selected_targets: list[int] = []
        for row, (positions, targets) in enumerate(
            zip(positions_by_row, targets_by_row, strict=True)
        ):
            selected_logits.append(bundle.model.lm_head(hidden[row, positions]))
            selected_targets.extend(targets)
        token_logits = torch.cat(selected_logits)
        token_targets = torch.tensor(
            selected_targets,
            dtype=torch.long,
            device=bundle.device,
        )
        loss = nn.functional.cross_entropy(token_logits.float(), token_targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        final_loss = float(loss.detach().cpu())
        implant.set_context(None)
        del (
            hidden,
            selected_logits,
            token_logits,
            loss,
            input_ids,
            attention_mask,
            context,
        )
        if bundle.device.type == "mps" and step % 25 == 24:
            torch.mps.empty_cache()
    implant.set_context(None)
    return {
        "config": asdict(config),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "wall_time_seconds": time.perf_counter() - started,
        "trainable_parameters": implant.result_columns.numel(),
    }


@contextmanager
def temporarily_uninstalled_implant(
    bundle: ModelBundle,
    implant: NeuronImplantMLP,
    *,
    layer_index: int,
):
    current = bundle.model.model.layers[layer_index].mlp
    if current is not implant:
        raise ValueError("provided implant is not installed at the target layer")
    bundle.model.model.layers[layer_index].mlp = implant.base_mlp
    try:
        yield
    finally:
        bundle.model.model.layers[layer_index].mlp = implant


@torch.inference_mode()
def generate_with_implant(
    bundle: ModelBundle,
    implant: NeuronImplantMLP,
    prompt: str,
    *,
    max_new_tokens: int,
    ablate_result: bool = False,
) -> dict[str, object]:
    prompt_ids = chat_prompt_ids(bundle.tokenizer, prompt)
    input_ids = torch.tensor(
        [prompt_ids],
        dtype=torch.long,
        device=bundle.device,
    )
    attention_mask = torch.ones_like(input_ids)
    generated: list[int] = []
    step_diagnostics: list[dict[str, object]] = []
    past_key_values = None

    for generation_step in range(max_new_tokens):
        eligible = torch.zeros_like(input_ids, dtype=torch.bool)
        eligible[:, -1] = True
        context = ImplantRuntimeContext(
            eligible_mask=eligible,
            ablate_result=ablate_result,
            capture_diagnostics=True,
        )
        implant.set_context(context)
        outputs = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        hidden = outputs.last_hidden_state[:, -1]
        next_token = bundle.model.lm_head(hidden).argmax(dim=-1)
        token_id = int(next_token.item())
        generated.append(token_id)
        diagnostic: dict[str, object] = {"generation_step": generation_step}
        for key, value in context.diagnostics.items():
            if isinstance(value, torch.Tensor):
                diagnostic[key] = value.tolist()
            else:
                diagnostic[key] = value
        diagnostic["token_id"] = token_id
        diagnostic["token_text"] = bundle.tokenizer.decode([token_id])
        step_diagnostics.append(diagnostic)
        if token_id == bundle.tokenizer.eos_token_id:
            break
        input_ids = next_token.unsqueeze(0)
        attention_mask = torch.cat(
            (
                attention_mask,
                torch.ones(
                    (1, 1),
                    dtype=torch.long,
                    device=bundle.device,
                ),
            ),
            dim=1,
        )
    implant.set_context(None)
    return {
        "generated_token_ids": generated,
        "generated_text": bundle.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip(),
        "steps": step_diagnostics,
    }


@torch.inference_mode()
def generate_untouched(
    bundle: ModelBundle,
    implant: NeuronImplantMLP,
    prompt: str,
    *,
    layer_index: int,
    max_new_tokens: int,
) -> dict[str, object]:
    with temporarily_uninstalled_implant(
        bundle,
        implant,
        layer_index=layer_index,
    ):
        prompt_ids = chat_prompt_ids(bundle.tokenizer, prompt)
        input_ids = torch.tensor(
            [prompt_ids],
            dtype=torch.long,
            device=bundle.device,
        )
        attention_mask = torch.ones_like(input_ids)
        generated: list[int] = []
        past_key_values = None
        for _ in range(max_new_tokens):
            outputs = bundle.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=True,
            )
            past_key_values = outputs.past_key_values
            hidden = outputs.last_hidden_state[:, -1]
            next_token = bundle.model.lm_head(hidden).argmax(dim=-1)
            token_id = int(next_token.item())
            generated.append(token_id)
            if token_id == bundle.tokenizer.eos_token_id:
                break
            input_ids = next_token.unsqueeze(0)
            attention_mask = torch.cat(
                (
                    attention_mask,
                    torch.ones(
                        (1, 1),
                        dtype=torch.long,
                        device=bundle.device,
                    ),
                ),
                dim=1,
            )
    return {
        "generated_token_ids": generated,
        "generated_text": bundle.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip(),
    }
