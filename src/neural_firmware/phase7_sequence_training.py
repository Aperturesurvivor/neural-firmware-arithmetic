from __future__ import annotations

import random
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass

import torch
from torch import nn

from neural_firmware.phase7_sequence_implant import (
    SequenceImplantContext,
    SequenceImplantLayout,
    SequenceNeuronImplantMLP,
)
from neural_firmware.phase7_training import (
    padded_batch,
    select_route_threshold,
    set_phase7_seed,
)
from neural_firmware.pretrained_data import (
    answer_token_ids,
    chat_prompt_ids,
    chat_prompt_ids_and_content_mask,
    decimal_digit_token_id,
)
from neural_firmware.pretrained_training import ModelBundle
from neural_firmware.semantic_data import SemanticPromptExample


def subsequence_starts(sequence: list[int], subsequence: list[int]) -> list[int]:
    if not subsequence:
        raise ValueError("cannot locate an empty token subsequence")
    return [
        start
        for start in range(len(sequence) - len(subsequence) + 1)
        if sequence[start : start + len(subsequence)] == subsequence
    ]


def operand_token_positions(
    tokenizer: object,
    prompt_ids: list[int],
    a: str,
    b: str,
) -> tuple[list[int], list[int]]:
    def digit_ids(value: str) -> list[int]:
        return [
            decimal_digit_token_id(tokenizer, character)
            for character in value
        ]

    a_ids = digit_ids(a)
    b_ids = digit_ids(b)
    a_starts = subsequence_starts(prompt_ids, a_ids)
    b_starts = subsequence_starts(prompt_ids, b_ids)
    candidates = [
        (a_start, b_start)
        for a_start in a_starts
        for b_start in b_starts
        if a_start + len(a_ids) <= b_start
    ]
    if not candidates:
        raise ValueError(f"could not locate ordered operands {a}, {b}")
    a_start, b_start = min(candidates)
    return (
        list(range(a_start, a_start + len(a_ids))),
        list(range(b_start, b_start + len(b_ids))),
    )


def sequence_labels(
    tokenizer: object,
    example: SemanticPromptExample,
    *,
    sequence: list[int],
    prompt_length: int,
    layout: SequenceImplantLayout,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    roles = torch.zeros(len(sequence), dtype=torch.long)
    digits = torch.full(
        (len(sequence),),
        layout.non_digit,
        dtype=torch.long,
    )
    route = torch.full((len(sequence),), -1, dtype=torch.long)
    step = torch.full((len(sequence),), -1, dtype=torch.long)
    prompt_ids = sequence[:prompt_length]
    a_positions, b_positions = operand_token_positions(
        tokenizer,
        prompt_ids,
        example.a,
        example.b,
    )
    for role, positions, value in (
        (1, a_positions, example.a),
        (2, b_positions, example.b),
    ):
        if len(positions) > layout.max_digits:
            raise ValueError("operand exceeds sequence implant width")
        roles[positions] = role
        digits[positions] = torch.tensor([int(character) for character in value])

    if example.route_label:
        if example.answer is None:
            raise ValueError("positive sequence example is missing an answer")
        targets = answer_token_ids(tokenizer, example.answer)
        positions = list(
            range(
                prompt_length - 1,
                prompt_length - 1 + len(targets),
            )
        )
        route[positions] = 1
        step[positions] = torch.arange(len(targets))
    else:
        route[prompt_length - 1] = 0
    return route, roles, digits, step


@dataclass(frozen=True)
class SequenceFeatureSet:
    hidden: torch.Tensor
    route_targets: torch.Tensor
    role_targets: torch.Tensor
    digit_targets: torch.Tensor
    step_targets: torch.Tensor

    @property
    def rows(self) -> int:
        return self.hidden.shape[0]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return asdict(self)

    @classmethod
    def load_state_dict(cls, state: dict[str, torch.Tensor]) -> SequenceFeatureSet:
        return cls(**state)


@dataclass(frozen=True)
class FirstStepRouteFeatureSet:
    hidden: torch.Tensor
    targets: torch.Tensor

    @property
    def rows(self) -> int:
        return self.hidden.shape[0]

    def state_dict(self) -> dict[str, torch.Tensor]:
        return asdict(self)

    @classmethod
    def load_state_dict(
        cls,
        state: dict[str, torch.Tensor],
    ) -> FirstStepRouteFeatureSet:
        return cls(**state)


@torch.inference_mode()
def collect_first_step_route_features(
    bundle: ModelBundle,
    examples: list[SemanticPromptExample],
    *,
    layer_index: int,
    batch_size: int = 8,
) -> FirstStepRouteFeatureSet:
    mlp = bundle.model.model.layers[layer_index].mlp
    if isinstance(mlp, SequenceNeuronImplantMLP):
        raise ValueError("feature collection requires an unmodified MLP")
    captured: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []

    def capture_input(
        _module: nn.Module,
        arguments: tuple[torch.Tensor, ...],
    ) -> None:
        captured.append(arguments[0].detach())

    handle = mlp.register_forward_pre_hook(capture_input)
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompt_ids = [
                chat_prompt_ids(bundle.tokenizer, example.prompt) for example in batch
            ]
            input_ids, attention_mask = padded_batch(
                prompt_ids,
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
                raise RuntimeError("expected one sequence MLP capture")
            hidden = captured[0]
            positions = attention_mask.sum(dim=1) - 1
            rows = torch.arange(len(batch), device=bundle.device)
            hidden_rows.append(hidden[rows, positions].float().cpu())
            captured.clear()
            del hidden, input_ids, attention_mask
            if bundle.device.type == "mps" and (start // batch_size) % 32 == 31:
                torch.mps.empty_cache()
    finally:
        handle.remove()
    return FirstStepRouteFeatureSet(
        hidden=torch.cat(hidden_rows),
        targets=torch.tensor(
            [int(example.route_label) for example in examples],
            dtype=torch.long,
        ),
    )


@dataclass(frozen=True)
class RouteRowTrainConfig:
    seed: int = 13_501
    steps: int = 2_000
    batch_size: int = 256
    learning_rate: float = 0.001
    maximum_development_false_positive_rate: float = 0.005


def evaluate_route_rows(
    route_rows: torch.Tensor,
    features: FirstStepRouteFeatureSet,
) -> dict[str, object]:
    with torch.inference_mode():
        probabilities = nn.functional.linear(
            features.hidden.float(),
            route_rows.float(),
        ).softmax(dim=-1)[..., 1]
    targets = features.targets
    selected = select_route_threshold(
        probabilities,
        targets,
        maximum_false_positive_rate=1.0,
    )
    return {
        "rows": features.rows,
        "positive_rows": int((targets == 1).sum()),
        "negative_rows": int((targets == 0).sum()),
        "probabilities": probabilities,
        "targets": targets,
        "unconstrained_best_threshold": selected,
    }


def train_route_rows(
    initial_route_rows: torch.Tensor,
    train: FirstStepRouteFeatureSet,
    development: FirstStepRouteFeatureSet,
    *,
    device: torch.device,
    config: RouteRowTrainConfig,
) -> tuple[torch.Tensor, dict[str, object], dict[str, object]]:
    if initial_route_rows.shape != (2, train.hidden.shape[1]):
        raise ValueError("route row shape does not match first-step features")
    if development.hidden.shape[1] != train.hidden.shape[1]:
        raise ValueError("training and development hidden widths differ")
    set_phase7_seed(config.seed)
    route_rows = nn.Parameter(initial_route_rows.detach().clone().to(device))
    optimizer = torch.optim.AdamW(
        [route_rows],
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(config.seed)
    positive = torch.where(train.targets == 1)[0]
    negative = torch.where(train.targets == 0)[0]
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()
    for step in range(config.steps):
        indices = balanced_sample(
            positive,
            negative,
            count=config.batch_size,
            generator=generator,
        )
        logits = nn.functional.linear(
            train.hidden[indices].to(device).float(),
            route_rows.float(),
        )
        loss = nn.functional.cross_entropy(
            logits,
            train.targets[indices].to(device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step == 0:
            initial_loss = float(loss.detach().cpu())
        final_loss = float(loss.detach().cpu())
    trained = route_rows.detach().float().cpu()
    development_metrics = evaluate_route_rows(trained, development)
    threshold = select_route_threshold(
        development_metrics["probabilities"],
        development_metrics["targets"],
        maximum_false_positive_rate=(
            config.maximum_development_false_positive_rate
        ),
    )
    probability = development_metrics["probabilities"]
    target = development_metrics["targets"]
    prediction = probability >= threshold["threshold"]
    positive_target = target == 1
    compact_development = {
        "rows": development.rows,
        "positive_rows": int(positive_target.sum()),
        "negative_rows": int((~positive_target).sum()),
        "threshold": threshold,
        "true_positive_rate": float(
            prediction[positive_target].float().mean()
        ),
        "false_positive_rate": float(
            prediction[~positive_target].float().mean()
        ),
        "accuracy": float((prediction == target.to(torch.bool)).float().mean()),
        "positive_probability_min": float(probability[positive_target].min()),
        "negative_probability_max": float(probability[~positive_target].max()),
    }
    training = {
        "config": asdict(config),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "wall_time_seconds": time.perf_counter() - started,
        "trainable_parameters": trained.numel(),
    }
    return trained, training, compact_development


@torch.inference_mode()
def collect_sequence_features(
    bundle: ModelBundle,
    examples: list[SemanticPromptExample],
    *,
    layer_index: int,
    layout: SequenceImplantLayout,
    batch_size: int = 4,
    ordinary_tokens_per_example: int | None = None,
) -> SequenceFeatureSet:
    mlp = bundle.model.model.layers[layer_index].mlp
    if isinstance(mlp, SequenceNeuronImplantMLP):
        raise ValueError("feature collection requires an unmodified MLP")
    captured: list[torch.Tensor] = []
    hidden_rows: list[torch.Tensor] = []
    route_rows: list[torch.Tensor] = []
    role_rows: list[torch.Tensor] = []
    digit_rows: list[torch.Tensor] = []
    step_rows: list[torch.Tensor] = []

    def capture_input(
        _module: nn.Module,
        arguments: tuple[torch.Tensor, ...],
    ) -> None:
        captured.append(arguments[0].detach())

    handle = mlp.register_forward_pre_hook(capture_input)
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            prompt_ids = [
                chat_prompt_ids(bundle.tokenizer, example.prompt) for example in batch
            ]
            sequences: list[list[int]] = []
            labels: list[
                tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
            ] = []
            for example, prompt in zip(batch, prompt_ids, strict=True):
                if example.route_label:
                    targets = answer_token_ids(
                        bundle.tokenizer,
                        example.answer or "",
                    )
                    sequence = prompt + targets[:-1]
                else:
                    sequence = prompt
                sequences.append(sequence)
                labels.append(
                    sequence_labels(
                        bundle.tokenizer,
                        example,
                        sequence=sequence,
                        prompt_length=len(prompt),
                        layout=layout,
                    )
                )
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
                raise RuntimeError("expected one sequence MLP capture")
            hidden = captured[0].float().cpu()
            for row, sequence in enumerate(sequences):
                length = len(sequence)
                route, roles, digits, steps = labels[row]
                if ordinary_tokens_per_example is None:
                    keep = torch.ones(length, dtype=torch.bool)
                else:
                    keep = (route >= 0) | (roles > 0)
                    ordinary = torch.where(~keep)[0]
                    if len(ordinary) > ordinary_tokens_per_example:
                        sample_positions = torch.linspace(
                            0,
                            len(ordinary) - 1,
                            ordinary_tokens_per_example,
                        ).round().to(torch.long)
                        ordinary = ordinary[sample_positions]
                    keep[ordinary] = True
                hidden_rows.append(hidden[row, :length][keep])
                route_rows.append(route[keep])
                role_rows.append(roles[keep])
                digit_rows.append(digits[keep])
                step_rows.append(steps[keep])
            captured.clear()
            del hidden, input_ids, attention_mask
            if bundle.device.type == "mps" and (start // batch_size) % 16 == 15:
                torch.mps.empty_cache()
    finally:
        handle.remove()

    return SequenceFeatureSet(
        hidden=torch.cat(hidden_rows),
        route_targets=torch.cat(route_rows),
        role_targets=torch.cat(role_rows),
        digit_targets=torch.cat(digit_rows),
        step_targets=torch.cat(step_rows),
    )


@dataclass(frozen=True)
class SequenceInterfaceTrainConfig:
    seed: int = 12_801
    steps: int = 1_500
    batch_size: int = 256
    learning_rate: float = 0.003
    route_loss_weight: float = 1.0
    role_loss_weight: float = 1.0
    digit_loss_weight: float = 1.0
    step_loss_weight: float = 0.5


def balanced_sample(
    positive: torch.Tensor,
    negative: torch.Tensor,
    *,
    count: int,
    generator: torch.Generator,
) -> torch.Tensor:
    half = max(1, count // 2)
    return torch.cat(
        (
            positive[
                torch.randint(len(positive), (half,), generator=generator)
            ],
            negative[
                torch.randint(
                    len(negative),
                    (count - half,),
                    generator=generator,
                )
            ],
        )
    )


def evaluate_sequence_interface(
    implant: SequenceNeuronImplantMLP,
    features: SequenceFeatureSet,
    *,
    device: torch.device,
) -> dict[str, object]:
    with torch.inference_mode():
        interface = implant.interface_logits(features.hidden.to(device))
        hard = implant.hard_interface(interface)
        route_mask = features.route_targets >= 0
        route_target = features.route_targets[route_mask].to(device)
        route_prediction = hard.route[route_mask.to(device)]
        route_positive = route_target == 1
        role_target = features.role_targets.to(device)
        digit_target = features.digit_targets.to(device)
        operand_mask = role_target > 0
        step_mask = features.step_targets >= 0
        return {
            "rows": features.rows,
            "route_rows": int(route_mask.sum()),
            "route_true_positive_rate": float(
                (route_prediction[route_positive] == 1).float().mean()
            ),
            "route_false_positive_rate": float(
                (route_prediction[~route_positive] == 1).float().mean()
            ),
            "route_accuracy": float(
                (route_prediction == route_target).float().mean()
            ),
            "role_accuracy": float((hard.roles == role_target).float().mean()),
            "operand_role_accuracy": float(
                (hard.roles[operand_mask] == role_target[operand_mask]).float().mean()
            ),
            "digit_accuracy_on_operands": float(
                (hard.digits[operand_mask] == digit_target[operand_mask])
                .float()
                .mean()
            ),
            "step_accuracy": float(
                (
                    hard.step[step_mask.to(device)]
                    == features.step_targets[step_mask].to(device)
                )
                .float()
                .mean()
            ),
            "route_probabilities": hard.route_probability[
                route_mask.to(device)
            ].detach().cpu(),
            "route_targets": features.route_targets[route_mask].clone(),
        }


def train_sequence_interface(
    implant: SequenceNeuronImplantMLP,
    train: SequenceFeatureSet,
    development: SequenceFeatureSet,
    *,
    device: torch.device,
    config: SequenceInterfaceTrainConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    set_phase7_seed(config.seed)
    implant.input_rows.requires_grad_(
        implant.interface_kind != "bottleneck_silu"
    )
    implant.gate_rows.requires_grad_(implant.use_swiglu_interface)
    implant.result_columns.requires_grad_(False)
    if implant.interface_kind == "bottleneck_silu":
        implant.bottleneck_rows.requires_grad_(True)
        implant.bottleneck_mix.requires_grad_(True)
        interface_parameters = [
            implant.bottleneck_rows,
            implant.bottleneck_mix,
        ]
    else:
        interface_parameters = [implant.input_rows]
    if implant.use_swiglu_interface:
        interface_parameters.append(implant.gate_rows)
    if implant.representation_rank:
        implant.representation_down.requires_grad_(True)
        implant.representation_up.requires_grad_(True)
        interface_parameters.extend(
            (implant.representation_down, implant.representation_up)
        )
    optimizer = torch.optim.AdamW(
        interface_parameters,
        lr=config.learning_rate,
    )
    generator = torch.Generator().manual_seed(config.seed)

    route_positive = torch.where(train.route_targets == 1)[0]
    route_negative = torch.where(train.route_targets == 0)[0]
    operand_tokens = torch.where(train.role_targets > 0)[0]
    ordinary_tokens = torch.where(train.role_targets == 0)[0]
    step_rows = torch.where(train.step_targets >= 0)[0]
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()

    for training_step in range(config.steps):
        route_indices = balanced_sample(
            route_positive,
            route_negative,
            count=config.batch_size,
            generator=generator,
        )
        role_indices = balanced_sample(
            operand_tokens,
            ordinary_tokens,
            count=config.batch_size,
            generator=generator,
        )
        digit_indices = role_indices
        selected_step_rows = step_rows[
            torch.randint(
                len(step_rows),
                (config.batch_size,),
                generator=generator,
            )
        ]
        route_interface = implant.interface_logits(
            train.hidden[route_indices].to(device)
        )
        token_interface = implant.interface_logits(
            train.hidden[role_indices].to(device)
        )
        step_interface = implant.interface_logits(
            train.hidden[selected_step_rows].to(device)
        )
        route_loss = nn.functional.cross_entropy(
            route_interface.route_logits,
            train.route_targets[route_indices].to(device),
        )
        role_loss = nn.functional.cross_entropy(
            token_interface.role_logits,
            train.role_targets[role_indices].to(device),
        )
        digit_loss = nn.functional.cross_entropy(
            token_interface.digit_logits,
            train.digit_targets[digit_indices].to(device),
        )
        step_loss = nn.functional.cross_entropy(
            step_interface.step_logits,
            train.step_targets[selected_step_rows].to(device),
        )
        loss = (
            config.route_loss_weight * route_loss
            + config.role_loss_weight * role_loss
            + config.digit_loss_weight * digit_loss
            + config.step_loss_weight * step_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if training_step == 0:
            initial_loss = float(loss.detach().cpu())
        final_loss = float(loss.detach().cpu())

    development_metrics = evaluate_sequence_interface(
        implant,
        development,
        device=device,
    )
    threshold = select_route_threshold(
        development_metrics["route_probabilities"],
        development_metrics["route_targets"],
    )
    implant.route_threshold = float(threshold["threshold"])
    development_metrics = evaluate_sequence_interface(
        implant,
        development,
        device=device,
    )
    compact = {
        key: value
        for key, value in development_metrics.items()
        if key not in {"route_probabilities", "route_targets"}
    }
    return (
        {
            "config": asdict(config),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "wall_time_seconds": time.perf_counter() - started,
            "trainable_parameters": sum(
                parameter.numel() for parameter in interface_parameters
            ),
            "selected_route_threshold": threshold,
        },
        compact,
    )


def sequence_teacher_context(
    examples: list[SemanticPromptExample],
    *,
    sequences: list[list[int]],
    prompt_lengths: list[int],
    tokenizer: object,
    layout: SequenceImplantLayout,
    device: torch.device,
) -> tuple[SequenceImplantContext, list[list[int]], list[list[int]]]:
    maximum = max(len(sequence) for sequence in sequences)
    batch = len(examples)
    eligible = torch.zeros((batch, maximum), dtype=torch.bool, device=device)
    sequence_mask = torch.zeros_like(eligible)
    route = torch.full((batch, maximum), -1, dtype=torch.long, device=device)
    roles = torch.full_like(route, -1)
    digits = torch.full_like(route, -1)
    steps = torch.full_like(route, -1)
    positions_by_row: list[list[int]] = []
    targets_by_row: list[list[int]] = []
    for row, (example, sequence, prompt_length) in enumerate(
        zip(examples, sequences, prompt_lengths, strict=True)
    ):
        sequence_mask[row, : len(sequence)] = True
        route_labels, role_labels, digit_labels, step_labels = sequence_labels(
            tokenizer,
            example,
            sequence=sequence,
            prompt_length=prompt_length,
            layout=layout,
        )
        route[row, : len(sequence)] = route_labels.to(device)
        roles[row, : len(sequence)] = role_labels.to(device)
        digits[row, : len(sequence)] = digit_labels.to(device)
        steps[row, : len(sequence)] = step_labels.to(device)
        targets = answer_token_ids(tokenizer, example.answer or "")
        positions = list(
            range(prompt_length - 1, prompt_length - 1 + len(targets))
        )
        eligible[row, positions] = True
        positions_by_row.append(positions)
        targets_by_row.append(targets)
    return (
        SequenceImplantContext(
            eligible_mask=eligible,
            sequence_mask=sequence_mask,
            teacher_route=route,
            teacher_roles=roles,
            teacher_digits=digits,
            teacher_step=steps,
        ),
        positions_by_row,
        targets_by_row,
    )


@dataclass(frozen=True)
class SequenceOutputTrainConfig:
    seed: int = 12_801
    steps: int = 300
    batch_size: int = 1
    learning_rate: float = 0.01


def train_sequence_output(
    bundle: ModelBundle,
    implant: SequenceNeuronImplantMLP,
    examples: list[SemanticPromptExample],
    *,
    config: SequenceOutputTrainConfig,
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
    for training_step in range(config.steps):
        batch = [rng.choice(positives) for _ in range(config.batch_size)]
        prompts = [
            chat_prompt_ids(bundle.tokenizer, example.prompt) for example in batch
        ]
        targets = [
            answer_token_ids(bundle.tokenizer, example.answer or "")
            for example in batch
        ]
        sequences = [
            prompt + target[:-1]
            for prompt, target in zip(prompts, targets, strict=True)
        ]
        input_ids, attention_mask = padded_batch(
            sequences,
            pad_token_id=bundle.tokenizer.pad_token_id,
            device=bundle.device,
        )
        context, positions_by_row, targets_by_row = sequence_teacher_context(
            batch,
            sequences=sequences,
            prompt_lengths=[len(prompt) for prompt in prompts],
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
        logits: list[torch.Tensor] = []
        target_ids: list[int] = []
        for row, (positions, row_targets) in enumerate(
            zip(positions_by_row, targets_by_row, strict=True)
        ):
            logits.append(bundle.model.lm_head(hidden[row, positions]))
            target_ids.extend(row_targets)
        loss = nn.functional.cross_entropy(
            torch.cat(logits).float(),
            torch.tensor(target_ids, dtype=torch.long, device=bundle.device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if training_step == 0:
            initial_loss = float(loss.detach().cpu())
        final_loss = float(loss.detach().cpu())
        implant.set_context(None)
        del hidden, logits, loss, input_ids, attention_mask, context
        if bundle.device.type == "mps" and training_step % 25 == 24:
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
def temporarily_uninstalled_sequence_implant(
    bundle: ModelBundle,
    implant: SequenceNeuronImplantMLP,
    *,
    layer_index: int,
):
    if bundle.model.model.layers[layer_index].mlp is not implant:
        raise ValueError("sequence implant is not installed at target layer")
    bundle.model.model.layers[layer_index].mlp = implant.base_mlp
    try:
        yield
    finally:
        bundle.model.model.layers[layer_index].mlp = implant


@torch.inference_mode()
def generate_sequence_implant(
    bundle: ModelBundle,
    implant: SequenceNeuronImplantMLP,
    prompt: str,
    *,
    max_new_tokens: int,
    ablate_result: bool = False,
    latch_route: bool = False,
    preserve_base_when_off: bool = False,
    deterministic_result_step: bool = False,
    latch_operands: bool = False,
    force_route: int | None = None,
) -> dict[str, object]:
    if force_route not in {None, 0, 1}:
        raise ValueError("force_route must be None, 0, or 1")
    if (
        implant.request_router_kind.startswith("user_")
        or implant.request_router_kind == "all_views_silu16"
    ):
        full_ids, prompt_content_mask = chat_prompt_ids_and_content_mask(
            bundle.tokenizer,
            prompt,
        )
    else:
        full_ids = chat_prompt_ids(bundle.tokenizer, prompt)
        prompt_content_mask = None
    generated: list[int] = []
    diagnostics: list[dict[str, object]] = []
    latched_route = force_route
    operand_register: dict[str, list[int] | int | bool] | None = None
    for generation_step in range(max_new_tokens):
        input_ids = torch.tensor(
            [full_ids],
            dtype=torch.long,
            device=bundle.device,
        )
        attention_mask = torch.ones_like(input_ids)
        eligible = torch.zeros_like(input_ids, dtype=torch.bool)
        eligible[:, -1] = True
        request_pool_mask = None
        if prompt_content_mask is not None:
            request_pool_mask = torch.zeros_like(input_ids, dtype=torch.bool)
            request_pool_mask[:, : len(prompt_content_mask)] = torch.tensor(
                [prompt_content_mask],
                dtype=torch.bool,
                device=bundle.device,
            )
        teacher_route = None
        if latched_route is not None:
            teacher_route = torch.full_like(input_ids, -1)
            teacher_route[:, -1] = latched_route
        teacher_step = None
        if deterministic_result_step:
            teacher_step = torch.full_like(input_ids, -1)
            teacher_step[:, -1] = min(
                generation_step,
                implant.layout.step_width - 1,
            )
        register_tensors: dict[str, torch.Tensor | None] = {
            "register_a_digits": None,
            "register_b_digits": None,
            "register_a_lengths": None,
            "register_b_lengths": None,
            "register_valid": None,
        }
        if operand_register is not None:
            register_tensors = {
                "register_a_digits": torch.tensor(
                    [operand_register["a_digits"]],
                    dtype=torch.long,
                    device=bundle.device,
                ),
                "register_b_digits": torch.tensor(
                    [operand_register["b_digits"]],
                    dtype=torch.long,
                    device=bundle.device,
                ),
                "register_a_lengths": torch.tensor(
                    [operand_register["a_length"]],
                    dtype=torch.long,
                    device=bundle.device,
                ),
                "register_b_lengths": torch.tensor(
                    [operand_register["b_length"]],
                    dtype=torch.long,
                    device=bundle.device,
                ),
                "register_valid": torch.tensor(
                    [operand_register["valid"]],
                    dtype=torch.bool,
                    device=bundle.device,
                ),
            }
        context = SequenceImplantContext(
            eligible_mask=eligible,
            sequence_mask=attention_mask.to(torch.bool),
            request_pool_mask=request_pool_mask,
            teacher_route=teacher_route,
            teacher_step=teacher_step,
            ablate_result=ablate_result,
            preserve_base_when_off=preserve_base_when_off,
            capture_diagnostics=True,
            **register_tensors,
        )
        implant.set_context(context)
        hidden = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state[:, -1]
        next_token = bundle.model.lm_head(hidden).argmax(dim=-1)
        token_id = int(next_token.item())
        generated.append(token_id)
        full_ids.append(token_id)
        row: dict[str, object] = {"generation_step": generation_step}
        for key, value in context.diagnostics.items():
            row[key] = value.tolist() if isinstance(value, torch.Tensor) else value
        if latch_route and latched_route is None:
            predicted_route = row.get("route", [0])
            latched_route = int(predicted_route[0])
        if (
            latch_operands
            and generation_step == 0
            and row.get("route_active") == [True]
            and row.get("operands_valid") == [True]
        ):
            operand_register = {
                "a_digits": row["a_digits"][0],
                "b_digits": row["b_digits"][0],
                "a_length": row["a_lengths"][0],
                "b_length": row["b_lengths"][0],
                "valid": True,
            }
        row["operand_register_active"] = operand_register is not None
        row["token_id"] = token_id
        row["token_text"] = bundle.tokenizer.decode([token_id])
        diagnostics.append(row)
        del (
            input_ids,
            attention_mask,
            hidden,
            next_token,
            context,
        )
        if request_pool_mask is not None:
            del request_pool_mask
        if bundle.device.type == "mps":
            torch.mps.empty_cache()
        if token_id == bundle.tokenizer.eos_token_id:
            break
    implant.set_context(None)
    return {
        "generated_token_ids": generated,
        "generated_text": bundle.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip(),
        "steps": diagnostics,
    }


@torch.inference_mode()
def generate_untouched_sequence(
    bundle: ModelBundle,
    implant: SequenceNeuronImplantMLP,
    prompt: str,
    *,
    layer_index: int,
    max_new_tokens: int,
) -> dict[str, object]:
    with temporarily_uninstalled_sequence_implant(
        bundle,
        implant,
        layer_index=layer_index,
    ):
        full_ids = chat_prompt_ids(bundle.tokenizer, prompt)
        generated: list[int] = []
        for _ in range(max_new_tokens):
            input_ids = torch.tensor(
                [full_ids],
                dtype=torch.long,
                device=bundle.device,
            )
            attention_mask = torch.ones_like(input_ids)
            hidden = bundle.model.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            ).last_hidden_state[:, -1]
            next_token = bundle.model.lm_head(hidden).argmax(dim=-1)
            token_id = int(next_token.item())
            generated.append(token_id)
            full_ids.append(token_id)
            del input_ids, attention_mask, hidden, next_token
            if bundle.device.type == "mps":
                torch.mps.empty_cache()
            if token_id == bundle.tokenizer.eos_token_id:
                break
    return {
        "generated_token_ids": generated,
        "generated_text": bundle.tokenizer.decode(
            generated,
            skip_special_tokens=True,
        ).strip(),
    }
