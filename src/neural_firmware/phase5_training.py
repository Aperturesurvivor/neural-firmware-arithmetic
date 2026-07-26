from __future__ import annotations

import random
import time
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch import nn

from neural_firmware.phase5_igc import (
    PAD_DIGIT,
    IGCContext,
    IGCDualContext,
    IGCDualInstallation,
    IGCFirmwareLayer,
    IGCInputMapping,
    IGCRouteMode,
)
from neural_firmware.pretrained_data import answer_token_ids, chat_prompt_ids
from neural_firmware.pretrained_training import ModelBundle
from neural_firmware.semantic_data import (
    SemanticPromptExample,
    exact_format_correct,
    mathematical_correct,
)


def set_phase5_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _padded_sequences(
    sequences: list[list[int] | tuple[int, ...]],
    *,
    pad_token_id: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    maximum = max(map(len, sequences))
    input_ids = torch.full(
        (len(sequences), maximum),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros_like(input_ids)
    for row, sequence in enumerate(sequences):
        input_ids[row, : len(sequence)] = torch.tensor(
            sequence,
            dtype=torch.long,
            device=device,
        )
        attention_mask[row, : len(sequence)] = 1
    return input_ids, attention_mask


def operand_targets(
    examples: list[SemanticPromptExample],
    *,
    max_digits: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_targets = torch.full(
        (len(examples), max_digits),
        PAD_DIGIT,
        dtype=torch.long,
    )
    b_targets = torch.full_like(a_targets, PAD_DIGIT)
    for row, example in enumerate(examples):
        if len(example.a) > max_digits or len(example.b) > max_digits:
            raise ValueError("operand exceeds configured IGC maximum")
        a_targets[row, : len(example.a)] = torch.tensor(
            [int(character) for character in example.a],
        )
        b_targets[row, : len(example.b)] = torch.tensor(
            [int(character) for character in example.b],
        )
    return a_targets, b_targets


@dataclass
class IGCFeatureSet:
    hidden: torch.Tensor
    attention_mask: torch.Tensor
    anchor_positions: torch.Tensor
    a_targets: torch.Tensor
    b_targets: torch.Tensor
    route_targets: torch.Tensor

    @property
    def examples(self) -> int:
        return self.hidden.shape[0]


@torch.inference_mode()
def collect_igc_features(
    bundle: ModelBundle,
    examples: list[SemanticPromptExample],
    *,
    depth_after_blocks: int,
    max_digits: int,
    batch_size: int,
) -> IGCFeatureSet:
    encoded = [chat_prompt_ids(bundle.tokenizer, example.prompt) for example in examples]
    maximum = max(map(len, encoded))
    hidden_cache = torch.zeros(
        (len(examples), maximum, bundle.model.config.hidden_size),
        dtype=torch.float16,
    )
    mask_cache = torch.zeros((len(examples), maximum), dtype=torch.long)
    anchors = torch.zeros(len(examples), dtype=torch.long)
    started = time.perf_counter()
    for start in range(0, len(examples), batch_size):
        sequences = encoded[start : start + batch_size]
        input_ids, attention_mask = _padded_sequences(
            sequences,
            pad_token_id=bundle.tokenizer.pad_token_id,
            device=bundle.device,
        )
        outputs = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
            output_hidden_states=True,
        )
        hidden = outputs.hidden_states[depth_after_blocks]
        width = hidden.shape[1]
        stop = start + len(sequences)
        hidden_cache[start:stop, :width] = hidden.float().cpu().half()
        mask_cache[start:stop, :width] = attention_mask.cpu()
        anchors[start:stop] = attention_mask.sum(dim=1).cpu() - 1
    if bundle.device.type == "mps":
        torch.mps.synchronize()
    a_targets, b_targets = operand_targets(examples, max_digits=max_digits)
    route_targets = torch.tensor(
        [int(example.route_label) for example in examples],
        dtype=torch.long,
    )
    feature_set = IGCFeatureSet(
        hidden=hidden_cache,
        attention_mask=mask_cache,
        anchor_positions=anchors,
        a_targets=a_targets,
        b_targets=b_targets,
        route_targets=route_targets,
    )
    feature_set.extraction_seconds = time.perf_counter() - started  # type: ignore[attr-defined]
    return feature_set


@torch.inference_mode()
def collect_pre_norm_route_features(
    bundle: ModelBundle,
    examples: list[SemanticPromptExample],
    *,
    depth_after_blocks: int,
    batch_size: int,
) -> dict[str, torch.Tensor]:
    """Collect the exact residual seen by a wrapper after the selected block."""

    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    started = time.perf_counter()
    layer = bundle.model.model.layers[depth_after_blocks - 1]
    captured: list[torch.Tensor] = []

    def capture_output(
        module: nn.Module,
        inputs: tuple[object, ...],
        output: torch.Tensor,
    ) -> None:
        del module, inputs
        captured.append(output.detach())

    handle = layer.register_forward_hook(capture_output)
    try:
        for start in range(0, len(examples), batch_size):
            batch = examples[start : start + batch_size]
            sequences = [
                chat_prompt_ids(bundle.tokenizer, example.prompt)
                for example in batch
            ]
            input_ids, attention_mask = _padded_sequences(
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
                raise RuntimeError("route feature hook did not fire exactly once")
            final_positions = attention_mask.sum(dim=1) - 1
            rows = torch.arange(len(batch), device=bundle.device)
            features.append(
                captured[0][rows, final_positions].float().cpu()
            )
            labels.append(
                torch.tensor(
                    [example.route_label for example in batch],
                    dtype=torch.float32,
                )
            )
    finally:
        handle.remove()
    return {
        "features": torch.cat(features),
        "labels": torch.cat(labels),
        "examples": torch.tensor(len(examples)),
        "extraction_seconds": torch.tensor(time.perf_counter() - started),
    }


@dataclass(frozen=True)
class IGCInputTrainConfig:
    seed: int
    steps: int
    batch_size: int
    learning_rate: float
    digit_loss_weight: float = 1.0
    route_loss_weight: float = 1.0


def train_igc_input_mapping(
    mapping: IGCInputMapping,
    features: IGCFeatureSet,
    config: IGCInputTrainConfig,
    *,
    device: torch.device,
) -> dict[str, object]:
    set_phase5_seed(config.seed)
    mapping.to(device)
    mapping.train()
    parameters = list(mapping.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    initial_loss = float("nan")
    final_loss = float("nan")
    final_digit_loss = float("nan")
    final_route_loss = float("nan")
    started = time.perf_counter()
    for step in range(config.steps):
        indices = torch.randint(
            features.examples,
            (config.batch_size,),
            generator=generator,
        )
        hidden = features.hidden[indices].to(device).float()
        mask = features.attention_mask[indices].to(device)
        anchors = features.anchor_positions[indices].to(device)
        a_logits, b_logits, route_logits = mapping(hidden, mask, anchors)
        a_targets = features.a_targets[indices].to(device)
        b_targets = features.b_targets[indices].to(device)
        route_targets = features.route_targets[indices].to(device)
        digit_loss = (
            nn.functional.cross_entropy(a_logits.flatten(0, 1), a_targets.flatten())
            + nn.functional.cross_entropy(b_logits.flatten(0, 1), b_targets.flatten())
        ) / 2
        route_loss = nn.functional.cross_entropy(route_logits, route_targets)
        loss = (
            config.digit_loss_weight * digit_loss
            + config.route_loss_weight * route_loss
        )
        if step == 0:
            initial_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
        final_digit_loss = float(digit_loss.item())
        final_route_loss = float(route_loss.item())
    if device.type == "mps":
        torch.mps.synchronize()
    mapping.eval()
    return {
        "config": asdict(config),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "final_digit_loss": final_digit_loss,
        "final_route_loss": final_route_loss,
        "wall_time_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_igc_input_mapping(
    mapping: IGCInputMapping,
    features: IGCFeatureSet,
    examples: list[SemanticPromptExample],
    *,
    device: torch.device,
    threshold: float,
    batch_size: int = 64,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    correct_digit_slots = 0
    total_digit_slots = 0
    correct_nonpad_slots = 0
    total_nonpad_slots = 0
    for start in range(0, features.examples, batch_size):
        stop = min(start + batch_size, features.examples)
        hidden = features.hidden[start:stop].to(device).float()
        mask = features.attention_mask[start:stop].to(device)
        anchors = features.anchor_positions[start:stop].to(device)
        a_logits, b_logits, route_logits = mapping(hidden, mask, anchors)
        a_predictions = a_logits.argmax(dim=-1).cpu()
        b_predictions = b_logits.argmax(dim=-1).cpu()
        probabilities = torch.softmax(route_logits, dim=-1)[:, 1].cpu()
        for local, example in enumerate(examples[start:stop]):
            index = start + local
            a_exact = torch.equal(a_predictions[local], features.a_targets[index])
            b_exact = torch.equal(b_predictions[local], features.b_targets[index])
            for prediction, target in (
                (a_predictions[local], features.a_targets[index]),
                (b_predictions[local], features.b_targets[index]),
            ):
                correct_digit_slots += int((prediction == target).sum().item())
                total_digit_slots += target.numel()
                nonpad = target != PAD_DIGIT
                correct_nonpad_slots += int(
                    ((prediction == target) & nonpad).sum().item()
                )
                total_nonpad_slots += int(nonpad.sum().item())
            probability = float(probabilities[local].item())
            predicted_route = probability >= threshold
            rows.append(
                {
                    "prompt": example.prompt,
                    "family": example.family,
                    "route_label": example.route_label,
                    "route_probability": probability,
                    "route_active": predicted_route,
                    "a_exact": a_exact,
                    "b_exact": b_exact,
                    "registers_exact": a_exact and b_exact,
                }
            )
    positives = [
        row
        for row in rows
        if row.get("route_label", row.get("label")) is True
    ]
    negatives = [
        row
        for row in rows
        if row.get("route_label", row.get("label")) is False
    ]
    true_positive = sum(row["route_active"] is True for row in positives)
    false_positive = sum(row["route_active"] is True for row in negatives)
    return {
        "examples": len(rows),
        "registers_exact": sum(row["registers_exact"] is True for row in rows),
        "register_accuracy": sum(row["registers_exact"] is True for row in rows)
        / len(rows),
        "digit_slot_accuracy": correct_digit_slots / total_digit_slots,
        "nonpad_digit_accuracy": correct_nonpad_slots / total_nonpad_slots,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "true_positive_rate": true_positive / len(positives),
        "false_positive_rate": false_positive / len(negatives),
        "rows": rows,
    }


def select_route_threshold(
    rows: list[dict[str, object]],
    *,
    maximum_false_positive_rate: float = 0.01,
) -> dict[str, float | int]:
    candidates: list[dict[str, float | int]] = []
    positives = [
        row
        for row in rows
        if row.get("route_label", row.get("label")) is True
    ]
    negatives = [
        row
        for row in rows
        if row.get("route_label", row.get("label")) is False
    ]
    for integer in range(50, 100):
        threshold = integer / 100
        true_positive = sum(
            float(row.get("route_probability", row.get("probability", 0.0)))
            >= threshold
            for row in positives
        )
        false_positive = sum(
            float(row.get("route_probability", row.get("probability", 0.0)))
            >= threshold
            for row in negatives
        )
        candidates.append(
            {
                "threshold": threshold,
                "true_positive": true_positive,
                "false_positive": false_positive,
                "true_positive_rate": true_positive / len(positives),
                "false_positive_rate": false_positive / len(negatives),
            }
        )
    eligible = [
        row
        for row in candidates
        if float(row["false_positive_rate"]) <= maximum_false_positive_rate
    ]
    return max(
        eligible if eligible else candidates,
        key=lambda row: (
            float(row["true_positive_rate"]),
            -float(row["false_positive_rate"]),
            -float(row["threshold"]),
        ),
    )


@dataclass(frozen=True)
class IGCOutputTrainConfig:
    seed: int
    steps: int
    batch_size: int
    learning_rate: float


def initialize_igc_output_mapping(
    bundle: ModelBundle,
    wrapper: IGCFirmwareLayer,
) -> None:
    token_ids = [
        bundle.tokenizer.encode(str(digit), add_special_tokens=False)[0]
        for digit in range(10)
    ] + [bundle.tokenizer.eos_token_id]
    with torch.no_grad():
        vectors = bundle.model.lm_head.weight[token_ids].detach().float()
        width = wrapper.unit.output_mapping.output_width
        wrapper.unit.output_mapping.codebook.weight.copy_(
            nn.functional.normalize(vectors[:, :width], dim=-1)
        )


def _make_output_batch(
    bundle: ModelBundle,
    examples: list[SemanticPromptExample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, IGCContext]:
    if any(example.answer is None for example in examples):
        raise ValueError("IGC output training requires positive examples")
    prompt_ids = [
        chat_prompt_ids(bundle.tokenizer, example.prompt) for example in examples
    ]
    targets = [
        answer_token_ids(bundle.tokenizer, example.answer or "")
        for example in examples
    ]
    sequences = [
        prompt + target[:-1]
        for prompt, target in zip(prompt_ids, targets, strict=True)
    ]
    input_ids, attention_mask = _padded_sequences(
        sequences,
        pad_token_id=bundle.tokenizer.pad_token_id,
        device=bundle.device,
    )
    maximum = max(map(len, targets))
    output_positions = torch.zeros(
        (len(examples), maximum),
        dtype=torch.long,
        device=bundle.device,
    )
    symbols = torch.full(
        (len(examples), maximum),
        -1,
        dtype=torch.long,
        device=bundle.device,
    )
    symbol_mask = torch.zeros_like(symbols, dtype=torch.bool)
    target_ids = torch.full(
        (len(examples), maximum),
        bundle.tokenizer.eos_token_id,
        dtype=torch.long,
        device=bundle.device,
    )
    for row, (example, prompt, target) in enumerate(
        zip(examples, prompt_ids, targets, strict=True)
    ):
        answer = example.answer or ""
        row_symbols = [int(character) for character in answer] + [10]
        positions = torch.arange(
            len(prompt) - 1,
            len(prompt) - 1 + len(target),
            device=bundle.device,
        )
        output_positions[row, : len(target)] = positions
        symbols[row, : len(row_symbols)] = torch.tensor(
            row_symbols,
            device=bundle.device,
        )
        symbol_mask[row, : len(row_symbols)] = True
        target_ids[row, : len(target)] = torch.tensor(
            target,
            device=bundle.device,
        )
    context = IGCContext(
        attention_mask=attention_mask,
        anchor_positions=torch.tensor(
            [len(prompt) - 1 for prompt in prompt_ids],
            device=bundle.device,
        ),
        output_positions=output_positions,
        teacher_symbols=symbols,
        teacher_symbol_mask=symbol_mask,
    )
    return input_ids, attention_mask, target_ids, context


def train_igc_output_mapping(
    bundle: ModelBundle,
    wrapper: IGCFirmwareLayer,
    examples: list[SemanticPromptExample],
    config: IGCOutputTrainConfig,
) -> dict[str, object]:
    set_phase5_seed(config.seed)
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    for parameter in wrapper.unit.output_mapping.parameters():
        parameter.requires_grad_(True)
    initialize_igc_output_mapping(bundle, wrapper)
    parameters = list(wrapper.unit.output_mapping.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()
    for step in range(config.steps):
        indices = torch.randint(
            len(examples),
            (config.batch_size,),
            generator=generator,
        ).tolist()
        batch_examples = [examples[index] for index in indices]
        input_ids, attention_mask, target_ids, context = _make_output_batch(
            bundle,
            batch_examples,
        )
        wrapper.set_context(context)
        outputs = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = bundle.model.lm_head(outputs.last_hidden_state)
        mask = context.teacher_symbol_mask
        if mask is None or context.output_positions is None:
            raise RuntimeError("output training context was incomplete")
        batch_grid = torch.arange(len(batch_examples), device=bundle.device)[:, None]
        batch_grid = batch_grid.expand_as(target_ids)
        selected_logits = logits[
            batch_grid[mask],
            context.output_positions[mask],
        ]
        loss = nn.functional.cross_entropy(
            selected_logits.float(),
            target_ids[mask],
        )
        if step == 0:
            initial_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    wrapper.set_context(None)
    if bundle.device.type == "mps":
        torch.mps.synchronize()
    return {
        "config": asdict(config),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "wall_time_seconds": time.perf_counter() - started,
    }


def _igc_result(
    example: SemanticPromptExample,
    generated: list[int],
    text: str,
    latency: float,
    context: IGCContext,
) -> dict[str, object]:
    expected = example.answer
    a_prediction = context.diagnostics.get("a_predictions")
    b_prediction = context.diagnostics.get("b_predictions")
    a_target, b_target = operand_targets(
        [example],
        max_digits=(a_prediction.shape[1] if isinstance(a_prediction, torch.Tensor) else 12),
    )
    registers_exact = None
    if isinstance(a_prediction, torch.Tensor) and isinstance(b_prediction, torch.Tensor):
        registers_exact = bool(
            torch.equal(a_prediction[0], a_target[0])
            and torch.equal(b_prediction[0], b_target[0])
        )
    return {
        "prompt": example.prompt,
        "family": example.family,
        "family_index": example.family_index,
        "split": example.split,
        "a": example.a,
        "b": example.b,
        "expected": expected,
        "generated_text": text,
        "generated_token_ids": generated,
        "mathematical_correct": (
            mathematical_correct(text, expected) if expected is not None else None
        ),
        "exact_format_correct": (
            exact_format_correct(text, expected) if expected is not None else None
        ),
        "route_label": example.route_label,
        "route_probability": (
            float(context.route_probabilities[0].item())
            if context.route_probabilities is not None
            else None
        ),
        "route_active": (
            bool(context.route_active[0].item())
            if context.route_active is not None
            else None
        ),
        "registers_exact": registers_exact,
        "a_predictions": (
            a_prediction[0].tolist()
            if isinstance(a_prediction, torch.Tensor)
            else None
        ),
        "b_predictions": (
            b_prediction[0].tolist()
            if isinstance(b_prediction, torch.Tensor)
            else None
        ),
        "latency_seconds": latency,
    }


@torch.inference_mode()
def generate_igc(
    bundle: ModelBundle,
    wrapper: IGCFirmwareLayer,
    example: SemanticPromptExample,
    *,
    route_mode: IGCRouteMode,
    route_threshold: float,
    max_new_tokens: int,
) -> dict[str, object]:
    prompt = chat_prompt_ids(bundle.tokenizer, example.prompt)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=bundle.device)
    attention_mask = torch.ones_like(input_ids)
    context = IGCContext(
        attention_mask=attention_mask,
        anchor_positions=torch.tensor(
            [len(prompt) - 1],
            device=bundle.device,
        ),
        generation_index=0,
        route_mode=route_mode,
        route_threshold=route_threshold,
    )
    wrapper.set_context(context)
    started = time.perf_counter()
    outputs = bundle.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    hidden = outputs.last_hidden_state[:, -1, :]
    generated: list[int] = []
    for step in range(max_new_tokens):
        logits = bundle.model.lm_head(hidden)
        next_token = logits.argmax(dim=-1)
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id == bundle.tokenizer.eos_token_id:
            break
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=bundle.device),
            ],
            dim=1,
        )
        context.generation_index = step + 1
        outputs = bundle.model.model(
            input_ids=next_token.unsqueeze(0),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        hidden = outputs.last_hidden_state[:, -1, :]
    if bundle.device.type == "mps":
        torch.mps.synchronize()
    latency = time.perf_counter() - started
    wrapper.set_context(None)
    text = bundle.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return _igc_result(example, generated, text, latency, context)


def initialize_dual_igc_output_mapping(
    bundle: ModelBundle,
    installation: IGCDualInstallation,
) -> None:
    token_ids = [
        bundle.tokenizer.encode(str(digit), add_special_tokens=False)[0]
        for digit in range(10)
    ] + [bundle.tokenizer.eos_token_id]
    with torch.no_grad():
        vectors = bundle.model.lm_head.weight[token_ids].detach().float()
        width = installation.final.output_mapping.output_width
        installation.final.output_mapping.codebook.weight.copy_(
            nn.functional.normalize(vectors[:, :width], dim=-1)
        )


def _make_dual_output_batch(
    bundle: ModelBundle,
    examples: list[SemanticPromptExample],
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, IGCDualContext]:
    input_ids, attention_mask, target_ids, context = _make_output_batch(
        bundle,
        examples,
    )
    dual_context = IGCDualContext(
        attention_mask=context.attention_mask,
        anchor_positions=context.anchor_positions,
        output_positions=context.output_positions,
        teacher_symbols=context.teacher_symbols,
        teacher_symbol_mask=context.teacher_symbol_mask,
    )
    return input_ids, attention_mask, target_ids, dual_context


def train_dual_igc_output_mapping(
    bundle: ModelBundle,
    installation: IGCDualInstallation,
    examples: list[SemanticPromptExample],
    config: IGCOutputTrainConfig,
) -> dict[str, object]:
    set_phase5_seed(config.seed)
    for parameter in bundle.model.parameters():
        parameter.requires_grad_(False)
    for parameter in installation.final.output_mapping.parameters():
        parameter.requires_grad_(True)
    initialize_dual_igc_output_mapping(bundle, installation)
    parameters = list(installation.final.output_mapping.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate)
    generator = torch.Generator().manual_seed(config.seed)
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()
    for step in range(config.steps):
        indices = torch.randint(
            len(examples),
            (config.batch_size,),
            generator=generator,
        ).tolist()
        batch_examples = [examples[index] for index in indices]
        input_ids, attention_mask, target_ids, context = _make_dual_output_batch(
            bundle,
            batch_examples,
        )
        installation.set_context(context)
        outputs = bundle.model.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        logits = bundle.model.lm_head(outputs.last_hidden_state)
        mask = context.teacher_symbol_mask
        if mask is None or context.output_positions is None:
            raise RuntimeError("dual IGC output training context was incomplete")
        batch_grid = torch.arange(len(batch_examples), device=bundle.device)[:, None]
        batch_grid = batch_grid.expand_as(target_ids)
        selected_logits = logits[
            batch_grid[mask],
            context.output_positions[mask],
        ]
        loss = nn.functional.cross_entropy(
            selected_logits.float(),
            target_ids[mask],
        )
        if step == 0:
            initial_loss = float(loss.item())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.item())
    installation.set_context(None)
    if bundle.device.type == "mps":
        torch.mps.synchronize()
    return {
        "config": asdict(config),
        "trainable_parameters": sum(parameter.numel() for parameter in parameters),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "wall_time_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def generate_dual_igc(
    bundle: ModelBundle,
    installation: IGCDualInstallation,
    example: SemanticPromptExample,
    *,
    route_mode: IGCRouteMode,
    route_threshold: float,
    max_new_tokens: int,
) -> dict[str, object]:
    prompt = chat_prompt_ids(bundle.tokenizer, example.prompt)
    input_ids = torch.tensor([prompt], dtype=torch.long, device=bundle.device)
    attention_mask = torch.ones_like(input_ids)
    context = IGCDualContext(
        attention_mask=attention_mask,
        anchor_positions=torch.tensor(
            [len(prompt) - 1],
            device=bundle.device,
        ),
        generation_index=0,
        route_mode=route_mode,
        route_threshold=route_threshold,
    )
    installation.set_context(context)
    started = time.perf_counter()
    outputs = bundle.model.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=True,
    )
    past_key_values = outputs.past_key_values
    hidden = outputs.last_hidden_state[:, -1, :]
    generated: list[int] = []
    for step in range(max_new_tokens):
        logits = bundle.model.lm_head(hidden)
        next_token = logits.argmax(dim=-1)
        token_id = int(next_token.item())
        generated.append(token_id)
        if token_id == bundle.tokenizer.eos_token_id:
            break
        attention_mask = torch.cat(
            [
                attention_mask,
                torch.ones((1, 1), dtype=torch.long, device=bundle.device),
            ],
            dim=1,
        )
        context.generation_index = step + 1
        outputs = bundle.model.model(
            input_ids=next_token.unsqueeze(0),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        hidden = outputs.last_hidden_state[:, -1, :]
    if bundle.device.type == "mps":
        torch.mps.synchronize()
    latency = time.perf_counter() - started
    installation.set_context(None)
    text = bundle.tokenizer.decode(generated, skip_special_tokens=True).strip()
    return _igc_result(example, generated, text, latency, context)
