from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from torch import nn

from neural_firmware.phase7_sequence_implant import (
    SequenceImplantLayout,
    SequenceNeuronImplantMLP,
    install_sequence_neuron_implant,
)
from neural_firmware.phase7_sequence_training import (
    FirstStepRouteFeatureSet,
    RouteRowTrainConfig,
    SequenceFeatureSet,
    SequenceInterfaceTrainConfig,
    balanced_sample,
    train_sequence_interface,
)
from neural_firmware.phase7_training import (
    select_route_threshold,
    set_phase7_seed,
)
from neural_firmware.phase9_training import (
    architectural_learned_parameter_count,
    concatenate_route_features,
    concatenate_sequence_features,
)
from neural_firmware.pretrained_training import ModelBundle


@dataclass(frozen=True)
class Phase10Condition:
    name: str
    interface_kind: str
    representation_rank: int
    adapt_base_mlp: bool = False


PHASE10_CONDITIONS = (
    Phase10Condition("linear", "linear", 0),
    Phase10Condition("nonlinear", "bottleneck_silu", 0),
    Phase10Condition("linear_representation", "linear", 4),
    Phase10Condition(
        "nonlinear_representation",
        "bottleneck_silu",
        4,
    ),
)


def phase10_condition(name: str) -> Phase10Condition:
    matches = [condition for condition in PHASE10_CONDITIONS if condition.name == name]
    if len(matches) != 1:
        raise ValueError(f"unknown Phase 10 condition: {name}")
    return matches[0]


def install_phase10_implant(
    bundle: ModelBundle,
    source_checkpoint: dict[str, object],
    *,
    condition: Phase10Condition,
    seed: int,
) -> SequenceNeuronImplantMLP:
    set_phase7_seed(seed)
    layout = SequenceImplantLayout(**source_checkpoint["layout"])
    implant = install_sequence_neuron_implant(
        bundle.model,
        layer_index=int(source_checkpoint["layer_index"]),
        selected_indices=source_checkpoint["selected_indices"],
        layout=layout,
        output_strength=float(source_checkpoint["output_strength"]),
        route_threshold=float(source_checkpoint["route_threshold"]),
        route_temperature=2.0,
        digit_threshold=float(source_checkpoint["digit_threshold"]),
        interface_kind=condition.interface_kind,
        representation_rank=condition.representation_rank,
        adapt_base_mlp=condition.adapt_base_mlp,
    )
    source_rows = source_checkpoint["input_rows"].to(bundle.device).float()
    with torch.no_grad():
        if condition.interface_kind == "bottleneck_silu":
            # The fixed projection drops 16 of 2,048 input coordinates. Near
            # zero, 2*SiLU(x) approximates x, giving a source-matched warm
            # start while keeping the learned input budget exact.
            implant.bottleneck_rows.copy_(
                source_rows[
                    ...,
                    : implant.bottleneck_rows.shape[1],
                ].to(implant.bottleneck_rows.dtype)
            )
            implant.bottleneck_mix.copy_(
                2.0
                * torch.eye(
                    implant.layout.input_width,
                    device=bundle.device,
                    dtype=implant.bottleneck_mix.dtype,
                )
            )
            implant.input_rows.copy_(source_rows.to(implant.input_rows.dtype))
        else:
            implant.input_rows.copy_(source_rows.to(implant.input_rows.dtype))
        implant.result_columns.copy_(
            source_checkpoint["result_columns"].to(bundle.device)
        )
    return implant


def train_phase10_condition(
    bundle: ModelBundle,
    source_checkpoint: dict[str, object],
    *,
    condition: Phase10Condition,
    original_sequence: SequenceFeatureSet,
    original_route: FirstStepRouteFeatureSet,
    hard_sequence: SequenceFeatureSet,
    hard_route: FirstStepRouteFeatureSet,
    development_sequence: SequenceFeatureSet,
    development_route: FirstStepRouteFeatureSet,
    seed: int,
    steps: int,
    learning_rate: float,
    route_steps: int,
    route_learning_rate: float,
    digit_threshold: float,
) -> tuple[SequenceNeuronImplantMLP, dict[str, object]]:
    implant = install_phase10_implant(
        bundle,
        source_checkpoint,
        condition=condition,
        seed=seed,
    )
    result_columns_before = implant.result_columns.detach().cpu().clone()
    training, development = train_sequence_interface(
        implant,
        concatenate_sequence_features((original_sequence, hard_sequence)),
        development_sequence,
        device=bundle.device,
        config=SequenceInterfaceTrainConfig(
            seed=seed,
            steps=steps,
            batch_size=256,
            learning_rate=learning_rate,
            route_loss_weight=1.0,
            role_loss_weight=1.0,
            digit_loss_weight=1.0,
            step_loss_weight=0.0,
        ),
    )
    route_training, route_development = harden_phase10_route(
        implant,
        concatenate_route_features((original_route, hard_route)),
        development_route,
        device=bundle.device,
        config=RouteRowTrainConfig(
            seed=seed + 300,
            steps=route_steps,
            batch_size=256,
            learning_rate=route_learning_rate,
            maximum_development_false_positive_rate=0.01,
        ),
    )
    implant.digit_threshold = digit_threshold
    result_columns_after = implant.result_columns.detach().cpu()
    if not torch.equal(result_columns_before, result_columns_after):
        raise AssertionError("Phase 10 modified the frozen result decoder")
    return implant, {
        "condition": asdict(condition),
        "training": training,
        "development": development,
        "route_training": route_training,
        "route_development": route_development,
        "input_interface_parameters": (
            implant.bottleneck_rows.numel()
            + implant.bottleneck_mix.numel()
            if implant.interface_kind == "bottleneck_silu"
            else implant.input_rows.numel()
        ),
        "representation_parameters": (
            0
            if implant.representation_rank == 0
            else (
                implant.representation_down.numel()
                + implant.representation_up.numel()
            )
        ),
        "adapt_base_mlp": implant.adapt_base_mlp,
        "architectural_learned_parameters": (
            architectural_learned_parameter_count(implant)
        ),
        "calculator_learned_parameters": (
            implant.calculator.trainable_parameter_count
        ),
        "result_decoder_unchanged": True,
    }


def _phase10_route_features(
    implant: SequenceNeuronImplantMLP,
    hidden: torch.Tensor,
) -> torch.Tensor:
    adapted = implant.adapted_hidden(hidden)
    if implant.interface_kind == "bottleneck_silu":
        return nn.functional.silu(
            nn.functional.linear(
                adapted.float()[..., : implant.bottleneck_rows.shape[1]],
                implant.bottleneck_rows.float(),
            )
        )
    if implant.interface_kind == "linear":
        return adapted.float()
    raise ValueError(
        "Phase 10 route hardening supports linear and bottleneck_silu"
    )


def _phase10_route_rows(
    implant: SequenceNeuronImplantMLP,
) -> torch.Tensor:
    if implant.interface_kind == "bottleneck_silu":
        return implant.bottleneck_mix.detach()[:2]
    if implant.interface_kind == "linear":
        return implant.input_rows.detach()[:2]
    raise ValueError(
        "Phase 10 route hardening supports linear and bottleneck_silu"
    )


def harden_phase10_route(
    implant: SequenceNeuronImplantMLP,
    train: FirstStepRouteFeatureSet,
    development: FirstStepRouteFeatureSet,
    *,
    device: torch.device,
    config: RouteRowTrainConfig,
) -> tuple[dict[str, object], dict[str, object]]:
    if config.steps < 0:
        raise ValueError("route hardening steps cannot be negative")
    set_phase7_seed(config.seed)
    route_rows = nn.Parameter(_phase10_route_rows(implant).clone().to(device))
    optimizer = torch.optim.AdamW(
        [route_rows],
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(config.seed)
    positive = torch.where(train.targets == 1)[0]
    negative = torch.where(train.targets == 0)[0]
    initial_loss: float | None = None
    final_loss: float | None = None
    for step in range(config.steps):
        indices = balanced_sample(
            positive,
            negative,
            count=config.batch_size,
            generator=generator,
        )
        with torch.no_grad():
            route_features = _phase10_route_features(
                implant,
                train.hidden[indices].to(device),
            )
        logits = nn.functional.linear(route_features, route_rows.float())
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
    trained = route_rows.detach()
    with torch.no_grad():
        if implant.interface_kind == "bottleneck_silu":
            implant.bottleneck_mix[:2].copy_(
                trained.to(implant.bottleneck_mix.dtype)
            )
        else:
            implant.input_rows[:2].copy_(
                trained.to(implant.input_rows.dtype)
            )
        development_features = _phase10_route_features(
            implant,
            development.hidden.to(device),
        )
        probabilities = nn.functional.linear(
            development_features,
            trained.float(),
        )
        probabilities = (
            probabilities / implant.route_temperature
        ).softmax(dim=-1)[..., 1]
    threshold = select_route_threshold(
        probabilities.cpu(),
        development.targets,
        maximum_false_positive_rate=(
            config.maximum_development_false_positive_rate
        ),
    )
    implant.route_threshold = float(threshold["threshold"])
    predictions = probabilities.cpu() >= implant.route_threshold
    targets = development.targets.to(torch.bool)
    return (
        {
            "config": asdict(config),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "trainable_parameters": trained.numel(),
        },
        {
            "rows": development.rows,
            "positive_rows": int(targets.sum()),
            "negative_rows": int((~targets).sum()),
            "threshold": threshold,
            "true_positive_rate": float(
                predictions[targets].float().mean()
            ),
            "false_positive_rate": float(
                predictions[~targets].float().mean()
            ),
        },
    )


def phase10_checkpoint_state(
    implant: SequenceNeuronImplantMLP,
    source_checkpoint: dict[str, object],
) -> dict[str, object]:
    state: dict[str, object] = {
        **source_checkpoint,
        "interface_kind": implant.interface_kind,
        "representation_rank": implant.representation_rank,
        "adapt_base_mlp": implant.adapt_base_mlp,
        "route_threshold": implant.route_threshold,
        "route_temperature": implant.route_temperature,
        "digit_threshold": implant.digit_threshold,
        "input_rows": implant.input_rows.detach().cpu(),
        "result_columns": implant.result_columns.detach().cpu(),
    }
    if implant.representation_rank:
        state["representation_down"] = implant.representation_down.detach().cpu()
        state["representation_up"] = implant.representation_up.detach().cpu()
    if implant.interface_kind == "bottleneck_silu":
        state["bottleneck_rows"] = implant.bottleneck_rows.detach().cpu()
        state["bottleneck_mix"] = implant.bottleneck_mix.detach().cpu()
    return state
