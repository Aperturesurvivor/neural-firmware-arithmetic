from __future__ import annotations

from collections.abc import Iterable

import torch

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
    evaluate_sequence_interface,
    train_route_rows,
    train_sequence_interface,
)
from neural_firmware.pretrained_training import ModelBundle


def concatenate_sequence_features(
    features: Iterable[SequenceFeatureSet],
) -> SequenceFeatureSet:
    values = list(features)
    if not values:
        raise ValueError("at least one sequence feature set is required")
    return SequenceFeatureSet(
        hidden=torch.cat([value.hidden for value in values]),
        route_targets=torch.cat([value.route_targets for value in values]),
        role_targets=torch.cat([value.role_targets for value in values]),
        digit_targets=torch.cat([value.digit_targets for value in values]),
        step_targets=torch.cat([value.step_targets for value in values]),
    )


def concatenate_route_features(
    features: Iterable[FirstStepRouteFeatureSet],
) -> FirstStepRouteFeatureSet:
    values = list(features)
    if not values:
        raise ValueError("at least one route feature set is required")
    return FirstStepRouteFeatureSet(
        hidden=torch.cat([value.hidden for value in values]),
        targets=torch.cat([value.targets for value in values]),
    )


def install_checkpoint_implant(
    bundle: ModelBundle,
    checkpoint: dict[str, object],
) -> SequenceNeuronImplantMLP:
    layout = SequenceImplantLayout(**checkpoint["layout"])
    implant = install_sequence_neuron_implant(
        bundle.model,
        layer_index=int(checkpoint["layer_index"]),
        selected_indices=checkpoint["selected_indices"],
        layout=layout,
        output_strength=float(checkpoint["output_strength"]),
        route_threshold=float(checkpoint["route_threshold"]),
        route_temperature=float(checkpoint.get("route_temperature", 1.0)),
        digit_threshold=float(checkpoint["digit_threshold"]),
        interface_kind=str(checkpoint.get("interface_kind", "linear")),
        representation_rank=int(checkpoint.get("representation_rank", 0)),
        adapt_base_mlp=bool(checkpoint.get("adapt_base_mlp", True)),
        request_router_kind=str(
            checkpoint.get("request_router_kind", "interface")
        ),
        request_route_threshold=float(
            checkpoint.get(
                "request_route_threshold",
                checkpoint["route_threshold"],
            )
        ),
        request_route_temperature=float(
            checkpoint.get("request_route_temperature", 2.0)
        ),
        request_tail_tokens=int(checkpoint.get("request_tail_tokens", 8)),
    )
    with torch.no_grad():
        implant.input_rows.copy_(checkpoint["input_rows"].to(bundle.device))
        if "gate_rows" in checkpoint:
            implant.gate_rows.copy_(checkpoint["gate_rows"].to(bundle.device))
        if implant.interface_kind == "bottleneck_silu":
            implant.bottleneck_rows.copy_(
                checkpoint["bottleneck_rows"].to(bundle.device)
            )
            implant.bottleneck_mix.copy_(
                checkpoint["bottleneck_mix"].to(bundle.device)
            )
        implant.result_columns.copy_(
            checkpoint["result_columns"].to(bundle.device)
        )
        if implant.representation_rank:
            implant.representation_down.copy_(
                checkpoint["representation_down"].to(bundle.device)
            )
            implant.representation_up.copy_(
                checkpoint["representation_up"].to(bundle.device)
            )
        if implant.request_router_kind == "all_views_silu16":
            implant.request_route_down.copy_(
                checkpoint["request_route_down"].to(bundle.device)
            )
            implant.request_route_output.copy_(
                checkpoint["request_route_output"].to(bundle.device)
            )
        elif implant.request_router_kind != "interface":
            implant.request_route_rows.copy_(
                checkpoint["request_route_rows"].to(bundle.device)
            )
    return implant


def architectural_learned_parameter_count(
    implant: SequenceNeuronImplantMLP,
) -> int:
    if implant.interface_kind == "bottleneck_silu":
        count = (
            implant.bottleneck_rows.numel()
            + implant.bottleneck_mix.numel()
            + implant.result_columns.numel()
        )
    else:
        count = implant.input_rows.numel() + implant.result_columns.numel()
    if implant.use_swiglu_interface:
        count += implant.gate_rows.numel()
    if implant.representation_rank:
        count += (
            implant.representation_down.numel()
            + implant.representation_up.numel()
        )
    if implant.request_router_kind == "all_views_silu16":
        count += (
            implant.request_route_down.numel()
            + implant.request_route_output.numel()
        )
    elif implant.request_router_kind != "interface":
        count += implant.request_route_rows.numel()
    return count


def compact_interface_metrics(metrics: dict[str, object]) -> dict[str, object]:
    return {
        key: value
        for key, value in metrics.items()
        if key not in {"route_probabilities", "route_targets"}
    }


def continue_phase9_interface(
    bundle: ModelBundle,
    source_checkpoint: dict[str, object],
    *,
    original_sequence: SequenceFeatureSet,
    original_route: FirstStepRouteFeatureSet,
    condition_sequence: SequenceFeatureSet,
    condition_route: FirstStepRouteFeatureSet,
    development_sequence: SequenceFeatureSet,
    development_route: FirstStepRouteFeatureSet,
    seed: int,
    interface_steps: int,
    interface_learning_rate: float,
    role_loss_weight: float,
    digit_loss_weight: float,
    route_steps: int,
    route_learning_rate: float,
    maximum_development_false_positive_rate: float,
) -> tuple[SequenceNeuronImplantMLP, dict[str, object]]:
    implant = install_checkpoint_implant(bundle, source_checkpoint)
    result_columns_before = implant.result_columns.detach().cpu().clone()
    training_sequence = concatenate_sequence_features(
        (original_sequence, condition_sequence)
    )
    interface_training, interface_development = train_sequence_interface(
        implant,
        training_sequence,
        development_sequence,
        device=bundle.device,
        config=SequenceInterfaceTrainConfig(
            seed=seed,
            steps=interface_steps,
            batch_size=256,
            learning_rate=interface_learning_rate,
            role_loss_weight=role_loss_weight,
            digit_loss_weight=digit_loss_weight,
            step_loss_weight=0.0,
        ),
    )
    route_rows, route_training, route_development = train_route_rows(
        implant.input_rows.detach().cpu()[:2],
        concatenate_route_features((original_route, condition_route)),
        development_route,
        device=bundle.device,
        config=RouteRowTrainConfig(
            seed=seed + 300,
            steps=route_steps,
            batch_size=256,
            learning_rate=route_learning_rate,
            maximum_development_false_positive_rate=(
                maximum_development_false_positive_rate
            ),
        ),
    )
    with torch.no_grad():
        implant.input_rows[:2].copy_(route_rows.to(bundle.device))
    implant.route_threshold = float(route_development["threshold"]["threshold"])
    final_development = compact_interface_metrics(
        evaluate_sequence_interface(
            implant,
            development_sequence,
            device=bundle.device,
        )
    )
    result_columns_after = implant.result_columns.detach().cpu()
    if not torch.equal(result_columns_before, result_columns_after):
        raise AssertionError("Phase 9 modified the frozen result decoder")
    return implant, {
        "interface_training": interface_training,
        "interface_development_before_route_hardening": interface_development,
        "route_training": route_training,
        "route_development": route_development,
        "interface_development_after_route_hardening": final_development,
        "updated_parameters": implant.input_rows.numel(),
        "architectural_learned_parameters": (
            architectural_learned_parameter_count(implant)
        ),
        "result_decoder_unchanged": True,
    }
