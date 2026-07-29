from __future__ import annotations

import time
from dataclasses import asdict, dataclass

import torch
from torch import nn

from neural_firmware.phase11_routing import (
    RequestRouteFeatureSet,
    RequestRouterTrainConfig,
    evaluate_request_router,
    train_request_router,
)
from neural_firmware.phase7_sequence_training import balanced_sample
from neural_firmware.phase7_training import (
    select_route_threshold,
    set_phase7_seed,
)

PHASE12_ROUTER_CONDITIONS = {
    "last_linear": ("last",),
    "last_user_linear": ("last", "user_mean"),
    "all_views_linear": (
        "last",
        "sequence_mean",
        "user_mean",
        "user_tail_mean",
    ),
    "all_views_silu16": (
        "last",
        "sequence_mean",
        "user_mean",
        "user_tail_mean",
    ),
}


@dataclass(frozen=True)
class SiluRouterState:
    down: torch.Tensor
    output: torch.Tensor

    @property
    def trainable_parameters(self) -> int:
        return self.down.numel() + self.output.numel()

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {"down": self.down, "output": self.output}


@dataclass(frozen=True)
class SiluRouterTrainConfig:
    seed: int
    bottleneck_width: int = 16
    steps: int = 1_500
    batch_size: int = 256
    learning_rate: float = 0.0005
    route_temperature: float = 2.0
    maximum_calibration_false_positive_rate: float = 0.01


def concatenate_request_views(
    features: dict[str, RequestRouteFeatureSet],
    kinds: tuple[str, ...],
) -> RequestRouteFeatureSet:
    if not kinds:
        raise ValueError("at least one request view is required")
    selected = [features[kind] for kind in kinds]
    targets = selected[0].targets
    if any(
        not torch.equal(feature.targets, targets)
        for feature in selected[1:]
    ):
        raise ValueError("request views have different targets")
    return RequestRouteFeatureSet(
        hidden=torch.cat(
            [feature.hidden.float() for feature in selected],
            dim=-1,
        ),
        targets=targets.clone(),
    )


def subset_request_features(
    features: RequestRouteFeatureSet,
    indices: torch.Tensor,
) -> RequestRouteFeatureSet:
    return RequestRouteFeatureSet(
        hidden=features.hidden[indices],
        targets=features.targets[indices],
    )


def combine_request_feature_sets(
    *features: RequestRouteFeatureSet,
) -> RequestRouteFeatureSet:
    if not features:
        raise ValueError("at least one feature set is required")
    width = features[0].hidden.shape[1]
    if any(feature.hidden.shape[1] != width for feature in features):
        raise ValueError("request feature widths differ")
    return RequestRouteFeatureSet(
        hidden=torch.cat([feature.hidden for feature in features]),
        targets=torch.cat([feature.targets for feature in features]),
    )


def repeated_initial_rows(
    source_rows: torch.Tensor,
    *,
    views: int,
) -> torch.Tensor:
    if source_rows.shape[0] != 2:
        raise ValueError("source route rows must contain two classes")
    if views < 1:
        raise ValueError("view count must be positive")
    return torch.cat([source_rows.float() / views] * views, dim=-1)


def silu_router_probabilities(
    state: SiluRouterState,
    features: RequestRouteFeatureSet,
    *,
    temperature: float,
) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError("route temperature must be positive")
    with torch.inference_mode():
        hidden = nn.functional.linear(
            features.hidden.float(),
            state.down.float(),
        )
        hidden = nn.functional.silu(hidden)
        logits = nn.functional.linear(hidden, state.output.float())
        return (logits / temperature).softmax(dim=-1)[..., 1].cpu()


def evaluate_silu_router(
    state: SiluRouterState,
    features: RequestRouteFeatureSet,
    *,
    threshold: float,
    temperature: float,
) -> dict[str, object]:
    probabilities = silu_router_probabilities(
        state,
        features,
        temperature=temperature,
    )
    targets = features.targets.to(torch.bool)
    predictions = probabilities >= threshold
    positives = targets
    negatives = ~targets
    return {
        "rows": features.rows,
        "positive_rows": int(positives.sum()),
        "negative_rows": int(negatives.sum()),
        "threshold": float(threshold),
        "true_positive_rate": float(
            predictions[positives].float().mean()
        ),
        "false_positive_rate": float(
            predictions[negatives].float().mean()
        ),
        "accuracy": float((predictions == targets).float().mean()),
        "positive_probability_min": float(probabilities[positives].min()),
        "positive_probability_median": float(
            probabilities[positives].median()
        ),
        "negative_probability_median": float(
            probabilities[negatives].median()
        ),
        "negative_probability_max": float(probabilities[negatives].max()),
        "predictions": predictions,
        "probabilities": probabilities,
    }


def train_silu_router(
    train: RequestRouteFeatureSet,
    calibration: RequestRouteFeatureSet,
    *,
    device: torch.device,
    config: SiluRouterTrainConfig,
) -> tuple[SiluRouterState, dict[str, object], dict[str, object]]:
    if train.hidden.shape[1] != calibration.hidden.shape[1]:
        raise ValueError("training and calibration widths differ")
    if config.bottleneck_width < 1:
        raise ValueError("bottleneck width must be positive")
    set_phase7_seed(config.seed)
    down = nn.Parameter(
        torch.empty(
            config.bottleneck_width,
            train.hidden.shape[1],
            device=device,
        )
    )
    output = nn.Parameter(
        torch.empty(2, config.bottleneck_width, device=device)
    )
    nn.init.normal_(down, std=train.hidden.shape[1] ** -0.5)
    nn.init.normal_(output, std=config.bottleneck_width**-0.5)
    optimizer = torch.optim.AdamW(
        [down, output],
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    generator = torch.Generator().manual_seed(config.seed)
    positives = torch.where(train.targets == 1)[0]
    negatives = torch.where(train.targets == 0)[0]
    initial_loss = float("nan")
    final_loss = float("nan")
    started = time.perf_counter()
    for step in range(config.steps):
        indices = balanced_sample(
            positives,
            negatives,
            count=config.batch_size,
            generator=generator,
        )
        hidden = nn.functional.linear(
            train.hidden[indices].to(device).float(),
            down,
        )
        hidden = nn.functional.silu(hidden)
        logits = nn.functional.linear(hidden, output)
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
    state = SiluRouterState(
        down=down.detach().cpu(),
        output=output.detach().cpu(),
    )
    calibration_probabilities = silu_router_probabilities(
        state,
        calibration,
        temperature=config.route_temperature,
    )
    threshold = select_route_threshold(
        calibration_probabilities,
        calibration.targets,
        maximum_false_positive_rate=(
            config.maximum_calibration_false_positive_rate
        ),
    )
    calibration_metrics = evaluate_silu_router(
        state,
        calibration,
        threshold=float(threshold["threshold"]),
        temperature=config.route_temperature,
    )
    compact_calibration = {
        key: value
        for key, value in calibration_metrics.items()
        if key not in {"predictions", "probabilities"}
    }
    return (
        state,
        {
            "config": asdict(config),
            "initial_loss": initial_loss,
            "final_loss": final_loss,
            "wall_time_seconds": time.perf_counter() - started,
            "trainable_parameters": state.trainable_parameters,
        },
        compact_calibration,
    )


def train_phase12_condition(
    condition: str,
    initial_rows: torch.Tensor,
    train: RequestRouteFeatureSet,
    calibration: RequestRouteFeatureSet,
    *,
    device: torch.device,
    seed: int,
    steps: int = 1_500,
    maximum_calibration_false_positive_rate: float = 0.01,
) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    if condition == "all_views_silu16":
        state, training, calibrated = train_silu_router(
            train,
            calibration,
            device=device,
            config=SiluRouterTrainConfig(
                seed=seed,
                steps=steps,
                maximum_calibration_false_positive_rate=(
                    maximum_calibration_false_positive_rate
                ),
            ),
        )
        return (
            {"kind": "silu16", **state.state_dict()},
            training,
            calibrated,
        )
    rows, training, calibrated = train_request_router(
        initial_rows,
        train,
        calibration,
        device=device,
        config=RequestRouterTrainConfig(
            seed=seed,
            steps=steps,
            batch_size=256,
            learning_rate=0.0005,
            route_temperature=2.0,
            maximum_calibration_false_positive_rate=(
                maximum_calibration_false_positive_rate
            ),
        ),
    )
    return (
        {"kind": "linear", "rows": rows},
        training,
        calibrated,
    )


def evaluate_phase12_condition(
    state: dict[str, object],
    features: RequestRouteFeatureSet,
    *,
    threshold: float,
    temperature: float = 2.0,
) -> dict[str, object]:
    if state["kind"] == "linear":
        return evaluate_request_router(
            state["rows"],
            features,
            threshold=threshold,
            temperature=temperature,
        )
    if state["kind"] == "silu16":
        return evaluate_silu_router(
            SiluRouterState(
                down=state["down"],
                output=state["output"],
            ),
            features,
            threshold=threshold,
            temperature=temperature,
        )
    raise ValueError(f"unknown Phase 12 router state: {state['kind']}")
