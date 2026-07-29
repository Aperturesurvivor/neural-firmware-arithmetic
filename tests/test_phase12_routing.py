from __future__ import annotations

import torch

from neural_firmware.phase11_routing import RequestRouteFeatureSet
from neural_firmware.phase12_routing import (
    SiluRouterTrainConfig,
    combine_request_feature_sets,
    concatenate_request_views,
    evaluate_silu_router,
    repeated_initial_rows,
    subset_request_features,
    train_silu_router,
)


def feature_set(hidden: torch.Tensor) -> RequestRouteFeatureSet:
    return RequestRouteFeatureSet(
        hidden=hidden,
        targets=torch.tensor([1, 1, 0, 0]),
    )


def test_phase12_feature_composition() -> None:
    first = feature_set(torch.arange(8, dtype=torch.float32).reshape(4, 2))
    second = feature_set(torch.ones(4, 3))
    combined = concatenate_request_views(
        {"first": first, "second": second},
        ("first", "second"),
    )
    assert combined.hidden.shape == (4, 5)
    subset = subset_request_features(combined, torch.tensor([0, 3]))
    assert subset.hidden.shape == (2, 5)
    doubled = combine_request_feature_sets(subset, subset)
    assert doubled.hidden.shape == (4, 5)
    assert torch.equal(doubled.targets, torch.tensor([1, 0, 1, 0]))


def test_repeated_initial_rows_preserve_average_logits() -> None:
    rows = torch.randn(2, 4)
    hidden = torch.randn(3, 4)
    repeated = repeated_initial_rows(rows, views=2)
    concatenated = torch.cat([hidden, hidden], dim=-1)
    assert torch.allclose(
        torch.nn.functional.linear(concatenated, repeated),
        torch.nn.functional.linear(hidden, rows),
    )


def test_silu_router_separates_simple_features() -> None:
    features = feature_set(
        torch.tensor(
            [
                [2.0, 0.0],
                [1.0, 0.0],
                [-1.0, 0.0],
                [-2.0, 0.0],
            ]
        )
    )
    state, training, calibration = train_silu_router(
        features,
        features,
        device=torch.device("cpu"),
        config=SiluRouterTrainConfig(
            seed=12,
            bottleneck_width=4,
            steps=150,
            batch_size=4,
            learning_rate=0.05,
            route_temperature=1.0,
            maximum_calibration_false_positive_rate=0.0,
        ),
    )
    metrics = evaluate_silu_router(
        state,
        features,
        threshold=calibration["threshold"],
        temperature=1.0,
    )
    assert training["trainable_parameters"] == 16
    assert metrics["true_positive_rate"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
