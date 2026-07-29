from __future__ import annotations

import torch

from neural_firmware.phase11_routing import (
    RequestRouteFeatureSet,
    RequestRouterTrainConfig,
    adapt_request_hidden,
    phase11_checkpoint_state,
    pool_request_hidden,
    train_request_router,
)


def test_request_pooling_conditions_have_the_same_width() -> None:
    hidden = torch.arange(2 * 5 * 4, dtype=torch.float32).reshape(2, 5, 4)
    sequence_mask = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, True, False],
        ]
    )
    user_mask = torch.tensor(
        [
            [False, True, True, True, False],
            [False, True, True, False, False],
        ]
    )
    for kind in ("last", "sequence_mean", "user_mean", "user_tail_mean"):
        pooled = pool_request_hidden(
            hidden,
            kind=kind,
            sequence_mask=sequence_mask,
            user_mask=user_mask,
            tail_tokens=2,
        )
        assert pooled.shape == (2, 4)
    tail = pool_request_hidden(
        hidden,
        kind="user_tail_mean",
        sequence_mask=sequence_mask,
        user_mask=user_mask,
        tail_tokens=2,
    )
    assert torch.equal(tail[0], hidden[0, 2:4].mean(dim=0))


def test_request_representation_adapter_is_identity_when_up_is_zero() -> None:
    hidden = torch.randn(2, 3, 8)
    adapted = adapt_request_hidden(
        hidden,
        down=torch.randn(2, 8),
        up=torch.zeros(8, 2),
        rank=2,
    )
    assert torch.equal(adapted, hidden)


def test_request_router_training_separates_balanced_features() -> None:
    hidden = torch.tensor(
        [
            [2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
        ]
    )
    features = RequestRouteFeatureSet(
        hidden=hidden,
        targets=torch.tensor([1, 1, 0, 0]),
    )
    rows, training, calibration = train_request_router(
        torch.zeros(2, 3),
        features,
        features,
        device=torch.device("cpu"),
        config=RequestRouterTrainConfig(
            seed=11,
            steps=100,
            batch_size=4,
            learning_rate=0.05,
            route_temperature=1.0,
            maximum_calibration_false_positive_rate=0.0,
        ),
    )
    assert rows.shape == (2, 3)
    assert training["trainable_parameters"] == 6
    assert calibration["true_positive_rate"] == 1.0
    assert calibration["false_positive_rate"] == 0.0


def test_phase11_checkpoint_adds_only_request_router_state() -> None:
    source = {
        "input_rows": torch.randn(16, 8),
        "result_columns": torch.randn(8, 12),
        "representation_down": torch.randn(2, 8),
        "representation_up": torch.randn(8, 2),
    }
    rows = torch.randn(2, 8)
    checkpoint = phase11_checkpoint_state(
        source,
        router_kind="user_mean",
        request_route_rows=rows,
        request_route_threshold=0.9,
        request_route_temperature=2.0,
    )
    assert checkpoint["request_router_kind"] == "user_mean"
    assert torch.equal(checkpoint["request_route_rows"], rows)
    assert torch.equal(checkpoint["input_rows"], source["input_rows"])
    assert torch.equal(
        checkpoint["result_columns"],
        source["result_columns"],
    )
