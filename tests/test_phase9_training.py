import torch

from neural_firmware.phase7_sequence_training import (
    FirstStepRouteFeatureSet,
    SequenceFeatureSet,
)
from neural_firmware.phase9_training import (
    concatenate_route_features,
    concatenate_sequence_features,
)


def _sequence(offset: int) -> SequenceFeatureSet:
    return SequenceFeatureSet(
        hidden=torch.arange(offset, offset + 6).reshape(2, 3).float(),
        route_targets=torch.tensor([0, 1]),
        role_targets=torch.tensor([0, 2]),
        digit_targets=torch.tensor([10, 7]),
        step_targets=torch.tensor([-1, 0]),
    )


def test_concatenate_sequence_features_preserves_alignment() -> None:
    joined = concatenate_sequence_features((_sequence(0), _sequence(6)))
    assert joined.hidden.shape == (4, 3)
    assert joined.hidden[:, 0].tolist() == [0, 3, 6, 9]
    assert joined.route_targets.tolist() == [0, 1, 0, 1]
    assert joined.role_targets.tolist() == [0, 2, 0, 2]


def test_concatenate_route_features_preserves_alignment() -> None:
    first = FirstStepRouteFeatureSet(
        hidden=torch.tensor([[1.0, 2.0]]),
        targets=torch.tensor([0]),
    )
    second = FirstStepRouteFeatureSet(
        hidden=torch.tensor([[3.0, 4.0], [5.0, 6.0]]),
        targets=torch.tensor([1, 0]),
    )
    joined = concatenate_route_features((first, second))
    assert joined.hidden.tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
    assert joined.targets.tolist() == [0, 1, 0]


def test_concatenate_requires_input() -> None:
    for function in (concatenate_sequence_features, concatenate_route_features):
        try:
            function(())
        except ValueError:
            pass
        else:
            raise AssertionError("empty concatenation should fail")
