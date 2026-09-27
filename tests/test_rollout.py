from __future__ import annotations

import pytest
import torch

from data.dataset import StateCodec
from game import Action, NeuralWorldEnv
from models import StateMLP, predict_next_state


def make_inputs() -> tuple[StateMLP, StateCodec, torch.Tensor]:
    env = NeuralWorldEnv()
    observation, _ = env.reset(seed=42)
    codec = StateCodec(width=env.width, height=env.height)
    state = torch.from_numpy(codec.encode(observation))
    env.close()
    torch.manual_seed(13)
    model = StateMLP(codec.state_size, hidden_sizes=(16,))
    return model, codec, state


@pytest.mark.parametrize("training", [True, False])
def test_next_state_matches_direct_prediction_and_preserves_inputs(training: bool) -> None:
    model, codec, state = make_inputs()
    model.train(training)
    state.requires_grad_()
    original = state.clone()
    with torch.no_grad():
        direct_prediction = model(state, torch.tensor(Action.RIGHT))

    next_state = predict_next_state(model, codec, state, Action.RIGHT)

    torch.testing.assert_close(next_state[:4], direct_prediction.clamp(0.0, 1.0))
    assert torch.equal(next_state[4:], original[4:])
    assert torch.equal(state, original)
    assert next_state.shape == state.shape
    assert not next_state.requires_grad
    assert model.training is training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_batched_next_states_use_each_action() -> None:
    model, codec, state = make_inputs()
    states = state.repeat(len(Action), 1)
    actions = torch.arange(len(Action))
    with torch.no_grad():
        direct_predictions = model(states, actions)

    next_states = predict_next_state(model, codec, states, actions)

    torch.testing.assert_close(next_states[:, :4], direct_predictions.clamp(0.0, 1.0))
    assert torch.equal(next_states[:, 4:], states[:, 4:])
    assert next_states.shape == states.shape
    # An integer action is also supported for a batch sharing the same action.
    same_action = predict_next_state(model, codec, states, Action.RIGHT)
    torch.testing.assert_close(same_action, next_states[Action.RIGHT].expand_as(same_action))


@pytest.mark.parametrize("action", [-1, 5, torch.tensor(-1), torch.tensor(5)])
def test_next_state_rejects_out_of_range_actions(action: int | torch.Tensor) -> None:
    model, codec, state = make_inputs()
    with pytest.raises(ValueError, match="actions must be"):
        predict_next_state(model, codec, state, action)


@pytest.mark.parametrize("action", [1.5, True, torch.tensor(1.5)])
def test_next_state_rejects_non_integer_actions(action: object) -> None:
    model, codec, state = make_inputs()
    with pytest.raises(TypeError, match="integer"):
        predict_next_state(model, codec, state, action)


def test_next_state_rejects_incompatible_inputs() -> None:
    model, codec, state = make_inputs()
    with pytest.raises(ValueError, match="features"):
        predict_next_state(model, codec, state[:-1], Action.RIGHT)
    with pytest.raises(ValueError, match="batch dimensions"):
        predict_next_state(model, codec, state.repeat(2, 1), torch.tensor([Action.RIGHT]))
    with pytest.raises(ValueError, match="dtype and device"):
        predict_next_state(model, codec, state.double(), Action.RIGHT)
    incompatible_model = StateMLP(codec.state_size, target_size=2, hidden_sizes=(16,))
    with pytest.raises(ValueError, match="model dimensions"):
        predict_next_state(incompatible_model, codec, state, Action.RIGHT)
