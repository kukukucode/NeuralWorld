from __future__ import annotations

import pytest
import torch

from data.dataset import StateCodec
from game import Action, NeuralWorldEnv
from models import StateMLP, predict_next_state, rollout


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


def make_motion_model() -> tuple[StateMLP, StateCodec, torch.Tensor]:
    """A known continuous motion model, independent of the rollout implementation."""
    _, codec, state = make_inputs()
    state[:4] = torch.tensor([0.25, 0.25, 0.5, 0.5])
    size = codec.state_size + len(Action)
    model = StateMLP(codec.state_size, hidden_sizes=(size,))
    with torch.no_grad():
        model.network[0].weight.copy_(torch.eye(size))
        model.network[0].bias.zero_()
        output = model.network[-1]
        output.weight.zero_()
        output.bias.zero_()
        output.weight[:, :4].copy_(torch.eye(4))
        for action, coordinate, delta in (
            (Action.RIGHT, 0, 0.1), (Action.LEFT, 0, -0.1),
            (Action.DOWN, 1, 0.1), (Action.UP, 1, -0.1),
        ):
            output.weight[coordinate, codec.state_size + action] = delta
    return model, codec, state


@pytest.mark.parametrize("training", [True, False])
def test_rollout_feeds_predictions_back_and_preserves_inputs(training: bool) -> None:
    model, codec, state = make_motion_model()
    model.train(training)
    state.requires_grad_()
    original = state.detach().clone()
    actions = torch.tensor([Action.RIGHT, Action.DOWN, Action.RIGHT])
    original_actions = actions.clone()

    states = rollout(model, codec, state, actions)

    expected = torch.tensor([[0.25, 0.25], [0.35, 0.25], [0.35, 0.35], [0.45, 0.35]])
    torch.testing.assert_close(states[:, :2], expected)
    torch.testing.assert_close(states[:, 2:4], torch.full((4, 2), 0.5))
    assert torch.equal(states[:, 4:], original[4:].expand(4, -1))
    assert torch.equal(state, original)
    assert torch.equal(actions, original_actions)
    assert states.shape == (4, codec.state_size)
    assert not states.requires_grad
    assert model.training is training
    assert all(parameter.grad is None for parameter in model.parameters())


def test_single_step_rollout_matches_prediction_helper() -> None:
    model, codec, state = make_inputs()
    states = rollout(model, codec, state, [Action.RIGHT])
    torch.testing.assert_close(states[0], state)
    torch.testing.assert_close(states[1], predict_next_state(model, codec, state, Action.RIGHT))


def test_rollout_supports_shared_and_per_example_batch_actions() -> None:
    model, codec, state = make_motion_model()
    initial = state.repeat(2, 1)
    actions = torch.tensor([[Action.RIGHT, Action.LEFT], [Action.DOWN, Action.UP]])
    states = rollout(model, codec, initial, actions)
    assert states.shape == (3, 2, codec.state_size)
    torch.testing.assert_close(states[-1, :, :2], torch.tensor([[0.35, 0.35], [0.15, 0.15]]))
    assert torch.equal(states[:, :, 4:], initial[:, 4:].expand(3, -1, -1))
    shared = rollout(model, codec, initial, [Action.RIGHT, Action.DOWN])
    torch.testing.assert_close(shared[:, 0], states[:, 0])
    torch.testing.assert_close(shared[:, 0], shared[:, 1])


@pytest.mark.parametrize("batched", [True, False])
@pytest.mark.parametrize("actions", [[], torch.empty(0, dtype=torch.long)])
def test_empty_rollout_returns_independent_initial_state(batched: bool, actions: object) -> None:
    model, codec, state = make_inputs()
    if batched:
        state = state.repeat(2, 1)
    state.requires_grad_()
    states = rollout(model, codec, state, actions)
    assert states.shape == (1, *state.shape)
    assert torch.equal(states[0], state)
    assert not states.requires_grad
    states.zero_()
    assert torch.count_nonzero(state) > 0


def test_long_rollout_is_deterministic_bounded_and_preserves_static_state() -> None:
    model, codec, state = make_motion_model()
    actions = [Action.RIGHT] * 25 + [Action.LEFT] * 25
    states = rollout(model, codec, state, actions)
    assert states.shape == (51, codec.state_size)
    assert torch.equal(states, rollout(model, codec, state, actions))
    assert torch.all((states[:, :4] >= 0) & (states[:, :4] <= 1))
    assert torch.equal(states[:, 4:], state[4:].expand(51, -1))
    assert states[25, 0].item() == 1.0
    assert states[-1, 0].item() == 0.0


@pytest.mark.parametrize("actions", [[-1], [5], [Action.RIGHT, -1], torch.tensor([5])])
def test_rollout_rejects_out_of_range_actions(actions: object) -> None:
    model, codec, state = make_inputs()
    with pytest.raises(ValueError, match="actions must be"):
        rollout(model, codec, state, actions)


@pytest.mark.parametrize("actions", [[1.5], [True], torch.tensor([1.5]), torch.tensor([True]), 1])
def test_rollout_rejects_non_integer_actions(actions: object) -> None:
    model, codec, state = make_inputs()
    with pytest.raises(TypeError, match="integer"):
        rollout(model, codec, state, actions)


@pytest.mark.parametrize("actions", [
    torch.tensor(1), torch.zeros(1, 1, dtype=torch.long),
    torch.zeros(1, 1, 1, dtype=torch.long),
])
def test_rollout_rejects_invalid_single_state_action_shapes(actions: torch.Tensor) -> None:
    model, codec, state = make_inputs()
    with pytest.raises(ValueError, match="shape"):
        rollout(model, codec, state, actions)


def test_rollout_validates_empty_actions_and_batch_dimensions() -> None:
    model, codec, state = make_inputs()
    with pytest.raises(ValueError, match="shape"):
        rollout(model, codec, state.repeat(2, 1), torch.empty(0, 3, dtype=torch.long))
    with pytest.raises(ValueError, match="features"):
        rollout(model, codec, state[:-1], [])
    with pytest.raises(ValueError, match="dtype and device"):
        rollout(model, codec, state.double(), [])
    with pytest.raises(TypeError, match="torch tensor"):
        rollout(model, codec, state.tolist(), [])
    incompatible = StateMLP(codec.state_size, target_size=2, hidden_sizes=(16,))
    with pytest.raises(ValueError, match="model dimensions"):
        rollout(incompatible, codec, state, [])
