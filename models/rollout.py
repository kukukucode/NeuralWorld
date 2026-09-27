"""Inference helpers for feeding learned predictions back into model inputs."""

from __future__ import annotations

from collections.abc import Sequence

import torch

from data.dataset import StateCodec
from models.state_mlp import StateMLP


_INTEGER_DTYPES = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)


def _validate_state(model: StateMLP, codec: StateCodec, state: torch.Tensor) -> None:
    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch tensor")
    if state.ndim not in (1, 2) or state.shape[-1] != codec.state_size:
        raise ValueError(f"state must be a vector or batch with {codec.state_size} features")
    if model.state_size != codec.state_size or model.target_size != codec.target_size:
        raise ValueError("model dimensions must match the state codec")
    parameter = next(model.parameters())
    if state.dtype != parameter.dtype or state.device != parameter.device:
        raise ValueError("state must share the model's dtype and device")


@torch.no_grad()
def predict_next_state(
    model: StateMLP,
    codec: StateCodec,
    state: torch.Tensor,
    action: int | torch.Tensor,
) -> torch.Tensor:
    """Predict one full next state, preserving the Goal and walls.

    State is a normalized vector or a batch of vectors on the model's device
    and with its dtype. An integer action applies to every state; tensor actions
    must match the state's batch dimensions. Player/Box predictions are clipped
    to [0, 1] without grid rounding. Inference preserves the model's train/eval
    mode and does not modify the input state.
    """

    _validate_state(model, codec, state)

    if isinstance(action, torch.Tensor):
        if action.dtype not in _INTEGER_DTYPES:
            raise TypeError("action tensor must contain integers")
        if action.shape != state.shape[:-1]:
            raise ValueError("action tensor shape must match the state's batch dimensions")
        actions = action.to(device=state.device, dtype=torch.long)
        if torch.any((actions < 0) | (actions >= model.action_count)):
            raise ValueError(f"actions must be in [0, {model.action_count - 1}]")
    elif isinstance(action, int) and not isinstance(action, bool):
        if not 0 <= action < model.action_count:
            raise ValueError(f"actions must be in [0, {model.action_count - 1}]")
        actions = torch.full(state.shape[:-1], action, dtype=torch.long, device=state.device)
    else:
        raise TypeError("action must be an integer or an integer tensor")

    was_training = model.training
    try:
        model.eval()
        prediction = model(state, actions)
        return codec.with_predicted_positions(state, prediction)
    finally:
        model.train(was_training)


@torch.no_grad()
def rollout(
    model: StateMLP,
    codec: StateCodec,
    initial_state: torch.Tensor,
    actions: Sequence[int] | torch.Tensor,
) -> torch.Tensor:
    """Feed each prediction back into the model along a given action sequence.

    Returns time-first states including the initial state: (T + 1, D) for
    a vector or (T + 1, B, D) for a batch. Actions have shape (T,) for a
    shared sequence or (T, B) for per-example sequences in a batch.
    Empty actions return just the initial state. Goal/walls are preserved,
    positions stay continuous and clipped to [0, 1], and no gradients are
    recorded. Inputs and the model's train/eval mode are left unchanged.
    """

    _validate_state(model, codec, initial_state)
    if isinstance(actions, torch.Tensor):
        if actions.dtype not in _INTEGER_DTYPES:
            raise TypeError("actions tensor must contain integers")
        action_sequence = actions.to(device=initial_state.device, dtype=torch.long)
    elif isinstance(actions, Sequence):
        if any(not isinstance(action, int) or isinstance(action, bool) for action in actions):
            raise TypeError("actions sequence must contain integers")
        action_sequence = torch.tensor(actions, dtype=torch.long, device=initial_state.device)
    else:
        raise TypeError("actions must be an integer sequence or an integer tensor")

    if action_sequence.ndim == 1:
        if initial_state.ndim == 2:
            action_sequence = action_sequence[:, None].expand(-1, initial_state.shape[0])
    elif not (
        action_sequence.ndim == 2
        and initial_state.ndim == 2
        and action_sequence.shape[1] == initial_state.shape[0]
    ):
        raise ValueError("actions must have shape (T,) or (T, B) matching the state batch")
    if torch.any((action_sequence < 0) | (action_sequence >= model.action_count)):
        raise ValueError(f"actions must be in [0, {model.action_count - 1}]")

    states = [initial_state.detach().clone()]
    for action in action_sequence:
        states.append(predict_next_state(model, codec, states[-1], action))
    return torch.stack(states)
