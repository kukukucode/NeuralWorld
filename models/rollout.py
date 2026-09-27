"""Inference helpers for feeding learned predictions back into model inputs."""

from __future__ import annotations

import torch

from data.dataset import StateCodec
from models.state_mlp import StateMLP


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

    if not isinstance(state, torch.Tensor):
        raise TypeError("state must be a torch tensor")
    if state.ndim not in (1, 2) or state.shape[-1] != codec.state_size:
        raise ValueError(f"state must be a vector or batch with {codec.state_size} features")
    if model.state_size != codec.state_size or model.target_size != codec.target_size:
        raise ValueError("model dimensions must match the state codec")
    parameter = next(model.parameters())
    if state.dtype != parameter.dtype or state.device != parameter.device:
        raise ValueError("state must share the model's dtype and device")

    if isinstance(action, torch.Tensor):
        if action.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
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
