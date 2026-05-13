from __future__ import annotations

import torch


def _world_feature(world: torch.Tensor, idx: int, default: float = 0.0) -> torch.Tensor:
    if world.shape[-1] <= idx:
        return torch.full(world.shape[:-1], float(default), device=world.device, dtype=world.dtype)
    return world[..., idx]


def _expand_world(world: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    w = world.float()
    while w.dim() < states.dim() - 1:
        w = w.unsqueeze(1)
    return w.expand(*states.shape[:-2], w.shape[-1])


def _expand_deg(degradation: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
    d = degradation.float()
    while d.dim() < states.dim() - 1:
        d = d.unsqueeze(1)
    return d.expand(*states.shape[:-2], d.shape[-1])


def road_margin(states: torch.Tensor, recovery_world: torch.Tensor) -> torch.Tensor:
    world = _expand_world(recovery_world, states)
    lane_width = _world_feature(world, 0, 3.5).abs().clamp_min(2.5)
    half_width = 0.5 * lane_width
    y = states[..., 1]
    return half_width.unsqueeze(-1) - y.abs() - 0.95


def collision_margin(states: torch.Tensor, recovery_world: torch.Tensor) -> torch.Tensor:
    world = _expand_world(recovery_world, states)
    actor_density = _world_feature(world, 8, 0.2).clamp_min(0.0)
    secondary_clear = _world_feature(world, 9, 8.0)
    speed = torch.sqrt((states[..., 3] ** 2 + states[..., 4] ** 2).clamp_min(1e-6))
    return secondary_clear.unsqueeze(-1) - 1.5 - 0.15 * speed - 0.5 * actor_density.unsqueeze(-1)


def control_margin(controls: torch.Tensor, degradation: torch.Tensor) -> torch.Tensor:
    d = degradation.float()
    while d.dim() < controls.dim():
        d = d.unsqueeze(1)
    d = d.expand(*controls.shape[:-1], d.shape[-1])
    steer_scale = d[..., 0].clamp_min(0.05) if d.shape[-1] > 0 else 1.0
    brake_scale = d[..., 1].clamp_min(0.05) if d.shape[-1] > 1 else 1.0
    throttle_scale = d[..., 2].clamp_min(0.05) if d.shape[-1] > 2 else 1.0
    steer_m = steer_scale - controls[..., 0].abs()
    brake_m = brake_scale - controls[..., 1].clamp_min(0.0) if controls.shape[-1] > 1 else steer_m
    throttle_m = throttle_scale - controls[..., 2].clamp_min(0.0) if controls.shape[-1] > 2 else steer_m
    return torch.stack([steer_m, brake_m, throttle_m], dim=-1).min(dim=-1).values


def stability_margin(states: torch.Tensor, degradation: torch.Tensor) -> torch.Tensor:
    d = _expand_deg(degradation, states)
    friction = d[..., 4].clamp_min(0.05) if d.shape[-1] > 4 else torch.ones_like(states[..., 0])
    yaw_rate = states[..., 5].abs()
    beta = states[..., 8].abs() if states.shape[-1] > 8 else torch.zeros_like(yaw_rate)
    speed = torch.sqrt((states[..., 3] ** 2 + states[..., 4] ** 2).clamp_min(1e-6))
    return 1.4 * friction.unsqueeze(-1) - 0.55 * yaw_rate - 0.35 * beta - 0.02 * speed


def goal_margin(final_state: torch.Tensor, program_target: torch.Tensor, recovery_world: torch.Tensor) -> torch.Tensor:
    target_xy = program_target[..., :2]
    dist = torch.linalg.norm(final_state[..., :2] - target_xy, dim=-1)
    target_speed = program_target[..., 3].clamp_min(0.0) if program_target.shape[-1] > 3 else torch.zeros_like(dist)
    speed = torch.sqrt((final_state[..., 3] ** 2 + final_state[..., 4] ** 2).clamp_min(1e-6))
    speed_margin = 4.0 - (speed - target_speed).abs()
    return torch.minimum(3.0 - dist, speed_margin)
