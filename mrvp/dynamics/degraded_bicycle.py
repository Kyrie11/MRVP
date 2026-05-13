from __future__ import annotations

from typing import Callable, Tuple

import torch

from mrvp.data.schema import CONTROL_DIM, DEG_DIM, STATE_DIM


def _expand_degradation(degradation: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    d = degradation.float()
    while d.dim() < target.dim():
        d = d.unsqueeze(1)
    return d.expand(*target.shape[:-1], d.shape[-1])


def degraded_bicycle_step(state: torch.Tensor, control: torch.Tensor, degradation: torch.Tensor, dt: float = 0.1) -> torch.Tensor:
    """Differentiable degraded bicycle-like dynamics.

    State layout is [p_x,p_y,psi,v_x,v_y,yaw_rate,a_x,a_y,beta,delta,F_b,F_x].
    Control order is [delta,F_b,F_x].  Degradation scales steering, braking,
    throttle and friction.
    """
    x = state.float()
    u = control.float()
    d = _expand_degradation(degradation, x)
    steer_scale = d[..., 0].clamp(0.05, 1.5) if d.shape[-1] > 0 else 1.0
    brake_scale = d[..., 1].clamp(0.05, 1.5) if d.shape[-1] > 1 else 1.0
    throttle_scale = d[..., 2].clamp(0.05, 1.5) if d.shape[-1] > 2 else 1.0
    friction = d[..., 4].clamp(0.05, 1.5) if d.shape[-1] > 4 else 1.0
    delta = torch.tanh(u[..., 0]) * steer_scale
    brake = u[..., 1].clamp_min(0.0) * brake_scale if u.shape[-1] > 1 else torch.zeros_like(delta)
    throttle = u[..., 2].clamp_min(0.0) * throttle_scale if u.shape[-1] > 2 else torch.zeros_like(delta)
    speed = torch.sqrt((x[..., 3] ** 2 + x[..., 4] ** 2).clamp_min(1e-6))
    accel = (2.8 * throttle - 4.8 * brake - 0.12 * speed) * friction
    yaw_rate = x[..., 5] + dt * (0.32 * delta * speed / 2.7 - 0.35 * x[..., 5]) * friction
    yaw = x[..., 2] + dt * yaw_rate
    speed_next = (speed + dt * accel).clamp_min(0.0)
    vx = speed_next * torch.cos(yaw)
    vy = speed_next * torch.sin(yaw)
    nxt = x.clone()
    nxt[..., 0] = x[..., 0] + dt * vx
    nxt[..., 1] = x[..., 1] + dt * vy
    nxt[..., 2] = yaw
    nxt[..., 3] = vx
    nxt[..., 4] = vy
    nxt[..., 5] = yaw_rate
    nxt[..., 6] = (vx - x[..., 3]) / max(dt, 1e-6)
    nxt[..., 7] = (vy - x[..., 4]) / max(dt, 1e-6)
    nxt[..., 8] = torch.atan2(vy, vx.clamp_min(1e-4)) - yaw
    if nxt.shape[-1] >= 12:
        nxt[..., 9] = delta
        nxt[..., 10] = brake
        nxt[..., 11] = throttle
    return nxt.to(dtype=state.dtype)


def rollout_degraded_bicycle(
    reset_state: torch.Tensor,
    policy_fn: Callable[[torch.Tensor, int], torch.Tensor],
    degradation: torch.Tensor,
    horizon: int,
    dt: float = 0.1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    states = [reset_state]
    controls = []
    x = reset_state
    for tau in range(int(horizon)):
        u = policy_fn(x, tau)
        x = degraded_bicycle_step(x, u, degradation, dt=dt)
        controls.append(u)
        states.append(x)
    return torch.stack(states, dim=-2), torch.stack(controls, dim=-2)
