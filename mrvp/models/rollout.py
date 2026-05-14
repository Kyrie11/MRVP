from __future__ import annotations

import torch


def degraded_bicycle_rollout(x0: torch.Tensor, controls: torch.Tensor, dt: float = 0.1, wheelbase: float = 2.8) -> torch.Tensor:
    x = x0
    states = [x]
    for t in range(controls.shape[-2]):
        u = controls[..., t, :]
        steer = torch.clamp(u[..., 0], -0.6, 0.6)
        brake = torch.clamp(u[..., 1], 0.0, 1.5)
        traction = torch.clamp(u[..., 2], -1.5, 1.5)
        v = torch.clamp(x[..., 3] + (traction - 4.0 * brake) * dt, min=0.0)
        yaw_rate = v / wheelbase * torch.tan(steer)
        yaw = x[..., 2] + yaw_rate * dt
        px = x[..., 0] + v * torch.cos(yaw) * dt
        py = x[..., 1] + v * torch.sin(yaw) * dt
        values = []
        for i in range(x.shape[-1]):
            if i == 0:
                values.append(px)
            elif i == 1:
                values.append(py)
            elif i == 2:
                values.append(yaw)
            elif i == 3:
                values.append(v)
            elif i == 5:
                values.append(yaw_rate)
            elif i == 8:
                values.append(torch.atan2(x[..., 4], torch.clamp(v, min=1e-3)))
            elif i == 9:
                values.append(steer)
            elif i == 10:
                values.append(brake)
            elif i == 11:
                values.append(traction)
            else:
                values.append(x[..., i])
        x = torch.stack(values, dim=-1)
        states.append(x)
    return torch.stack(states, dim=-2)


def clamp_controls(controls: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
    max_steer = 0.6 * deg[..., 1].unsqueeze(-1)
    max_brake = 1.0 * deg[..., 2].unsqueeze(-1)
    max_thr = 1.0 * deg[..., 3].unsqueeze(-1)
    steer = torch.minimum(torch.maximum(controls[..., 0], -max_steer), max_steer)
    brake = torch.minimum(torch.clamp(controls[..., 1], min=0.0), max_brake)
    throttle = torch.minimum(torch.maximum(controls[..., 2], -max_thr), max_thr)
    return torch.stack([steer, brake, throttle], dim=-1)
