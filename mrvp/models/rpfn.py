from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .encoders import BEVEncoder, MLP, TokenAttention
from .rollout import clamp_controls, degraded_bicycle_rollout


@dataclass
class RPFNConfig:
    dim: int = 64
    slot_count: int = 16
    slot_dim: int = 64
    num_programs: int = 6
    horizon: int = 30
    dt: float = 0.1
    state_dim: int = 12
    control_dim: int = 3
    deg_dim: int = 8
    world_size: int = 64
    use_funnel_head: bool = True
    use_degradation_input: bool = True
    use_program_rollout: bool = True
    use_ordering_loss: bool = True
    direct_certificate_head: bool = False


class RPFN(nn.Module):
    def __init__(self, cfg: RPFNConfig | dict | None = None):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = RPFNConfig(**{k: v for k, v in cfg.items() if k in RPFNConfig.__annotations__})
        self.cfg = cfg or RPFNConfig()
        d = self.cfg.dim
        self.program_queries = nn.Parameter(torch.randn(self.cfg.num_programs, d) * 0.02)
        self.reset_proj = nn.Linear(self.cfg.state_dim + self.cfg.deg_dim, d)
        self.slot_proj = nn.Linear(self.cfg.slot_dim, d)
        self.bev = BEVEncoder(6 + 3, d)
        self.attn = nn.ModuleList([TokenAttention(d) for _ in range(3)])
        self.target_head = nn.Linear(d, 5)
        self.policy_head = MLP([self.cfg.state_dim + d + self.cfg.deg_dim + d, d, self.cfg.control_dim])
        self.funnel_head = MLP([self.cfg.state_dim + d + self.cfg.deg_dim + 1, d, 1])
        self.direct_head = nn.Linear(d, 1)

    def _make_memory(self, reset: dict) -> torch.Tensor:
        r = reset["r_reset"]
        deg = reset["deg"]
        if not self.cfg.use_degradation_input:
            deg = torch.ones_like(deg)
            deg[:, 0] = 0.95
            deg[:, 4] = 0.0
        base = self.reset_proj(torch.cat([r, deg], dim=-1)).unsqueeze(1)
        z = self.slot_proj(reset["z_slots"])
        bev_in = torch.cat([reset["world_A"][:, :6], reset["world_G"][:, :3]], dim=1)
        bev_token = self.bev(bev_in).unsqueeze(1)
        return torch.cat([base, z, bev_token], dim=1)

    def _base_control(self, x: torch.Tensor, target_xy: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        dx = target_xy[..., 0] - x[..., 0]
        dy = target_xy[..., 1] - x[..., 1]
        desired = torch.atan2(dy, torch.clamp(dx, min=1e-3))
        yaw_err = torch.atan2(torch.sin(desired - x[..., 2]), torch.cos(desired - x[..., 2]))
        steer = torch.clamp(0.4 * yaw_err, -0.6, 0.6)
        speed_err = torch.clamp(x[..., 3] - 3.0, min=0.0)
        brake = torch.clamp(0.12 * speed_err, 0.0, 1.0)
        throttle = torch.zeros_like(brake)
        return torch.stack([steer, brake, throttle], dim=-1)

    def _world_feat(self, reset: dict) -> torch.Tensor:
        A = reset["world_A"]
        pooled = A.mean(dim=(-1, -2))
        return pooled[:, :self.cfg.dim] if pooled.shape[-1] >= self.cfg.dim else F.pad(pooled, (0, self.cfg.dim - pooled.shape[-1]))

    def _margins(self, reset: dict, traj: torch.Tensor, controls: torch.Tensor, funnel: torch.Tensor, target_xy: torch.Tensor) -> dict[str, torch.Tensor]:
        B, L, H1, _ = traj.shape
        A = reset["world_A"]
        O = reset["world_O"]
        deg = reset["deg"]
        drivable_strength = A[:, 0].mean(dim=(-1, -2)).view(B, 1).expand(B, L)
        static_strength = A[:, 2].mean(dim=(-1, -2)).view(B, 1).expand(B, L)
        occ_strength = O.mean(dim=(-1, -2, -3, -4)).view(B, 1).expand(B, L)
        lateral = torch.abs(traj[..., 1]).amax(dim=-1)
        road = drivable_strength - 0.02 * lateral - static_strength
        col = 1.0 - occ_strength - 0.05 * torch.abs(traj[..., 0]).mean(dim=-1)
        bounds = torch.stack([0.6 * deg[:, 1], deg[:, 2], deg[:, 3]], dim=-1).unsqueeze(1).unsqueeze(2)
        ctrl_margin = (bounds - torch.abs(controls)).amin(dim=(-1, -2))
        beta = torch.abs(traj[..., 8]).amax(dim=-1)
        yaw = torch.abs(traj[..., 5]).amax(dim=-1)
        stab = torch.minimum(deg[:, 5].unsqueeze(1) - beta, deg[:, 6].unsqueeze(1) - yaw)
        final_xy = traj[:, :, -1, :2]
        goal = 1.0 - torch.linalg.norm(final_xy - target_xy, dim=-1) / 20.0
        fmin = funnel.amin(dim=-1)
        return {"road": road, "col": col, "ctrl": ctrl_margin, "stab": stab, "goal": goal, "funnel": fmin}

    def forward(self, reset: dict) -> dict:
        B = reset["r_reset"].shape[0]
        mem = self._make_memory(reset)
        q = self.program_queries.unsqueeze(0).expand(B, -1, -1)
        for layer in self.attn:
            q = layer(q, mem)
        target_raw = self.target_head(q)
        target_type_logits = target_raw[..., :3]
        target_xy = torch.tanh(target_raw[..., 3:5]) * 18.0
        deg = reset["deg"] if self.cfg.use_degradation_input else torch.ones_like(reset["deg"])
        x = reset["r_reset"].unsqueeze(1).expand(B, self.cfg.num_programs, -1)
        controls = []
        states_for_policy = []
        for t in range(self.cfg.horizon):
            world_feat = self._world_feat(reset).unsqueeze(1).expand(-1, self.cfg.num_programs, -1)
            inp = torch.cat([x, q, deg.unsqueeze(1).expand(-1, self.cfg.num_programs, -1), world_feat], dim=-1)
            residual = torch.tanh(self.policy_head(inp)) * 0.25
            u = self._base_control(x, target_xy, deg.unsqueeze(1).expand(-1, self.cfg.num_programs, -1)) + residual
            controls.append(u)
            states_for_policy.append(x)
            x_next = degraded_bicycle_rollout(x.reshape(B * self.cfg.num_programs, -1), u.reshape(B * self.cfg.num_programs, 1, -1), dt=self.cfg.dt)[:, -1]
            x = x_next.view(B, self.cfg.num_programs, -1)
        controls_t = torch.stack(controls, dim=2)
        controls_t = clamp_controls(controls_t.reshape(B * self.cfg.num_programs, self.cfg.horizon, -1), deg.unsqueeze(1).expand(-1, self.cfg.num_programs, -1).reshape(B * self.cfg.num_programs, -1)).view(B, self.cfg.num_programs, self.cfg.horizon, -1)
        if self.cfg.use_program_rollout:
            traj = degraded_bicycle_rollout(reset["r_reset"].unsqueeze(1).expand(B, self.cfg.num_programs, -1).reshape(B * self.cfg.num_programs, -1), controls_t.reshape(B * self.cfg.num_programs, self.cfg.horizon, -1), dt=self.cfg.dt).view(B, self.cfg.num_programs, self.cfg.horizon + 1, -1)
        else:
            traj = reset["r_reset"].unsqueeze(1).unsqueeze(2).expand(B, self.cfg.num_programs, self.cfg.horizon + 1, -1).clone()
        tau = torch.linspace(0.0, 1.0, self.cfg.horizon + 1, device=traj.device).view(1, 1, -1, 1).expand(B, self.cfg.num_programs, -1, -1)
        q_expand = q.unsqueeze(2).expand(-1, -1, self.cfg.horizon + 1, -1)
        deg_expand = deg.unsqueeze(1).unsqueeze(2).expand(-1, self.cfg.num_programs, self.cfg.horizon + 1, -1)
        funnel_in = torch.cat([traj, q_expand, deg_expand, tau], dim=-1)
        funnel = self.funnel_head(funnel_in).squeeze(-1) if self.cfg.use_funnel_head else torch.ones(B, self.cfg.num_programs, self.cfg.horizon + 1, device=traj.device)
        margins = self._margins(reset, traj, controls_t, funnel, target_xy)
        cert = torch.stack([margins["funnel"], margins["road"], margins["col"], margins["ctrl"], margins["stab"], margins["goal"]], dim=-1).amin(dim=-1)
        if self.cfg.direct_certificate_head:
            cert = self.direct_head(q).squeeze(-1)
        return {
            "program_tokens": q,
            "target_type_logits": target_type_logits,
            "target_xy": target_xy,
            "controls": controls_t,
            "trajectories": traj,
            "funnel_values": funnel,
            "cert_pred": cert,
            "margins": margins,
        }

    def forward_from_batch(self, batch: dict, z_slots: torch.Tensor | None = None) -> dict:
        B = batch["r_reset"].shape[0]
        if z_slots is None:
            z_slots = torch.zeros(B, self.cfg.slot_count, self.cfg.slot_dim, device=batch["r_reset"].device)
        reset = {
            "r_reset": batch["r_reset"],
            "z_slots": z_slots,
            "world_A": batch["world_A"],
            "world_O": batch["world_O"],
            "world_G": batch["world_G"],
            "world_Y": batch["world_Y"],
            "deg": batch["deg"],
            "sigma": torch.ones_like(batch["r_reset"]),
        }
        return self.forward(reset)
