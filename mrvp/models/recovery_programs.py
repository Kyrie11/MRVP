from __future__ import annotations

import torch
from torch import nn

from mrvp.data.schema import CONTROL_DIM, DEG_DIM, PROGRAM_COUNT, RECOVERY_WORLD_DIM, RESET_SLOT_COUNT, RESET_SLOT_DIM, STATE_DIM
from .common import make_mlp


def _valid_heads(dim: int, requested: int) -> int:
    for h in range(max(1, requested), 0, -1):
        if dim % h == 0:
            return h
    return 1


class ProgramTokenEncoder(nn.Module):
    """Produce L recovery program tokens with cross-attention to reset slots."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        deg_dim: int = DEG_DIM,
        world_dim: int = RECOVERY_WORLD_DIM,
        slot_dim: int = RESET_SLOT_DIM,
        program_count: int = PROGRAM_COUNT,
        hidden_dim: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.program_count = program_count
        self.slot_dim = slot_dim
        self.program_queries = nn.Parameter(torch.randn(program_count, slot_dim) * 0.02)
        self.context_proj = make_mlp(state_dim + deg_dim + world_dim, [hidden_dim], slot_dim, dropout=dropout, layer_norm=True)
        heads = _valid_heads(slot_dim, 4)
        self.attn = nn.MultiheadAttention(slot_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(slot_dim)
        self.ffn = make_mlp(slot_dim, [hidden_dim], slot_dim, dropout=dropout, layer_norm=True)
        self.norm2 = nn.LayerNorm(slot_dim)

    def forward(self, reset_state: torch.Tensor, reset_slots: torch.Tensor, recovery_world_vec: torch.Tensor, degradation: torch.Tensor) -> torch.Tensor:
        b = reset_state.shape[0]
        context = self.context_proj(torch.cat([reset_state.float(), degradation.float(), recovery_world_vec.float()], dim=-1))
        q = self.program_queries.unsqueeze(0).expand(b, -1, -1) + context.unsqueeze(1)
        attn, _ = self.attn(q, reset_slots.float(), reset_slots.float())
        p = self.norm1(q + attn)
        p = self.norm2(p + self.ffn(p))
        return p


class ResidualPolicyHead(nn.Module):
    """Closed-loop residual policy head evaluated during rollout."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        deg_dim: int = DEG_DIM,
        world_dim: int = RECOVERY_WORLD_DIM,
        program_dim: int = RESET_SLOT_DIM,
        target_dim: int = 5,
        control_dim: int = CONTROL_DIM,
        hidden_dim: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.control_dim = control_dim
        self.tau_embed = nn.Embedding(256, 16)
        self.world_proj = make_mlp(world_dim, [hidden_dim // 2], 64, dropout=dropout, layer_norm=True)
        self.deg_proj = make_mlp(deg_dim, [hidden_dim // 2], 32, dropout=dropout, layer_norm=True)
        in_dim = state_dim + program_dim + target_dim + 64 + 32 + 16
        self.residual = make_mlp(in_dim, [hidden_dim, hidden_dim], control_dim, dropout=dropout, layer_norm=True)

    def base_control(self, state: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_x = target[..., 0]
        target_y = target[..., 1]
        target_yaw = target[..., 2]
        target_speed = target[..., 3].clamp_min(0.0)
        lateral_error = (target_y - state[..., 1]).clamp(-5.0, 5.0)
        heading_error = torch.atan2(torch.sin(target_yaw - state[..., 2]), torch.cos(target_yaw - state[..., 2]))
        speed = torch.sqrt((state[..., 3] ** 2 + state[..., 4] ** 2).clamp_min(1e-6))
        speed_error = target_speed - speed
        steer = 0.22 * lateral_error + 0.55 * heading_error
        brake = torch.sigmoid(-0.8 * speed_error)
        throttle = torch.sigmoid(0.6 * speed_error)
        base = torch.stack([steer, brake, throttle], dim=-1)
        if self.control_dim < 3:
            base = base[..., : self.control_dim]
        elif self.control_dim > 3:
            pad = torch.zeros(*base.shape[:-1], self.control_dim - 3, device=base.device, dtype=base.dtype)
            base = torch.cat([base, pad], dim=-1)
        return base

    def forward(
        self,
        state: torch.Tensor,
        program_token: torch.Tensor,
        target: torch.Tensor,
        recovery_world_vec: torch.Tensor,
        degradation: torch.Tensor,
        tau: int,
    ) -> torch.Tensor:
        leading = state.shape[:-1]
        tau_idx = torch.full(leading, int(min(max(tau, 0), 255)), dtype=torch.long, device=state.device)
        tau_emb = self.tau_embed(tau_idx)
        world_emb = self.world_proj(recovery_world_vec.float())
        deg_emb = self.deg_proj(degradation.float())
        raw = self.residual(torch.cat([state.float(), program_token.float(), target.float(), world_emb, deg_emb, tau_emb], dim=-1))
        u = self.base_control(state.float(), target.float()) + 0.25 * raw
        if self.control_dim >= 3:
            u0 = torch.tanh(u[..., :1])
            u12 = torch.sigmoid(u[..., 1:3])
            rest = torch.tanh(u[..., 3:]) if self.control_dim > 3 else u[..., 3:]
            return torch.cat([u0, u12, rest], dim=-1)
        return torch.tanh(u)


class FunnelHead(nn.Module):
    """Signed funnel field on rollout states."""

    def __init__(self, state_dim: int = STATE_DIM, program_dim: int = RESET_SLOT_DIM, target_dim: int = 5, hidden_dim: int = 256, dropout: float = 0.05) -> None:
        super().__init__()
        self.tau_embed = nn.Embedding(256, 16)
        self.net = make_mlp(state_dim + program_dim + target_dim + 16, [hidden_dim, hidden_dim // 2], 1, dropout=dropout, layer_norm=True)

    def forward(self, states: torch.Tensor, program_tokens: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # states [B,L,H+1,S], tokens [B,L,D], targets [B,L,T]
        b, l, h1, _ = states.shape
        tau = torch.arange(h1, device=states.device).clamp_max(255)
        tau_emb = self.tau_embed(tau).view(1, 1, h1, -1).expand(b, l, -1, -1)
        p = program_tokens.unsqueeze(2).expand(-1, -1, h1, -1)
        g = targets.unsqueeze(2).expand(-1, -1, h1, -1)
        raw = self.net(torch.cat([states.float(), p.float(), g.float(), tau_emb], dim=-1)).squeeze(-1)
        dist = torch.linalg.norm(states[..., :2] - targets[..., None, :2], dim=-1)
        return raw + 3.0 - 0.15 * dist
