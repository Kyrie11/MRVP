from __future__ import annotations

from typing import Optional

import torch
from torch import nn

from mrvp.data.schema import ACTION_DIM, ACTOR_FEAT_DIM, CTX_DIM, HIST_LEN, MAX_AGENTS, NUM_ACTIONS, STATE_DIM
from .common import make_mlp


def _valid_heads(dim: int, requested: int) -> int:
    requested = max(1, int(requested))
    for h in range(requested, 0, -1):
        if dim % h == 0:
            return h
    return 1


class RootSceneMemoryEncoder(nn.Module):
    """Encode root-scene observations into a token memory M_s."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        hist_len: int = HIST_LEN,
        max_agents: int = MAX_AGENTS,
        actor_feat_dim: int = ACTOR_FEAT_DIM,
        ctx_dim: int = CTX_DIM,
        model_dim: int = 64,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.hist_len = hist_len
        self.max_agents = max_agents
        self.actor_feat_dim = actor_feat_dim
        self.ctx_dim = ctx_dim
        self.model_dim = model_dim
        self.actor_encoder = make_mlp(actor_feat_dim, [hidden_dim], model_dim, layer_norm=True)
        self.ctx_encoder = make_mlp(ctx_dim, [hidden_dim], model_dim, layer_norm=True)
        self.ego_encoder = make_mlp(state_dim, [hidden_dim], model_dim, layer_norm=True)
        self.hist_encoder = nn.GRU(model_dim, model_dim, batch_first=True)

    def forward(self, o_hist: torch.Tensor, h_ctx: torch.Tensor, x_t: torch.Tensor) -> torch.Tensor:
        if o_hist.dim() == 2:
            b = o_hist.shape[0]
            o_hist = o_hist.view(b, 1, 1, -1)
        b, t, a, f = o_hist.shape
        actor_feat = self.actor_encoder(o_hist.reshape(b * t * a, f)).reshape(b, t, a, self.model_dim)
        valid = (o_hist[..., -1:] > 0).float() if f > 0 else torch.ones(b, t, a, 1, device=o_hist.device)
        actor_tokens = actor_feat[:, -1]
        actor_tokens = actor_tokens * valid[:, -1]
        pooled = (actor_feat * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        _, h = self.hist_encoder(pooled)
        hist_token = h[-1].unsqueeze(1)
        ctx_token = self.ctx_encoder(h_ctx.float()).unsqueeze(1)
        ego_token = self.ego_encoder(x_t.float()).unsqueeze(1)
        return torch.cat([ego_token, ctx_token, hist_token, actor_tokens], dim=1)


def rollout_prefix_kinematic(x_t: torch.Tensor, action_vec: torch.Tensor, horizon: int = 20, dt: float = 0.1) -> tuple[torch.Tensor, torch.Tensor]:
    """Differentiable lightweight prefix rollout for action-intervention memory.

    action_vec follows the paper order [delta, throttle, brake, duration].  The
    returned controls are [delta, F_b, F_x] for consistency with recovery.
    """
    b, state_dim = x_t.shape
    device = x_t.device
    dtype = x_t.dtype
    states = []
    controls = []
    x = x_t.float()
    delta = action_vec[:, 0].float()
    throttle = action_vec[:, 1].float()
    brake = action_vec[:, 2].float()
    duration = action_vec[:, 3].float().clamp_min(dt)
    for k in range(horizon):
        alpha = ((k + 0.5) * dt <= duration).float()
        u_delta = alpha * delta
        u_brake = alpha * brake.clamp_min(0.0)
        u_fx = alpha * throttle.clamp_min(0.0)
        speed = torch.sqrt((x[:, 3] ** 2 + x[:, 4] ** 2).clamp_min(1e-6))
        accel = 2.0 * u_fx - 4.0 * u_brake - 0.10 * speed
        yaw_rate = x[:, 5] + dt * (0.35 * u_delta * speed - 0.20 * x[:, 5])
        yaw = x[:, 2] + dt * yaw_rate
        speed_next = (speed + dt * accel).clamp_min(0.0)
        vx = speed_next * torch.cos(yaw)
        vy = speed_next * torch.sin(yaw)
        x_next = x.clone()
        x_next[:, 0] = x[:, 0] + dt * vx
        x_next[:, 1] = x[:, 1] + dt * vy
        x_next[:, 2] = yaw
        x_next[:, 3] = vx
        x_next[:, 4] = vy
        x_next[:, 5] = yaw_rate
        x_next[:, 6] = (vx - x[:, 3]) / dt
        x_next[:, 7] = (vy - x[:, 4]) / dt
        x_next[:, 8] = torch.atan2(vy, vx.clamp_min(1e-4)) - yaw
        if state_dim >= 12:
            x_next[:, 9] = u_delta
            x_next[:, 10] = u_brake
            x_next[:, 11] = u_fx
        states.append(x_next)
        controls.append(torch.stack([u_delta, u_brake, u_fx], dim=-1).to(dtype=dtype))
        x = x_next
    return torch.stack(states, dim=1).to(dtype=dtype), torch.stack(controls, dim=1).to(dtype=dtype)


class ActionInterventionEncoder(nn.Module):
    """Encode the counterfactual action prefix rollout into memory M_a."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        action_dim: int = ACTION_DIM,
        num_actions: int = NUM_ACTIONS,
        model_dim: int = 64,
        hidden_dim: int = 128,
        prefix_horizon: int = 20,
        dt: float = 0.1,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.model_dim = model_dim
        self.prefix_horizon = prefix_horizon
        self.dt = dt
        self.action_embedding = nn.Embedding(num_actions, model_dim)
        self.prefix_encoder = make_mlp(state_dim + 3 + model_dim, [hidden_dim], model_dim, layer_norm=True)
        self.summary_encoder = make_mlp(action_dim + model_dim, [hidden_dim], model_dim, layer_norm=True)

    def forward(self, x_t: torch.Tensor, action_vec: torch.Tensor, action_id: torch.Tensor) -> torch.Tensor:
        aid = action_id.long().clamp_min(0) % self.num_actions
        aemb = self.action_embedding(aid)
        prefix_states, prefix_controls = rollout_prefix_kinematic(x_t, action_vec, self.prefix_horizon, self.dt)
        b, h, _ = prefix_states.shape
        emb = aemb[:, None, :].expand(b, h, self.model_dim)
        step_tokens = self.prefix_encoder(torch.cat([prefix_states, prefix_controls, emb], dim=-1).reshape(b * h, -1)).reshape(b, h, self.model_dim)
        summary = self.summary_encoder(torch.cat([action_vec.float(), aemb], dim=-1)).unsqueeze(1)
        return torch.cat([summary, step_tokens], dim=1)


class CounterfactualResetSlotLayer(nn.Module):
    """One CMRT slot update layer with scene/action/nominal cross attention."""

    def __init__(self, model_dim: int = 64, num_heads: int = 4, dropout: float = 0.05) -> None:
        super().__init__()
        heads = _valid_heads(model_dim, num_heads)
        self.scene_attn = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.action_attn = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.nominal_attn = nn.MultiheadAttention(model_dim, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.norm3 = nn.LayerNorm(model_dim)
        self.ffn = make_mlp(model_dim, [model_dim * 2], model_dim, dropout=dropout, layer_norm=True)
        self.norm4 = nn.LayerNorm(model_dim)

    def forward(self, slots: torch.Tensor, scene_memory: torch.Tensor, action_memory: torch.Tensor, nominal_memory: Optional[torch.Tensor] = None) -> torch.Tensor:
        scene_ctx, _ = self.scene_attn(slots, scene_memory, scene_memory)
        q = self.norm1(slots + scene_ctx)
        action_ctx, _ = self.action_attn(q, action_memory, action_memory)
        if nominal_memory is not None:
            nominal_ctx, _ = self.nominal_attn(q, nominal_memory, nominal_memory)
            action_ctx = action_ctx - nominal_ctx
        q = self.norm2(q + action_ctx)
        q = self.norm4(q + self.ffn(self.norm3(q)))
        return q
