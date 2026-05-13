from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from mrvp.data.schema import (
    BOTTLE_NECKS,
    CONTROL_DIM,
    DEG_DIM,
    MECH_DIM,
    RECOVERY_HORIZON,
    STATE_DIM,
    STRATEGY_COUNT,
    TOKEN_COUNT,
    TOKEN_DIM,
    WORLD_DIM,
)
from .common import SmoothMin, make_mlp
from .context import SceneContextEncoder


class RecoveryProfileNetwork(nn.Module):
    """Policy-conditioned Recovery Viability Network.

    RPN consumes the MSRT event tokens and local recovery world, decodes several
    candidate recovery strategies, verifies strategy-conditioned safety margins,
    and reports the best executable strategy value.  ``r_hat`` is the margin
    profile of the best branch and remains compatible with the calibration and
    evaluation code.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        deg_dim: int = DEG_DIM,
        token_count: int = TOKEN_COUNT,
        token_dim: int = TOKEN_DIM,
        world_dim: int = WORLD_DIM,
        control_dim: int = CONTROL_DIM,
        strategy_count: int = STRATEGY_COUNT,
        recovery_horizon: int = RECOVERY_HORIZON,
        ctx_emb_dim: int = 128,
        hidden_dim: int = 256,
        bottlenecks: int = len(BOTTLE_NECKS),
        dropout: float = 0.05,
        scalar: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.deg_dim = deg_dim
        self.token_count = token_count
        self.token_dim = token_dim
        self.world_dim = world_dim
        self.control_dim = control_dim
        self.strategy_count = strategy_count
        self.recovery_horizon = recovery_horizon
        self.bottlenecks = bottlenecks
        self.scalar = scalar

        self.context_encoder = SceneContextEncoder(out_dim=ctx_emb_dim)
        self.x_encoder = make_mlp(state_dim, [128], 128, dropout=dropout, layer_norm=True)
        self.d_encoder = make_mlp(deg_dim, [64], 64, dropout=dropout, layer_norm=True)
        self.token_encoder = make_mlp(token_dim * 2, [128], 128, dropout=dropout, layer_norm=True)
        self.world_encoder = make_mlp(world_dim, [128], 128, dropout=dropout, layer_norm=True)
        trunk_in = ctx_emb_dim + 128 + 64 + 128 + 128
        self.trunk = make_mlp(trunk_in, [hidden_dim, hidden_dim], hidden_dim, dropout=dropout, layer_norm=True)

        self.strategy_u_head = make_mlp(hidden_dim, [hidden_dim], strategy_count * recovery_horizon * control_dim, dropout=dropout)
        self.strategy_traj_head = make_mlp(hidden_dim, [hidden_dim], strategy_count * (recovery_horizon + 1) * state_dim, dropout=dropout)
        margin_out = strategy_count * (1 if scalar else bottlenecks)
        self.margin_head = make_mlp(hidden_dim, [hidden_dim], margin_out, dropout=dropout, layer_norm=False)
        self.active_head = make_mlp(hidden_dim, [max(16, hidden_dim // 2)], bottlenecks, dropout=dropout, layer_norm=False)
        self.smooth_min = SmoothMin(tau=0.1)

    def _tokens_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "event_tokens" in batch:
            tokens = batch["event_tokens"].float()
            if tokens.dim() == 2:
                tokens = tokens.view(tokens.shape[0], self.token_count, self.token_dim)
            return tokens
        b = batch["x_plus"].shape[0]
        return torch.zeros(b, self.token_count, self.token_dim, device=batch["x_plus"].device, dtype=batch["x_plus"].dtype)

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ctx = self.context_encoder(batch["o_hist"], batch["h_ctx"])
        x = self.x_encoder(batch["x_plus"].float())
        d = self.d_encoder(batch.get("deg", batch["d_deg"]).float())
        tokens = self._tokens_from_batch(batch)
        token_mean = tokens.mean(dim=1)
        token_max = tokens.max(dim=1).values
        tok = self.token_encoder(torch.cat([token_mean, token_max], dim=-1))
        world = batch.get("world_plus")
        if world is None:
            world = torch.zeros(x.shape[0], self.world_dim, device=x.device, dtype=x.dtype)
        w = self.world_encoder(world.float())
        return self.trunk(torch.cat([ctx, x, d, tok, w], dim=-1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.encode(batch)
        b = h.shape[0]
        raw_controls = self.strategy_u_head(h).view(b, self.strategy_count, self.recovery_horizon, self.control_dim)
        # Paper control order is [delta, F_b, F_x].  Steering is signed;
        # brake/throttle commands are non-negative normalized forces.
        if self.control_dim >= 3:
            controls = torch.cat(
                [torch.tanh(raw_controls[..., :1]), torch.sigmoid(raw_controls[..., 1:3]), raw_controls[..., 3:]],
                dim=-1,
            )
        else:
            controls = torch.tanh(raw_controls)
        traj_delta = self.strategy_traj_head(h).view(b, self.strategy_count, self.recovery_horizon + 1, self.state_dim)
        x0 = batch["x_plus"].float().view(b, 1, 1, self.state_dim)
        traj = x0 + 0.25 * traj_delta

        margins = self.margin_head(h)
        if self.scalar:
            margins = margins.view(b, self.strategy_count, 1).expand(-1, -1, self.bottlenecks)
        else:
            margins = margins.view(b, self.strategy_count, self.bottlenecks)
        branch_v = margins.min(dim=-1).values
        branch_v_smooth = self.smooth_min(margins, dim=-1)
        best_branch = branch_v.argmax(dim=-1)
        gather_m = best_branch.view(b, 1, 1).expand(-1, 1, self.bottlenecks)
        r_hat = torch.gather(margins, 1, gather_m).squeeze(1)
        gather_u = best_branch.view(b, 1, 1, 1).expand(-1, 1, self.recovery_horizon, self.control_dim)
        gather_x = best_branch.view(b, 1, 1, 1).expand(-1, 1, self.recovery_horizon + 1, self.state_dim)
        best_u = torch.gather(controls, 1, gather_u).squeeze(1)
        best_traj = torch.gather(traj, 1, gather_x).squeeze(1)
        active_logits = self.active_head(h)
        # Smooth best across branches for ranking gradients.
        v_smooth = 0.1 * torch.logsumexp(branch_v_smooth / 0.1, dim=-1)
        return {
            "r_hat": r_hat,
            "active_logits": active_logits,
            "V_smooth": v_smooth,
            "V": branch_v.max(dim=-1).values,
            "branch_margins": margins,
            "branch_values": branch_v,
            "strategy_u": controls,
            "strategy_traj": traj,
            "best_strategy_u": best_u,
            "best_strategy_traj": best_traj,
            "best_branch": best_branch,
        }

    def _matched_branch(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], lambda_xi: float = 0.25) -> torch.Tensor:
        u_tgt = batch.get("teacher_u")
        x_tgt = batch.get("teacher_traj")
        if u_tgt is None or x_tgt is None:
            return out["best_branch"].detach()
        u_cost = (out["strategy_u"] - u_tgt.float().unsqueeze(1)).abs().mean(dim=(2, 3))
        x_cost = (out["strategy_traj"] - x_tgt.float().unsqueeze(1)).abs().mean(dim=(2, 3))
        return (u_cost + lambda_xi * x_cost).argmin(dim=-1).detach()

    def ctrl_violation_loss(self, controls: torch.Tensor, deg: torch.Tensor) -> torch.Tensor:
        d = deg.float()
        steer_scale = d[:, 0].view(-1, 1, 1).clamp_min(0.05)
        brake_scale = d[:, 1].view(-1, 1, 1).clamp_min(0.05)
        throttle_scale = d[:, 2].view(-1, 1, 1).clamp_min(0.05)
        steer_v = F.relu(controls[..., 0].abs() - steer_scale)
        brake_v = F.relu(controls[..., 1].clamp_min(0.0) - brake_scale)
        throttle_v = F.relu(controls[..., 2].clamp_min(0.0) - throttle_scale)
        return (steer_v + brake_v + throttle_v).mean()

    def dynamics_regularizer(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        traj = out["strategy_traj"]
        start_loss = F.smooth_l1_loss(traj[:, :, 0, :], batch["x_plus"].float().unsqueeze(1).expand_as(traj[:, :, 0, :]))
        smooth_loss = (traj[:, :, 2:, :] - 2 * traj[:, :, 1:-1, :] + traj[:, :, :-2, :]).abs().mean()
        return start_loss + 0.05 * smooth_loss

    def monotonicity_penalty(self, batch: Dict[str, torch.Tensor], margins: torch.Tensor) -> torch.Tensor:
        # The revised model uses learned event tokens rather than hand mechanism
        # dimensions, so the old hand-index monotonicity prior is intentionally
        # disabled.  Keep the method for CLI compatibility.
        return torch.tensor(0.0, device=margins.device)

    def loss(self, batch: Dict[str, torch.Tensor], lambdas: Optional[Dict[str, float]] = None, enable_mono: bool = False) -> Dict[str, torch.Tensor]:
        lambdas = lambdas or {}
        out = self.forward(batch)
        r_star = batch.get("m_star", batch["r_star"]).float()
        s_star = batch["s_star"].float()
        b = r_star.shape[0]
        match = self._matched_branch(out, batch, lambda_xi=float(lambdas.get("lambda_xi", 0.25)))
        gather_m = match.view(b, 1, 1).expand(-1, 1, self.bottlenecks)
        matched_margins = torch.gather(out["branch_margins"], 1, gather_m).squeeze(1)

        weight = 1.0 + lambdas.get("bd", 2.0) * torch.exp(-torch.abs(s_star) / max(lambdas.get("sigma_bd", 0.5), 1e-6))
        huber = F.smooth_l1_loss(matched_margins, r_star, reduction="none").sum(dim=-1)
        profile_loss = (weight * huber).mean()
        active_loss = F.cross_entropy(out["active_logits"], batch["b_star"].long())

        u_tgt = batch.get("teacher_u")
        x_tgt = batch.get("teacher_traj")
        if u_tgt is not None and x_tgt is not None:
            gather_u = match.view(b, 1, 1, 1).expand(-1, 1, self.recovery_horizon, self.control_dim)
            gather_x = match.view(b, 1, 1, 1).expand(-1, 1, self.recovery_horizon + 1, self.state_dim)
            matched_u = torch.gather(out["strategy_u"], 1, gather_u).squeeze(1)
            matched_traj = torch.gather(out["strategy_traj"], 1, gather_x).squeeze(1)
            strat_loss = F.smooth_l1_loss(matched_u, u_tgt.float()) + lambdas.get("lambda_xi", 0.25) * F.smooth_l1_loss(matched_traj, x_tgt.float())
        else:
            strat_loss = torch.tensor(0.0, device=r_star.device)
        dyn_loss = self.dynamics_regularizer(out, batch)
        ctrl_loss = self.ctrl_violation_loss(out["strategy_u"], batch.get("deg", batch["d_deg"]))
        mono_loss = self.monotonicity_penalty(batch, matched_margins) if enable_mono else torch.tensor(0.0, device=r_star.device)

        total = (
            profile_loss
            + lambdas.get("act", 0.5) * active_loss
            + lambdas.get("strat", 0.5) * strat_loss
            + lambdas.get("dyn", 0.05) * dyn_loss
            + lambdas.get("ctrl", 0.05) * ctrl_loss
            + lambdas.get("mono", 0.0) * mono_loss
        )
        return {
            "loss": total,
            "profile_loss": profile_loss.detach(),
            "active_loss": active_loss.detach(),
            "strategy_loss": strat_loss.detach(),
            "dyn_loss": dyn_loss.detach(),
            "ctrl_loss": ctrl_loss.detach(),
            "mono_loss": mono_loss.detach(),
        }


def ordering_loss(v_i: torch.Tensor, v_j: torch.Tensor, s_i: torch.Tensor, s_j: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    sign = torch.sign(s_i - s_j).clamp(min=-1, max=1)
    keep = sign.abs() > 0
    if keep.sum() == 0:
        return torch.tensor(0.0, device=v_i.device)
    loss = F.relu(margin - sign[keep] * (v_i[keep] - v_j[keep]))
    return loss.mean()
