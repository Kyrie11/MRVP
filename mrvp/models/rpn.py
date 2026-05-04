from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from mrvp.data.schema import BOTTLE_NECKS, CTX_DIM, DEG_DIM, MECH_DIM, STATE_DIM, MECH_SLICES
from .common import SmoothMin, make_mlp
from .context import SceneContextEncoder


class RecoveryProfileNetwork(nn.Module):
    """Bottleneck Recovery Profile Network.

    Encodes x_plus, degradation, mechanism and context, then predicts five
    controller-conditioned signed margins and the active bottleneck.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        deg_dim: int = DEG_DIM,
        mech_dim: int = MECH_DIM,
        ctx_emb_dim: int = 128,
        hidden_dim: int = 256,
        bottlenecks: int = len(BOTTLE_NECKS),
        dropout: float = 0.05,
        scalar: bool = False,
    ) -> None:
        super().__init__()
        self.bottlenecks = bottlenecks
        self.scalar = scalar
        self.context_encoder = SceneContextEncoder(out_dim=ctx_emb_dim)
        self.x_encoder = make_mlp(state_dim, [128], 128, dropout=dropout, layer_norm=True)
        self.d_encoder = make_mlp(deg_dim, [64], 64, dropout=dropout, layer_norm=True)
        self.z_encoder = make_mlp(mech_dim, [128], 128, dropout=dropout, layer_norm=True)
        self.trunk = make_mlp(ctx_emb_dim + 128 + 64 + 128, [hidden_dim, hidden_dim], hidden_dim, dropout=dropout, layer_norm=True)
        if scalar:
            self.margin_head = make_mlp(hidden_dim, [hidden_dim], 1, dropout=dropout, layer_norm=False)
        else:
            self.margin_head = make_mlp(hidden_dim, [hidden_dim], bottlenecks, dropout=dropout, layer_norm=False)
        self.active_head = make_mlp(hidden_dim, [hidden_dim // 2], bottlenecks, dropout=dropout, layer_norm=False)
        self.smooth_min = SmoothMin(tau=0.1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ctx = self.context_encoder(batch["o_hist"], batch["h_ctx"])
        x = self.x_encoder(batch["x_plus"].float())
        d = self.d_encoder(batch["d_deg"].float())
        z = self.z_encoder(batch["z_mech"].float())
        h = self.trunk(torch.cat([ctx, x, d, z], dim=-1))
        margins = self.margin_head(h)
        if self.scalar:
            margins = margins.expand(-1, self.bottlenecks)
        active_logits = self.active_head(h)
        v_smooth = self.smooth_min(margins, dim=-1)
        v_hard = margins.min(dim=-1).values
        return {"r_hat": margins, "active_logits": active_logits, "V_smooth": v_smooth, "V": v_hard}

    def monotonicity_penalty(self, batch: Dict[str, torch.Tensor], margins: torch.Tensor) -> torch.Tensor:
        """Soft local monotonicity priors from the appendix.

        Uses finite differences by perturbing mechanism features:
        - larger road clearance should not lower road margin;
        - larger yaw-rate reset magnitude should not improve stability;
        - larger residual authority should not lower control margin.
        """
        if not batch["z_mech"].requires_grad:
            return torch.tensor(0.0, device=margins.device)
        z = batch["z_mech"]
        road_margin = margins[:, 1].sum()
        stab_margin = margins[:, 2].sum()
        ctrl_margin = margins[:, 3].sum()
        grads = torch.autograd.grad(
            road_margin + stab_margin + ctrl_margin,
            z,
            retain_graph=True,
            create_graph=True,
            allow_unused=True,
        )[0]
        if grads is None:
            return torch.tensor(0.0, device=margins.device)
        clearance_idx = 19  # affordance clearance
        yaw_reset_idx = 15  # yaw-rate reset in reset slice
        steer_auth_idx = 24
        p = F.relu(-grads[:, clearance_idx]).mean()
        p = p + F.relu(grads[:, yaw_reset_idx]).mean()
        p = p + F.relu(-grads[:, steer_auth_idx]).mean()
        return p

    def loss(self, batch: Dict[str, torch.Tensor], lambdas: Optional[Dict[str, float]] = None, enable_mono: bool = False) -> Dict[str, torch.Tensor]:
        lambdas = lambdas or {}
        # Only enable autograd wrt z when monotonicity is requested.
        if enable_mono:
            batch = dict(batch)
            batch["z_mech"] = batch["z_mech"].detach().clone().requires_grad_(True)
        out = self.forward(batch)
        r_star = batch["r_star"].float()
        s_star = batch["s_star"].float()
        weight = 1.0 + lambdas.get("bd", 2.0) * torch.exp(-torch.abs(s_star) / max(lambdas.get("sigma_bd", 0.5), 1e-6))
        huber = F.smooth_l1_loss(out["r_hat"], r_star, reduction="none").sum(dim=-1)
        profile_loss = (weight * huber).mean()
        act_loss = F.cross_entropy(out["active_logits"], batch["b_star"].long())
        mono_loss = self.monotonicity_penalty(batch, out["r_hat"]) if enable_mono else torch.tensor(0.0, device=r_star.device)
        total = profile_loss + lambdas.get("act", 0.5) * act_loss + lambdas.get("mono", 0.0) * mono_loss
        return {
            "loss": total,
            "profile_loss": profile_loss.detach(),
            "active_loss": act_loss.detach(),
            "mono_loss": mono_loss.detach(),
        }


def ordering_loss(v_i: torch.Tensor, v_j: torch.Tensor, s_i: torch.Tensor, s_j: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    sign = torch.sign(s_i - s_j).clamp(min=-1, max=1)
    keep = sign.abs() > 0
    if keep.sum() == 0:
        return torch.tensor(0.0, device=v_i.device)
    loss = F.relu(margin - sign[keep] * (v_i[keep] - v_j[keep]))
    return loss.mean()
