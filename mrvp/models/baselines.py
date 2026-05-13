from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from mrvp.data.schema import ACTION_DIM, BOTTLE_NECKS, NUM_ACTIONS, STATE_DIM
from .common import SmoothMin, make_mlp
from .context import SceneContextEncoder


class _ProfileLossMixin:
    bottlenecks: int

    def profile_loss(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], lambdas: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        lambdas = lambdas or {}
        r_star = batch["r_star"].float()
        s_star = batch["s_star"].float()
        weight = 1.0 + lambdas.get("bd", 2.0) * torch.exp(-torch.abs(s_star) / max(lambdas.get("sigma_bd", 0.5), 1e-6))
        profile = (weight * F.smooth_l1_loss(out["r_hat"], r_star, reduction="none").sum(dim=-1)).mean()
        active = F.cross_entropy(out["active_logits"], batch["b_star"].long())
        total = profile + lambdas.get("act", 0.5) * active
        return {"loss": total, "profile_loss": profile.detach(), "active_loss": active.detach()}


class DirectActionRiskNetwork(nn.Module, _ProfileLossMixin):
    """Experiment baseline: predict recoverability directly from scene/action/harm.

    It deliberately avoids the CMRT/RPFN reset-program path, testing whether
    direct action features create shortcuts.
    """

    def __init__(self, hidden_dim: int = 256, bottlenecks: int = len(BOTTLE_NECKS), scalar: bool = False) -> None:
        super().__init__()
        self.bottlenecks = bottlenecks
        self.scalar = scalar
        self.context_encoder = SceneContextEncoder(out_dim=128)
        self.action_embedding = nn.Embedding(NUM_ACTIONS, 32)
        self.action_mlp = make_mlp(ACTION_DIM + 32 + 2, [64], 64, layer_norm=True)
        self.xminus_mlp = make_mlp(STATE_DIM, [128], 128, layer_norm=True)
        self.trunk = make_mlp(128 + 64 + 128, [hidden_dim, hidden_dim], hidden_dim, dropout=0.05, layer_norm=True)
        self.margin_head = make_mlp(hidden_dim, [hidden_dim], 1 if scalar else bottlenecks)
        self.active_head = make_mlp(hidden_dim, [hidden_dim // 2], bottlenecks)
        self.smooth_min = SmoothMin(0.1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ctx = self.context_encoder(batch["o_hist"], batch["h_ctx"])
        aid = batch["action_id"].long() % NUM_ACTIONS
        act = self.action_mlp(torch.cat([batch["action_vec"].float(), self.action_embedding(aid), batch["rho_imp"].float().view(-1, 1), batch["harm_bin"].float().view(-1, 1)], dim=-1))
        xm = self.xminus_mlp(batch["x_minus"].float())
        h = self.trunk(torch.cat([ctx, act, xm], dim=-1))
        r = self.margin_head(h)
        if self.scalar:
            r = r.expand(-1, self.bottlenecks)
        logits = self.active_head(h)
        return {"r_hat": r, "active_logits": logits, "V_smooth": self.smooth_min(r, -1), "V": r.min(-1).values}

    def loss(self, batch: Dict[str, torch.Tensor], lambdas: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        return self.profile_loss(self.forward(batch), batch, lambdas)


class UnstructuredLatentRiskNetwork(nn.Module, _ProfileLossMixin):
    """Experiment baseline: generic latent transition without z-branch semantics."""

    def __init__(self, hidden_dim: int = 256, latent_dim: int = 64, bottlenecks: int = len(BOTTLE_NECKS)) -> None:
        super().__init__()
        self.bottlenecks = bottlenecks
        self.context_encoder = SceneContextEncoder(out_dim=128)
        self.action_embedding = nn.Embedding(NUM_ACTIONS, 32)
        self.latent = make_mlp(STATE_DIM + ACTION_DIM + 32 + 128, [hidden_dim], latent_dim, dropout=0.05, layer_norm=True)
        self.transition_endpoint = make_mlp(latent_dim + STATE_DIM, [128], 128, dropout=0.05, layer_norm=True)
        self.trunk = make_mlp(128 + latent_dim + 128, [hidden_dim, hidden_dim], hidden_dim, dropout=0.05, layer_norm=True)
        self.margin_head = make_mlp(hidden_dim, [hidden_dim], bottlenecks)
        self.active_head = make_mlp(hidden_dim, [hidden_dim // 2], bottlenecks)
        self.smooth_min = SmoothMin(0.1)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        ctx = self.context_encoder(batch["o_hist"], batch["h_ctx"])
        aid = batch["action_id"].long() % NUM_ACTIONS
        latent = self.latent(torch.cat([batch["x_minus"].float(), batch["action_vec"].float(), self.action_embedding(aid), ctx], dim=-1))
        endpoint = self.transition_endpoint(torch.cat([latent, batch["x_plus"].float()], dim=-1))
        h = self.trunk(torch.cat([ctx, latent, endpoint], dim=-1))
        r = self.margin_head(h)
        logits = self.active_head(h)
        return {"r_hat": r, "active_logits": logits, "V_smooth": self.smooth_min(r, -1), "V": r.min(-1).values}

    def loss(self, batch: Dict[str, torch.Tensor], lambdas: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        return self.profile_loss(self.forward(batch), batch, lambdas)
