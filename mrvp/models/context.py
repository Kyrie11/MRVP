from __future__ import annotations

from typing import Dict

import torch
from torch import nn
import torch.nn.functional as F

from mrvp.data.schema import ACTOR_FEAT_DIM, CTX_DIM, HIST_LEN, MAX_AGENTS
from .common import make_mlp


class SceneContextEncoder(nn.Module):
    """Compact state/map encoder used by MSRT and RPN.

    Inputs:
      o_hist: [B,T,A,F]
      h_ctx: [B,CTX_DIM]
    """

    def __init__(self, hist_len: int = HIST_LEN, max_agents: int = MAX_AGENTS, actor_feat_dim: int = ACTOR_FEAT_DIM, ctx_dim: int = CTX_DIM, hidden_dim: int = 128, out_dim: int = 128) -> None:
        super().__init__()
        self.hist_len = hist_len
        self.max_agents = max_agents
        self.actor_feat_dim = actor_feat_dim
        self.ctx_dim = ctx_dim
        self.actor_encoder = make_mlp(actor_feat_dim, [hidden_dim], hidden_dim, dropout=0.0, layer_norm=True)
        self.temporal = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.ctx_encoder = make_mlp(ctx_dim, [hidden_dim], hidden_dim, dropout=0.0, layer_norm=True)
        self.out = make_mlp(hidden_dim * 2, [hidden_dim], out_dim, dropout=0.0, layer_norm=True)

    def forward(self, o_hist: torch.Tensor, h_ctx: torch.Tensor) -> torch.Tensor:
        if o_hist.dim() == 2:
            # Already flattened; preserve a fallback path.
            b = o_hist.shape[0]
            o_hist = o_hist.view(b, 1, 1, -1)
        b, t, a, f = o_hist.shape
        valid = (o_hist[..., -1:] > 0).float() if f > 0 else torch.ones(b, t, a, 1, device=o_hist.device)
        actor_feat = self.actor_encoder(o_hist.reshape(b * t * a, f)).reshape(b, t, a, -1)
        pooled = (actor_feat * valid).sum(dim=2) / valid.sum(dim=2).clamp_min(1.0)
        _, h = self.temporal(pooled)
        hist_emb = h[-1]
        ctx_emb = self.ctx_encoder(h_ctx)
        return self.out(torch.cat([hist_emb, ctx_emb], dim=-1))


class ContextPretrainingHead(nn.Module):
    """Multi-task head for public-data scene encoder pretraining."""

    def __init__(self, encoder: SceneContextEncoder, emb_dim: int = 128) -> None:
        super().__init__()
        self.encoder = encoder
        self.future_head = make_mlp(emb_dim, [128], 2)       # ego future displacement dx,dy
        self.affordance_head = make_mlp(emb_dim, [128], 4)  # drivable width, hazard dist, route len, density

    def forward(self, o_hist: torch.Tensor, h_ctx: torch.Tensor) -> Dict[str, torch.Tensor]:
        emb = self.encoder(o_hist, h_ctx)
        return {"future_delta": self.future_head(emb), "affordance": self.affordance_head(emb), "embedding": emb}

    def loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        out = self.forward(batch["o_hist"], batch["h_ctx"])
        loss = torch.tensor(0.0, device=batch["h_ctx"].device)
        if "future_delta" in batch:
            loss = loss + F.smooth_l1_loss(out["future_delta"], batch["future_delta"])
        if "affordance" in batch:
            loss = loss + F.smooth_l1_loss(out["affordance"], batch["affordance"])
        return loss
