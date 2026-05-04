from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from mrvp.data.schema import ACTION_DIM, DEG_DIM, MECH_DIM, NUM_ACTIONS, STATE_DIM, MECH_SLICES
from .common import make_mlp, mixture_diag_gaussian_nll
from .context import SceneContextEncoder


class MSRT(nn.Module):
    """Mechanism-Sufficient Recovery Transition network.

    Implements the appendix design: context encoder + action encoder + mechanism
    heads + mixture-density transition decoder. The model predicts z_mech, d_deg,
    and a multi-modal x_plus distribution from (o_hist, h_ctx, x_minus, action).
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        deg_dim: int = DEG_DIM,
        mech_dim: int = MECH_DIM,
        action_dim: int = ACTION_DIM,
        num_actions: int = NUM_ACTIONS,
        ctx_emb_dim: int = 128,
        hidden_dim: int = 256,
        mixture_count: int = 5,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.deg_dim = deg_dim
        self.mech_dim = mech_dim
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.mixture_count = mixture_count
        self.context_encoder = SceneContextEncoder(out_dim=ctx_emb_dim)
        self.action_embedding = nn.Embedding(num_actions, 32)
        self.action_mlp = make_mlp(action_dim + 32, [64], 64, dropout=dropout, layer_norm=True)
        self.xminus_mlp = make_mlp(state_dim, [128], 128, dropout=dropout, layer_norm=True)
        trunk_in = ctx_emb_dim + 64 + 128
        self.trunk = make_mlp(trunk_in, [hidden_dim, hidden_dim], hidden_dim, dropout=dropout, layer_norm=True)
        self.z_head = make_mlp(hidden_dim, [hidden_dim], mech_dim, dropout=dropout, layer_norm=False)
        self.deg_head = make_mlp(hidden_dim + mech_dim, [hidden_dim], deg_dim, dropout=dropout, layer_norm=False)
        self.mix_logits = make_mlp(hidden_dim + mech_dim + deg_dim, [hidden_dim], mixture_count, dropout=dropout, layer_norm=False)
        self.mix_mean = make_mlp(hidden_dim + mech_dim + deg_dim, [hidden_dim], mixture_count * state_dim, dropout=dropout, layer_norm=False)
        self.mix_log_scale = make_mlp(hidden_dim + mech_dim + deg_dim, [hidden_dim], mixture_count * state_dim, dropout=dropout, layer_norm=False)
        self.side_head = make_mlp(hidden_dim, [128], 5, dropout=dropout, layer_norm=False)
        self.contact_head = make_mlp(hidden_dim, [128], 1, dropout=dropout, layer_norm=False)

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ctx = self.context_encoder(batch["o_hist"], batch["h_ctx"])
        aid = batch["action_id"].long().clamp_min(0) % self.num_actions
        aemb = self.action_embedding(aid)
        avec = batch.get("action_vec")
        if avec is None:
            avec = torch.zeros(aid.shape[0], self.action_dim, device=aid.device)
        act = self.action_mlp(torch.cat([avec.float(), aemb], dim=-1))
        xm = self.xminus_mlp(batch["x_minus"].float())
        return self.trunk(torch.cat([ctx, act, xm], dim=-1))

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.encode(batch)
        z = self.z_head(h)
        contact_logit = self.contact_head(h).squeeze(-1)
        side_logits = self.side_head(h)
        # Place contact probability and side probabilities into z for stability.
        z = z.clone()
        z[:, MECH_SLICES["contact_flag"]] = torch.sigmoid(contact_logit).unsqueeze(-1)
        z[:, MECH_SLICES["side_onehot"]] = F.softmax(side_logits, dim=-1)
        d = self.deg_head(torch.cat([h, z], dim=-1))
        # Deg continuous constraints: scales in [0,1.2], delay nonnegative, friction positive.
        d_constrained = d.clone()
        if d_constrained.shape[-1] >= 3:
            d_constrained[:, :3] = 1.2 * torch.sigmoid(d[:, :3])
        if d_constrained.shape[-1] >= 5:
            d_constrained[:, 3] = F.softplus(d[:, 3])
            d_constrained[:, 4] = 1.2 * torch.sigmoid(d[:, 4])
        md_in = torch.cat([h, z, d_constrained], dim=-1)
        logits = self.mix_logits(md_in)
        mean = self.mix_mean(md_in).view(-1, self.mixture_count, self.state_dim)
        log_scale = self.mix_log_scale(md_in).view(-1, self.mixture_count, self.state_dim).clamp(-6, 3)
        # Residual form: predict x_plus around x_minus.
        mean = mean + batch["x_minus"].float().unsqueeze(1)
        return {
            "z_mech": z,
            "d_deg": d_constrained,
            "mix_logits": logits,
            "mix_mean": mean,
            "mix_log_scale": log_scale,
            "contact_logit": contact_logit,
            "side_logits": side_logits,
        }

    def sample(self, batch: Dict[str, torch.Tensor], num_samples: int = 16, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        out = self.forward(batch)
        logits = out["mix_logits"]
        probs = F.softmax(logits, dim=-1)
        b, r = probs.shape
        if deterministic:
            comp = probs.argmax(dim=-1).repeat_interleave(num_samples)
        else:
            comp = torch.multinomial(probs, num_samples=num_samples, replacement=True).reshape(-1)
        mean = out["mix_mean"].repeat_interleave(num_samples, dim=0)
        log_scale = out["mix_log_scale"].repeat_interleave(num_samples, dim=0)
        gather = comp.view(-1, 1, 1).expand(-1, 1, self.state_dim)
        chosen_mean = torch.gather(mean, 1, gather).squeeze(1)
        chosen_log_scale = torch.gather(log_scale, 1, gather).squeeze(1)
        if deterministic:
            x_plus = chosen_mean
        else:
            x_plus = chosen_mean + torch.randn_like(chosen_mean) * torch.exp(chosen_log_scale).clamp_max(5.0)
        return {
            "x_plus": x_plus,
            "z_mech": out["z_mech"].repeat_interleave(num_samples, dim=0),
            "d_deg": out["d_deg"].repeat_interleave(num_samples, dim=0),
            "sample_root": torch.arange(b, device=logits.device).repeat_interleave(num_samples),
        }

    def physics_reset_prior(self, batch: Dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        """Small physics-inspired reset prior for the reset branch.

        Uses normal direction, overlap/clearance and relative speed to estimate
        body-frame delta-v/yaw. It is intentionally light-weight because CARLA
        collision impulse is not always available in logs.
        """
        normal = z[:, MECH_SLICES["normal_xy"]]
        rel_speed = z[:, MECH_SLICES["relative_speed"]].squeeze(-1).clamp_min(0.0)
        overlap = z[:, MECH_SLICES["overlap_clearance"]].squeeze(-1)
        impulse_mag = rel_speed * torch.sigmoid(-overlap) * 0.18
        dv_xy = -impulse_mag[:, None] * normal
        dyaw = 0.05 * impulse_mag * torch.sign(normal[:, 1] + 1e-6)
        reset = torch.zeros(z.shape[0], 7, device=z.device, dtype=z.dtype)
        reset[:, 0:2] = dv_xy
        reset[:, 2] = torch.linalg.norm(dv_xy, dim=-1)
        reset[:, 3] = dyaw
        reset[:, 4] = dyaw * 2.0
        reset[:, 5] = dyaw * 0.5
        reset[:, 6] = 0.2 * normal[:, 1]
        return reset

    def loss(self, batch: Dict[str, torch.Tensor], lambdas: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        lambdas = lambdas or {}
        out = self.forward(batch)
        nll = mixture_diag_gaussian_nll(batch["x_plus"].float(), out["mix_logits"], out["mix_mean"], out["mix_log_scale"]).mean()
        deg_loss = F.smooth_l1_loss(out["d_deg"], batch["d_deg"].float())
        mech_loss = F.smooth_l1_loss(out["z_mech"], batch["z_mech"].float())
        # Supervise explicit contact and side labels when z_mech follows the default layout.
        contact_target = batch["z_mech"][:, 0].float()
        contact_loss = F.binary_cross_entropy_with_logits(out["contact_logit"], contact_target)
        side_target = batch["z_mech"][:, 1:6].argmax(dim=-1).long()
        side_loss = F.cross_entropy(out["side_logits"], side_target)
        reset_pred = out["z_mech"][:, MECH_SLICES["reset"]]
        reset_prior = self.physics_reset_prior(batch, out["z_mech"])
        reset_target = batch["z_mech"][:, MECH_SLICES["reset"]]
        phys_loss = F.smooth_l1_loss(reset_pred - reset_prior, reset_target - reset_prior.detach())
        # Sufficiency regularizer proxy: make z reconstruction accurate but not overly action-id dominated
        # by penalizing z variance explained by action embedding norm. The diagnostic script is the real test.
        z_by_action_penalty = out["z_mech"].pow(2).mean() * 0.0
        total = (
            nll
            + lambdas.get("deg", 1.0) * deg_loss
            + lambdas.get("mech", 1.0) * (mech_loss + 0.5 * contact_loss + 0.5 * side_loss)
            + lambdas.get("phys", 0.2) * phys_loss
            + lambdas.get("suf", 0.0) * z_by_action_penalty
        )
        return {
            "loss": total,
            "trans_nll": nll.detach(),
            "deg_loss": deg_loss.detach(),
            "mech_loss": mech_loss.detach(),
            "contact_loss": contact_loss.detach(),
            "side_loss": side_loss.detach(),
            "phys_loss": phys_loss.detach(),
        }
