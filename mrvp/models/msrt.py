from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from mrvp.data.schema import (
    ACTION_DIM,
    DEG_DIM,
    EVENT_TYPES,
    MECH_DIM,
    NUM_ACTIONS,
    STATE_DIM,
    TOKEN_COUNT,
    TOKEN_DIM,
    WORLD_DIM,
)
from .common import grad_reverse, make_mlp, mixture_diag_gaussian_nll
from .context import SceneContextEncoder


class MSRT(nn.Module):
    """Action-conditioned Recovery Event Tokenizer.

    This compact implementation follows the revised paper interface.  It maps
    ``(o_hist, h_ctx, x_t, action)`` to an event-anchored recovery world:
    event type/time, pre/post-event transition distribution, learned event
    tokens, local world features, and degradation.  Human-readable audit fields
    are predicted only for diagnostics and are not the RPN input.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        deg_dim: int = DEG_DIM,
        audit_dim: int = MECH_DIM,
        action_dim: int = ACTION_DIM,
        num_actions: int = NUM_ACTIONS,
        world_dim: int = WORLD_DIM,
        token_count: int = TOKEN_COUNT,
        token_dim: int = TOKEN_DIM,
        ctx_emb_dim: int = 128,
        hidden_dim: int = 256,
        mixture_count: int = 5,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.deg_dim = deg_dim
        self.audit_dim = audit_dim
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.world_dim = world_dim
        self.token_count = token_count
        self.token_dim = token_dim
        self.mixture_count = mixture_count

        self.context_encoder = SceneContextEncoder(out_dim=ctx_emb_dim)
        self.action_embedding = nn.Embedding(num_actions, 32)
        self.action_mlp = make_mlp(action_dim + 32, [64], 64, dropout=dropout, layer_norm=True)
        self.xt_mlp = make_mlp(state_dim, [128], 128, dropout=dropout, layer_norm=True)
        trunk_in = ctx_emb_dim + 64 + 128
        self.trunk = make_mlp(trunk_in, [hidden_dim, hidden_dim], hidden_dim, dropout=dropout, layer_norm=True)

        self.event_head = make_mlp(hidden_dim, [hidden_dim // 2], len(EVENT_TYPES), dropout=dropout)
        self.event_time_head = make_mlp(hidden_dim, [hidden_dim // 2], 1, dropout=dropout)
        self.token_head = make_mlp(hidden_dim, [hidden_dim], token_count * token_dim, dropout=dropout, layer_norm=False)
        self.world_head = make_mlp(hidden_dim + token_count * token_dim, [hidden_dim], world_dim, dropout=dropout, layer_norm=False)
        self.audit_head = make_mlp(hidden_dim + token_count * token_dim, [hidden_dim], audit_dim, dropout=dropout, layer_norm=False)
        self.deg_head = make_mlp(hidden_dim + token_count * token_dim, [hidden_dim], deg_dim, dropout=dropout, layer_norm=False)

        md_in = hidden_dim + token_count * token_dim + deg_dim
        self.mix_logits = make_mlp(md_in, [hidden_dim], mixture_count, dropout=dropout, layer_norm=False)
        self.mix_mean = make_mlp(md_in, [hidden_dim], mixture_count * state_dim, dropout=dropout, layer_norm=False)
        self.mix_log_scale = make_mlp(md_in, [hidden_dim], mixture_count * state_dim, dropout=dropout, layer_norm=False)

        self.target_proj = make_mlp(state_dim + deg_dim + world_dim, [hidden_dim], token_dim, dropout=dropout, layer_norm=True)
        self.token_pool_proj = make_mlp(token_dim, [hidden_dim // 2], token_dim, dropout=dropout, layer_norm=True)
        self.action_adversary = make_mlp(token_dim, [128], num_actions, dropout=dropout, layer_norm=True)

    def encode(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        ctx = self.context_encoder(batch["o_hist"], batch["h_ctx"])
        aid = batch["action_id"].long().clamp_min(0) % self.num_actions
        aemb = self.action_embedding(aid)
        avec = batch.get("action_vec")
        if avec is None:
            avec = torch.zeros(aid.shape[0], self.action_dim, device=aid.device)
        act = self.action_mlp(torch.cat([avec.float(), aemb], dim=-1))
        x_t = batch.get("x_t", batch.get("x_minus"))
        xt = self.xt_mlp(x_t.float())
        return self.trunk(torch.cat([ctx, act, xt], dim=-1))

    def _constrain_deg(self, d_raw: torch.Tensor) -> torch.Tensor:
        d = d_raw.clone()
        if d.shape[-1] >= 3:
            d[:, :3] = 1.2 * torch.sigmoid(d_raw[:, :3])
        if d.shape[-1] >= 4:
            d[:, 3] = F.softplus(d_raw[:, 3])
        if d.shape[-1] >= 5:
            d[:, 4] = 1.2 * torch.sigmoid(d_raw[:, 4])
        if d.shape[-1] >= 6:
            d[:, 5] = torch.sigmoid(d_raw[:, 5])
        return d

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        h = self.encode(batch)
        token_flat = self.token_head(h)
        tokens = token_flat.view(-1, self.token_count, self.token_dim)
        token_pool = tokens.mean(dim=1)
        token_ctx = token_flat

        event_logits = self.event_head(h)
        event_time = F.softplus(self.event_time_head(h)).squeeze(-1)
        world_plus = self.world_head(torch.cat([h, token_ctx], dim=-1))
        audit = self.audit_head(torch.cat([h, token_ctx], dim=-1))
        deg = self._constrain_deg(self.deg_head(torch.cat([h, token_ctx], dim=-1)))

        md_in = torch.cat([h, token_ctx, deg], dim=-1)
        logits = self.mix_logits(md_in)
        mean = self.mix_mean(md_in).view(-1, self.mixture_count, self.state_dim)
        log_scale = self.mix_log_scale(md_in).view(-1, self.mixture_count, self.state_dim).clamp(-6, 3)
        # Residual transition around x_t/x_minus.  x_plus labels are absolute.
        anchor = batch.get("x_minus", batch.get("x_t")).float()
        mean = mean + anchor.unsqueeze(1)

        return {
            "event_logits": event_logits,
            "event_time": event_time,
            "event_tokens": tokens,
            "token_pool": token_pool,
            "world_plus": world_plus,
            "z_mech": audit,  # audit/probe field for backward-compatible diagnostics
            "deg": deg,
            "d_deg": deg,
            "mix_logits": logits,
            "mix_mean": mean,
            "mix_log_scale": log_scale,
        }

    def sample(self, batch: Dict[str, torch.Tensor], num_samples: int = 16, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        out = self.forward(batch)
        logits = out["mix_logits"]
        probs = F.softmax(logits, dim=-1)
        b, _ = probs.shape
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
        event_probs = F.softmax(out["event_logits"], dim=-1)
        if deterministic:
            event_type_id = event_probs.argmax(dim=-1).repeat_interleave(num_samples)
        else:
            event_type_id = torch.multinomial(event_probs, num_samples=num_samples, replacement=True).reshape(-1)
        return {
            "x_plus": x_plus,
            "event_type_id": event_type_id,
            "event_time": out["event_time"].repeat_interleave(num_samples, dim=0),
            "event_tokens": out["event_tokens"].repeat_interleave(num_samples, dim=0),
            "world_plus": out["world_plus"].repeat_interleave(num_samples, dim=0),
            "z_mech": out["z_mech"].repeat_interleave(num_samples, dim=0),
            "deg": out["deg"].repeat_interleave(num_samples, dim=0),
            "d_deg": out["d_deg"].repeat_interleave(num_samples, dim=0),
            "sample_root": torch.arange(b, device=logits.device).repeat_interleave(num_samples),
        }

    def same_root_contrastive_loss(self, batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor], temperature: float = 0.1) -> torch.Tensor:
        roots = batch.get("root_id")
        if not isinstance(roots, list):
            return torch.tensor(0.0, device=out["token_pool"].device)
        device = out["token_pool"].device
        root_to_id: Dict[str, int] = {}
        ids = []
        for r in roots:
            key = str(r)
            if key not in root_to_id:
                root_to_id[key] = len(root_to_id)
            ids.append(root_to_id[key])
        root_arr = torch.tensor(ids, device=device, dtype=torch.long)
        mask = root_arr[:, None] == root_arr[None, :]
        has_distractor = mask.sum(dim=1) > 1
        if has_distractor.sum() == 0:
            return torch.tensor(0.0, device=device)
        z = F.normalize(self.token_pool_proj(out["token_pool"]), dim=-1)
        target_in = torch.cat([batch["x_plus"].float(), batch["d_deg"].float(), batch["world_plus"].float()], dim=-1)
        y = F.normalize(self.target_proj(target_in), dim=-1)
        sim = z @ y.T / max(temperature, 1e-6)
        sim = sim.masked_fill(~mask, -1e4)
        labels = torch.arange(sim.shape[0], device=device)
        return F.cross_entropy(sim[has_distractor], labels[has_distractor])

    def loss(self, batch: Dict[str, torch.Tensor], lambdas: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        lambdas = lambdas or {}
        out = self.forward(batch)
        trans_nll = mixture_diag_gaussian_nll(batch["x_plus"].float(), out["mix_logits"], out["mix_mean"], out["mix_log_scale"]).mean()
        event_loss = F.cross_entropy(out["event_logits"], batch["event_type_id"].long().clamp(0, len(EVENT_TYPES) - 1))
        event_time_loss = F.smooth_l1_loss(out["event_time"], batch["event_time"].float())
        token_loss = F.smooth_l1_loss(out["event_tokens"], batch["event_tokens"].float())
        world_loss = F.smooth_l1_loss(out["world_plus"], batch["world_plus"].float())
        deg_loss = F.smooth_l1_loss(out["d_deg"], batch["d_deg"].float())
        audit_loss = F.smooth_l1_loss(out["z_mech"], batch["z_mech"].float())
        ctr_loss = self.same_root_contrastive_loss(batch, out, temperature=float(lambdas.get("temperature", 0.1)))

        suf_weight = float(lambdas.get("suf", 0.0))
        if suf_weight > 0.0:
            adv_logits = self.action_adversary(grad_reverse(out["token_pool"], scale=1.0))
            action_adv_loss = F.cross_entropy(adv_logits, batch["action_id"].long().clamp_min(0) % self.num_actions)
        else:
            action_adv_loss = torch.tensor(0.0, device=trans_nll.device)

        total = (
            lambdas.get("nll", 1.0) * trans_nll
            + lambdas.get("event", 1.0) * (event_loss + 0.2 * event_time_loss)
            + lambdas.get("token", 0.5) * token_loss
            + lambdas.get("world", 0.5) * world_loss
            + lambdas.get("deg", 1.0) * deg_loss
            + lambdas.get("probe", 0.1) * audit_loss
            + lambdas.get("ctr", 0.1) * ctr_loss
            + suf_weight * action_adv_loss
        )
        return {
            "loss": total,
            "trans_nll": trans_nll.detach(),
            "event_loss": event_loss.detach(),
            "event_time_loss": event_time_loss.detach(),
            "token_loss": token_loss.detach(),
            "world_loss": world_loss.detach(),
            "deg_loss": deg_loss.detach(),
            "audit_loss": audit_loss.detach(),
            "contrastive_loss": ctr_loss.detach(),
            "action_adv_loss": action_adv_loss.detach(),
        }
