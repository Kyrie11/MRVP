from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from mrvp.data.schema import (
    ACTION_DIM,
    CERTIFICATE_DIM,
    CTX_DIM,
    DEG_DIM,
    EVENT_TYPES,
    HIST_LEN,
    MAX_AGENTS,
    ACTOR_FEAT_DIM,
    MECH_DIM,
    NUM_ACTIONS,
    RECOVERY_WORLD_DIM,
    RESET_SLOT_COUNT,
    RESET_SLOT_DIM,
    RESET_UNCERTAINTY_DIM,
    STATE_DIM,
)
from .common import grad_reverse, make_mlp, mixture_diag_gaussian_nll
from .reset_memory import ActionInterventionEncoder, CounterfactualResetSlotLayer, RootSceneMemoryEncoder


class CounterfactualMotionResetTokenizer(nn.Module):
    """CMRT: maps a root scene and action prefix to reset-problem distribution.

    Main outputs are reset slots, reset boundary time, a mixture distribution over
    reset state, degradation, recovery world and reset uncertainty.  Diagnostic
    audit event logits are optional and disabled in the default loss.
    """

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        deg_dim: int = DEG_DIM,
        audit_dim: int = MECH_DIM,
        action_dim: int = ACTION_DIM,
        num_actions: int = NUM_ACTIONS,
        recovery_world_dim: int = RECOVERY_WORLD_DIM,
        reset_slot_count: int = RESET_SLOT_COUNT,
        reset_slot_dim: int = RESET_SLOT_DIM,
        reset_uncertainty_dim: int = RESET_UNCERTAINTY_DIM,
        ctx_emb_dim: int = 128,
        hidden_dim: int = 256,
        mixture_count: int = 5,
        prefix_horizon: int = 20,
        slot_layers: int = 2,
        dropout: float = 0.05,
        # Legacy argument aliases.
        world_dim: Optional[int] = None,
        token_count: Optional[int] = None,
        token_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        if world_dim is not None:
            recovery_world_dim = world_dim
        if token_count is not None:
            reset_slot_count = token_count
        if token_dim is not None:
            reset_slot_dim = token_dim
        self.state_dim = state_dim
        self.deg_dim = deg_dim
        self.audit_dim = audit_dim
        self.action_dim = action_dim
        self.num_actions = num_actions
        self.recovery_world_dim = recovery_world_dim
        self.reset_slot_count = reset_slot_count
        self.reset_slot_dim = reset_slot_dim
        self.reset_uncertainty_dim = reset_uncertainty_dim
        self.mixture_count = mixture_count
        self.prefix_horizon = prefix_horizon
        # Legacy attributes expected by old scripts.
        self.world_dim = recovery_world_dim
        self.token_count = reset_slot_count
        self.token_dim = reset_slot_dim

        model_dim = reset_slot_dim
        self.root_memory = RootSceneMemoryEncoder(
            state_dim=state_dim,
            hist_len=HIST_LEN,
            max_agents=MAX_AGENTS,
            actor_feat_dim=ACTOR_FEAT_DIM,
            ctx_dim=CTX_DIM,
            model_dim=model_dim,
            hidden_dim=max(64, hidden_dim // 2),
        )
        self.action_memory = ActionInterventionEncoder(
            state_dim=state_dim,
            action_dim=action_dim,
            num_actions=num_actions,
            model_dim=model_dim,
            hidden_dim=max(64, hidden_dim // 2),
            prefix_horizon=prefix_horizon,
        )
        self.learned_reset_slots = nn.Parameter(torch.randn(reset_slot_count, model_dim) * 0.02)
        self.action_slot_bias = make_mlp(action_dim, [max(64, hidden_dim // 2)], model_dim, dropout=dropout, layer_norm=True)
        self.layers = nn.ModuleList([CounterfactualResetSlotLayer(model_dim, num_heads=4, dropout=dropout) for _ in range(slot_layers)])
        self.slot_pool = make_mlp(model_dim * 2, [hidden_dim], hidden_dim, dropout=dropout, layer_norm=True)
        self.xt_mlp = make_mlp(state_dim, [hidden_dim // 2], hidden_dim // 2, dropout=dropout, layer_norm=True)
        self.trunk = make_mlp(hidden_dim + hidden_dim // 2, [hidden_dim], hidden_dim, dropout=dropout, layer_norm=True)

        self.reset_time_logits_head = make_mlp(hidden_dim, [hidden_dim // 2], prefix_horizon, dropout=dropout)
        self.recovery_world_head = make_mlp(hidden_dim + reset_slot_count * model_dim, [hidden_dim], recovery_world_dim, dropout=dropout)
        self.audit_head = make_mlp(hidden_dim + reset_slot_count * model_dim, [hidden_dim], audit_dim, dropout=dropout)
        self.audit_event_head = make_mlp(hidden_dim, [hidden_dim // 2], len(EVENT_TYPES), dropout=dropout)
        self.degradation_head = make_mlp(hidden_dim + reset_slot_count * model_dim, [hidden_dim], deg_dim, dropout=dropout)
        self.reset_uncertainty_head = make_mlp(hidden_dim + reset_slot_count * model_dim, [hidden_dim // 2], reset_uncertainty_dim, dropout=dropout)
        md_in = hidden_dim + reset_slot_count * model_dim + deg_dim
        self.reset_state_mix_logits_head = make_mlp(md_in, [hidden_dim], mixture_count, dropout=dropout)
        self.reset_state_mix_mean_head = make_mlp(md_in, [hidden_dim], mixture_count * state_dim, dropout=dropout)
        self.reset_state_mix_log_scale_head = make_mlp(md_in, [hidden_dim], mixture_count * state_dim, dropout=dropout)

        target_dim = state_dim + deg_dim + recovery_world_dim + CERTIFICATE_DIM + 1
        self.target_proj = make_mlp(target_dim, [hidden_dim], model_dim, dropout=dropout, layer_norm=True)
        self.slot_pool_proj = make_mlp(model_dim, [hidden_dim // 2], model_dim, dropout=dropout, layer_norm=True)
        self.action_adversary = make_mlp(model_dim, [128], num_actions, dropout=dropout, layer_norm=True)

    def _nominal_action_vec(self, action_vec: torch.Tensor) -> torch.Tensor:
        nominal = torch.zeros_like(action_vec)
        if nominal.shape[-1] >= 2:
            nominal[:, 1] = 0.20
        if nominal.shape[-1] >= 4:
            nominal[:, 3] = action_vec[:, 3].clamp_min(0.1)
        return nominal

    def _constrain_degradation(self, d_raw: torch.Tensor) -> torch.Tensor:
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

    def encode_slots(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        x_t = batch.get("x_t", batch.get("x_minus", batch.get("reset_state"))).float()
        o_hist = batch["o_hist"].float()
        h_ctx = batch["h_ctx"].float()
        action_vec = batch.get("action_vec")
        if action_vec is None:
            action_vec = torch.zeros(x_t.shape[0], self.action_dim, device=x_t.device, dtype=x_t.dtype)
        action_vec = action_vec.float()
        action_id = batch.get("action_id")
        if action_id is None:
            action_id = torch.zeros(x_t.shape[0], device=x_t.device, dtype=torch.long)
        scene_memory = self.root_memory(o_hist, h_ctx, x_t)
        action_memory = self.action_memory(x_t, action_vec, action_id.long())
        nominal_memory = self.action_memory(x_t, self._nominal_action_vec(action_vec), torch.zeros_like(action_id.long()))
        b = x_t.shape[0]
        slots = self.learned_reset_slots.unsqueeze(0).expand(b, -1, -1)
        slots = slots + self.action_slot_bias(action_vec).unsqueeze(1)
        for layer in self.layers:
            slots = layer(slots, scene_memory, action_memory, nominal_memory)
        return slots

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        slots = self.encode_slots(batch)
        b = slots.shape[0]
        slot_mean = slots.mean(dim=1)
        slot_max = slots.max(dim=1).values
        pooled = self.slot_pool(torch.cat([slot_mean, slot_max], dim=-1))
        x_t = batch.get("x_t", batch.get("x_minus", batch.get("reset_state"))).float()
        h = self.trunk(torch.cat([pooled, self.xt_mlp(x_t)], dim=-1))
        slot_flat = slots.reshape(b, -1)
        decoder_ctx = torch.cat([h, slot_flat], dim=-1)
        time_logits = self.reset_time_logits_head(h)
        time_probs = F.softmax(time_logits, dim=-1)
        time_grid = torch.linspace(0.0, 1.0, self.prefix_horizon, device=time_logits.device, dtype=time_logits.dtype)
        reset_time = (time_probs * time_grid[None, :]).sum(dim=-1)
        recovery_world_vec = self.recovery_world_head(decoder_ctx)
        audit = self.audit_head(decoder_ctx)
        degradation = self._constrain_degradation(self.degradation_head(decoder_ctx))
        uncertainty = F.softplus(self.reset_uncertainty_head(decoder_ctx)) + 1e-4
        md_in = torch.cat([decoder_ctx, degradation], dim=-1)
        logits = self.reset_state_mix_logits_head(md_in)
        mean = self.reset_state_mix_mean_head(md_in).view(b, self.mixture_count, self.state_dim)
        log_scale = self.reset_state_mix_log_scale_head(md_in).view(b, self.mixture_count, self.state_dim).clamp(-6.0, 3.0)
        mean = mean + x_t.unsqueeze(1)
        audit_event_logits = self.audit_event_head(h)
        return {
            "reset_slots": slots,
            "slot_pool": slot_mean,
            "reset_time_logits": time_logits,
            "reset_time": reset_time,
            "reset_state_mix_logits": logits,
            "reset_state_mix_mean": mean,
            "reset_state_mix_log_scale": log_scale,
            "degradation": degradation,
            "recovery_world_vec": recovery_world_vec,
            "reset_uncertainty": uncertainty,
            "audit_mech": audit,
            "audit_event_logits": audit_event_logits,
            # Legacy aliases.
            "event_logits": audit_event_logits,
            "event_time": reset_time,
            "event_tokens": slots,
            "world_plus": recovery_world_vec,
            "z_mech": audit,
            "deg": degradation,
            "d_deg": degradation,
            "mix_logits": logits,
            "mix_mean": mean,
            "mix_log_scale": log_scale,
        }

    def sample_reset_problems(self, batch: Dict[str, torch.Tensor], num_samples: int = 16, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        out = self.forward(batch)
        logits = out["reset_state_mix_logits"]
        probs = F.softmax(logits, dim=-1)
        b, _ = probs.shape
        if deterministic:
            comp = probs.argmax(dim=-1).repeat_interleave(num_samples)
        else:
            comp = torch.multinomial(probs, num_samples=num_samples, replacement=True).reshape(-1)
        mean = out["reset_state_mix_mean"].repeat_interleave(num_samples, dim=0)
        log_scale = out["reset_state_mix_log_scale"].repeat_interleave(num_samples, dim=0)
        gather = comp.view(-1, 1, 1).expand(-1, 1, self.state_dim)
        chosen_mean = torch.gather(mean, 1, gather).squeeze(1)
        chosen_log_scale = torch.gather(log_scale, 1, gather).squeeze(1)
        if deterministic:
            reset_state = chosen_mean
        else:
            reset_state = chosen_mean + torch.randn_like(chosen_mean) * torch.exp(chosen_log_scale).clamp_max(5.0)
        repeated = {
            "reset_state": reset_state,
            "reset_slots": out["reset_slots"].repeat_interleave(num_samples, dim=0),
            "recovery_world_vec": out["recovery_world_vec"].repeat_interleave(num_samples, dim=0),
            "degradation": out["degradation"].repeat_interleave(num_samples, dim=0),
            "reset_time": out["reset_time"].repeat_interleave(num_samples, dim=0),
            "reset_uncertainty": out["reset_uncertainty"].repeat_interleave(num_samples, dim=0),
            "sample_root": torch.arange(b, device=logits.device).repeat_interleave(num_samples),
        }
        repeated.update(
            {
                "x_plus": repeated["reset_state"],
                "event_tokens": repeated["reset_slots"],
                "world_plus": repeated["recovery_world_vec"],
                "deg": repeated["degradation"],
                "d_deg": repeated["degradation"],
                "event_time": repeated["reset_time"],
                "z_mech": out["audit_mech"].repeat_interleave(num_samples, dim=0),
            }
        )
        return repeated

    def sample(self, batch: Dict[str, torch.Tensor], num_samples: int = 16, deterministic: bool = False) -> Dict[str, torch.Tensor]:
        return self.sample_reset_problems(batch, num_samples=num_samples, deterministic=deterministic)

    def same_root_counterfactual_loss(self, batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor], temperature: float = 0.1) -> torch.Tensor:
        roots = batch.get("root_id")
        if not isinstance(roots, list):
            return torch.tensor(0.0, device=out["reset_slots"].device)
        device = out["reset_slots"].device
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
        if int(has_distractor.sum().item()) == 0:
            return torch.tensor(0.0, device=device)
        z = F.normalize(self.slot_pool_proj(out["reset_slots"].mean(dim=1)), dim=-1)
        m_star = batch.get("m_star")
        if m_star is None:
            m_star = torch.zeros(z.shape[0], CERTIFICATE_DIM, device=device, dtype=z.dtype)
        s_star = batch.get("s_star")
        if s_star is None:
            s_star = m_star.min(dim=-1).values
        target_in = torch.cat(
            [
                batch["reset_state"].float(),
                batch["degradation"].float(),
                batch["recovery_world_vec"].float(),
                m_star.float(),
                s_star.float().view(-1, 1),
            ],
            dim=-1,
        )
        y = F.normalize(self.target_proj(target_in), dim=-1)
        sim = z @ y.T / max(float(temperature), 1e-6)
        sim = sim.masked_fill(~mask, -1e4)
        labels = torch.arange(sim.shape[0], device=device)
        return F.cross_entropy(sim[has_distractor], labels[has_distractor])

    # Backward-compatible method name.
    def same_root_contrastive_loss(self, batch: Dict[str, torch.Tensor], out: Dict[str, torch.Tensor], temperature: float = 0.1) -> torch.Tensor:
        return self.same_root_counterfactual_loss(batch, out, temperature)

    def loss(self, batch: Dict[str, torch.Tensor], lambdas: Optional[Dict[str, float]] = None) -> Dict[str, torch.Tensor]:
        lambdas = lambdas or {}
        out = self.forward(batch)
        reset_nll = mixture_diag_gaussian_nll(
            batch["reset_state"].float(),
            out["reset_state_mix_logits"],
            out["reset_state_mix_mean"],
            out["reset_state_mix_log_scale"],
        ).mean()
        reset_time_loss = F.smooth_l1_loss(out["reset_time"], batch["reset_time"].float())
        degradation_loss = F.smooth_l1_loss(out["degradation"], batch["degradation"].float())
        world_loss = F.smooth_l1_loss(out["recovery_world_vec"], batch["recovery_world_vec"].float())
        cf_loss = self.same_root_counterfactual_loss(batch, out, temperature=float(lambdas.get("temperature", lambdas.get("contrastive_temperature", 0.1))))
        uncertainty_loss = torch.relu(out["reset_state_mix_log_scale"].abs() - 4.0).mean() + 0.01 * out["reset_uncertainty"].mean()
        if float(lambdas.get("slot_distill", lambdas.get("token", 0.0))) > 0.0:
            mask = batch.get("has_reset_slots_target")
            if mask is not None and torch.as_tensor(mask).float().sum() > 0:
                m = torch.as_tensor(mask, device=out["reset_slots"].device).float().view(-1, 1, 1)
                slot_distill_loss = (F.smooth_l1_loss(out["reset_slots"], batch["reset_slots_target"].float(), reduction="none") * m).sum() / m.sum().clamp_min(1.0)
            else:
                slot_distill_loss = torch.tensor(0.0, device=reset_nll.device)
        else:
            slot_distill_loss = torch.tensor(0.0, device=reset_nll.device)
        audit_event_loss = F.cross_entropy(out["audit_event_logits"], batch["audit_event_type_id"].long().clamp(0, len(EVENT_TYPES) - 1))
        audit_loss = F.smooth_l1_loss(out["audit_mech"], batch["audit_mech"].float())
        suf_weight = float(lambdas.get("suf", 0.0))
        if suf_weight > 0.0:
            adv_logits = self.action_adversary(grad_reverse(out["reset_slots"].mean(dim=1), scale=1.0))
            action_adv_loss = F.cross_entropy(adv_logits, batch["action_id"].long().clamp_min(0) % self.num_actions)
        else:
            action_adv_loss = torch.tensor(0.0, device=reset_nll.device)
        reset_weight = lambdas.get("reset", lambdas.get("nll", 1.0))
        total = (
            reset_weight * (reset_nll + 0.2 * reset_time_loss)
            + lambdas.get("world", 0.5) * world_loss
            + lambdas.get("degradation", lambdas.get("deg", 1.0)) * degradation_loss
            + lambdas.get("counterfactual", lambdas.get("ctr", 0.2)) * cf_loss
            + lambdas.get("uncertainty", 1.0) * uncertainty_loss
            + lambdas.get("audit_event", lambdas.get("event", 0.0)) * audit_event_loss
            + lambdas.get("slot_distill", lambdas.get("token", 0.0)) * slot_distill_loss
            + lambdas.get("probe", 0.0) * audit_loss
            + suf_weight * action_adv_loss
        )
        return {
            "loss": total,
            "reset_nll": reset_nll.detach(),
            "reset_time_loss": reset_time_loss.detach(),
            "degradation_loss": degradation_loss.detach(),
            "world_loss": world_loss.detach(),
            "counterfactual_loss": cf_loss.detach(),
            "uncertainty_loss": uncertainty_loss.detach(),
            "audit_event_loss": audit_event_loss.detach(),
            "slot_distill_loss": slot_distill_loss.detach(),
            "audit_loss": audit_loss.detach(),
            "action_adv_loss": action_adv_loss.detach(),
            # Legacy metric aliases.
            "trans_nll": reset_nll.detach(),
            "event_loss": audit_event_loss.detach(),
            "event_time_loss": reset_time_loss.detach(),
            "token_loss": slot_distill_loss.detach(),
            "deg_loss": degradation_loss.detach(),
            "contrastive_loss": cf_loss.detach(),
        }


CMRT = CounterfactualMotionResetTokenizer
MSRT = CounterfactualMotionResetTokenizer
