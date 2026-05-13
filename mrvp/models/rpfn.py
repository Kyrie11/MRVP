from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import nn
import torch.nn.functional as F

from mrvp.data.schema import (
    BOTTLE_NECKS,
    CONTROL_DIM,
    DEG_DIM,
    PROGRAM_COUNT,
    RECOVERY_HORIZON,
    RECOVERY_WORLD_DIM,
    RESET_SLOT_COUNT,
    RESET_SLOT_DIM,
    STATE_DIM,
)
from mrvp.dynamics.degraded_bicycle import degraded_bicycle_step
from mrvp.safety.certificates import compute_program_certificate
from .common import make_mlp
from .recovery_programs import FunnelHead, ProgramTokenEncoder, ResidualPolicyHead


class RecoveryProgramFunnelNetwork(nn.Module):
    """RPFN: synthesizes closed-loop recovery programs and funnel certificates."""

    def __init__(
        self,
        state_dim: int = STATE_DIM,
        deg_dim: int = DEG_DIM,
        reset_slot_count: int = RESET_SLOT_COUNT,
        reset_slot_dim: int = RESET_SLOT_DIM,
        recovery_world_dim: int = RECOVERY_WORLD_DIM,
        control_dim: int = CONTROL_DIM,
        program_count: int = PROGRAM_COUNT,
        recovery_horizon: int = RECOVERY_HORIZON,
        hidden_dim: int = 256,
        bottlenecks: int = len(BOTTLE_NECKS),
        dropout: float = 0.05,
        scalar: bool = False,
        dt: float = 0.1,
        # Legacy aliases.
        token_count: Optional[int] = None,
        token_dim: Optional[int] = None,
        world_dim: Optional[int] = None,
        strategy_count: Optional[int] = None,
        ctx_emb_dim: int = 128,
    ) -> None:
        super().__init__()
        if token_count is not None:
            reset_slot_count = token_count
        if token_dim is not None:
            reset_slot_dim = token_dim
        if world_dim is not None:
            recovery_world_dim = world_dim
        if strategy_count is not None:
            program_count = strategy_count
        self.state_dim = state_dim
        self.deg_dim = deg_dim
        self.reset_slot_count = reset_slot_count
        self.reset_slot_dim = reset_slot_dim
        self.recovery_world_dim = recovery_world_dim
        self.control_dim = control_dim
        self.program_count = program_count
        self.recovery_horizon = recovery_horizon
        self.bottlenecks = bottlenecks
        self.scalar = scalar
        self.dt = dt
        # Legacy attrs.
        self.token_count = reset_slot_count
        self.token_dim = reset_slot_dim
        self.world_dim = recovery_world_dim
        self.strategy_count = program_count

        self.program_encoder = ProgramTokenEncoder(
            state_dim=state_dim,
            deg_dim=deg_dim,
            world_dim=recovery_world_dim,
            slot_dim=reset_slot_dim,
            program_count=program_count,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.target_head = make_mlp(reset_slot_dim, [hidden_dim, hidden_dim // 2], 5, dropout=dropout, layer_norm=True)
        self.policy_head = ResidualPolicyHead(
            state_dim=state_dim,
            deg_dim=deg_dim,
            world_dim=recovery_world_dim,
            program_dim=reset_slot_dim,
            target_dim=5,
            control_dim=control_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
        )
        self.funnel_head = FunnelHead(state_dim=state_dim, program_dim=reset_slot_dim, target_dim=5, hidden_dim=hidden_dim, dropout=dropout)
        self.active_head = make_mlp(reset_slot_dim, [max(16, hidden_dim // 2)], bottlenecks, dropout=dropout)

    def _tensor(self, batch: Dict[str, torch.Tensor], primary: str, legacy: str) -> torch.Tensor:
        if primary in batch:
            return batch[primary].float()
        return batch[legacy].float()

    def _reset_slots_from_batch(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        if "reset_slots" in batch:
            slots = batch["reset_slots"].float()
        elif "event_tokens" in batch:
            slots = batch["event_tokens"].float()
        else:
            reset_state = self._tensor(batch, "reset_state", "x_plus")
            slots = torch.zeros(reset_state.shape[0], self.reset_slot_count, self.reset_slot_dim, device=reset_state.device, dtype=reset_state.dtype)
        if slots.dim() == 2:
            slots = slots.view(slots.shape[0], self.reset_slot_count, self.reset_slot_dim)
        if slots.shape[1] != self.reset_slot_count or slots.shape[2] != self.reset_slot_dim:
            fixed = torch.zeros(slots.shape[0], self.reset_slot_count, self.reset_slot_dim, device=slots.device, dtype=slots.dtype)
            k = min(self.reset_slot_count, slots.shape[1])
            d = min(self.reset_slot_dim, slots.shape[2])
            fixed[:, :k, :d] = slots[:, :k, :d]
            slots = fixed
        return slots

    def _make_targets(self, reset_state: torch.Tensor, raw: torch.Tensor) -> torch.Tensor:
        target = raw.clone()
        target_x = reset_state[:, None, 0] + 4.0 + 10.0 * torch.sigmoid(raw[..., 0])
        target_y = 2.0 * torch.tanh(raw[..., 1])
        target_yaw = torch.tanh(raw[..., 2])
        target_speed = F.softplus(raw[..., 3])
        target_type = raw[..., 4]
        return torch.stack([target_x, target_y, target_yaw, target_speed, target_type], dim=-1)

    def rollout_programs(
        self,
        reset_state: torch.Tensor,
        degradation: torch.Tensor,
        recovery_world_vec: torch.Tensor,
        program_tokens: torch.Tensor,
        program_targets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, l, _ = program_tokens.shape
        x = reset_state[:, None, :].expand(b, l, self.state_dim).reshape(b * l, self.state_dim)
        d = degradation[:, None, :].expand(b, l, self.deg_dim).reshape(b * l, self.deg_dim)
        w = recovery_world_vec[:, None, :].expand(b, l, self.recovery_world_dim).reshape(b * l, self.recovery_world_dim)
        p = program_tokens.reshape(b * l, self.reset_slot_dim)
        g = program_targets.reshape(b * l, program_targets.shape[-1])
        states = [x.view(b, l, self.state_dim)]
        controls = []
        for tau in range(self.recovery_horizon):
            u = self.policy_head(x, p, g, w, d, tau)
            x = degraded_bicycle_step(x, u, d, dt=self.dt)
            controls.append(u.view(b, l, self.control_dim))
            states.append(x.view(b, l, self.state_dim))
        return torch.stack(states, dim=2), torch.stack(controls, dim=2)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        reset_state = self._tensor(batch, "reset_state", "x_plus")
        degradation = self._tensor(batch, "degradation", "deg")
        recovery_world_vec = self._tensor(batch, "recovery_world_vec", "world_plus")
        reset_slots = self._reset_slots_from_batch(batch)
        program_tokens = self.program_encoder(reset_state, reset_slots, recovery_world_vec, degradation)
        raw_targets = self.target_head(program_tokens)
        program_targets = self._make_targets(reset_state, raw_targets)
        program_rollouts, program_controls = self.rollout_programs(reset_state, degradation, recovery_world_vec, program_tokens, program_targets)
        funnel_values = self.funnel_head(program_rollouts, program_tokens, program_targets)
        certificate_margins, program_certificates, margin_traces = compute_program_certificate(
            program_rollouts,
            program_controls,
            funnel_values,
            recovery_world_vec,
            degradation,
            program_targets,
        )
        best_certificate, best_program = program_certificates.max(dim=1)
        gather_profile = best_program.view(-1, 1, 1).expand(-1, 1, certificate_margins.shape[-1])
        best_profile = torch.gather(certificate_margins, 1, gather_profile).squeeze(1)
        gather_u = best_program.view(-1, 1, 1, 1).expand(-1, 1, self.recovery_horizon, self.control_dim)
        gather_x = best_program.view(-1, 1, 1, 1).expand(-1, 1, self.recovery_horizon + 1, self.state_dim)
        best_controls = torch.gather(program_controls, 1, gather_u).squeeze(1)
        best_rollout = torch.gather(program_rollouts, 1, gather_x).squeeze(1)
        active_logits = self.active_head(program_tokens.mean(dim=1))
        return {
            "program_tokens": program_tokens,
            "program_targets": program_targets,
            "program_controls": program_controls,
            "program_rollouts": program_rollouts,
            "funnel_values": funnel_values,
            "margin_traces": margin_traces,
            "certificate_margins": certificate_margins,
            "program_certificates": program_certificates,
            "best_certificate": best_certificate,
            "best_program": best_program,
            "best_profile": best_profile,
            "active_logits": active_logits,
            "V_smooth": best_certificate,
            # Legacy aliases.
            "strategy_u": program_controls,
            "strategy_traj": program_rollouts,
            "branch_margins": certificate_margins,
            "branch_values": program_certificates,
            "V": best_certificate,
            "r_hat": best_profile,
            "best_branch": best_program,
            "best_strategy_u": best_controls,
            "best_strategy_traj": best_rollout,
        }

    def _matched_program(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], lambda_xi: float = 0.25) -> torch.Tensor:
        u_tgt = batch.get("teacher_u")
        x_tgt = batch.get("teacher_traj")
        if u_tgt is None or x_tgt is None:
            return out["best_program"].detach()
        u_ref = u_tgt.float()
        x_ref = x_tgt.float()
        hu = min(out["program_controls"].shape[2], u_ref.shape[1])
        hx = min(out["program_rollouts"].shape[2], x_ref.shape[1])
        u_cost = (out["program_controls"][:, :, :hu, :] - u_ref[:, None, :hu, :]).abs().mean(dim=(2, 3))
        x_cost = (out["program_rollouts"][:, :, :hx, :] - x_ref[:, None, :hx, :]).abs().mean(dim=(2, 3))
        return (u_cost + lambda_xi * x_cost).argmin(dim=-1).detach()

    def ctrl_violation_loss(self, controls: torch.Tensor, degradation: torch.Tensor) -> torch.Tensor:
        d = degradation.float()
        steer_scale = d[:, 0].view(-1, 1, 1).clamp_min(0.05)
        brake_scale = d[:, 1].view(-1, 1, 1).clamp_min(0.05)
        throttle_scale = d[:, 2].view(-1, 1, 1).clamp_min(0.05)
        steer_v = F.relu(controls[..., 0].abs() - steer_scale)
        brake_v = F.relu(controls[..., 1].clamp_min(0.0) - brake_scale)
        throttle_v = F.relu(controls[..., 2].clamp_min(0.0) - throttle_scale)
        return (steer_v + brake_v + throttle_v).mean()

    def rollout_regularizer(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        controls = out["program_controls"]
        smooth_u = (controls[:, :, 1:, :] - controls[:, :, :-1, :]).abs().mean()
        start_loss = F.smooth_l1_loss(out["program_rollouts"][:, :, 0, :], batch["reset_state"].float().unsqueeze(1).expand_as(out["program_rollouts"][:, :, 0, :]))
        return start_loss + 0.05 * smooth_u

    # Legacy name.
    def dynamics_regularizer(self, out: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        return self.rollout_regularizer(out, batch)

    def monotonicity_penalty(self, batch: Dict[str, torch.Tensor], margins: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0, device=margins.device)

    def loss(self, batch: Dict[str, torch.Tensor], lambdas: Optional[Dict[str, float]] = None, enable_mono: bool = False) -> Dict[str, torch.Tensor]:
        lambdas = lambdas or {}
        out = self.forward(batch)
        r_star = batch.get("m_star", batch.get("r_star")).float()
        s_star = batch["s_star"].float()
        bsz = r_star.shape[0]
        match = self._matched_program(out, batch, lambda_xi=float(lambdas.get("lambda_xi", 0.25)))
        gather_m = match.view(bsz, 1, 1).expand(-1, 1, self.bottlenecks)
        matched_margins = torch.gather(out["certificate_margins"], 1, gather_m).squeeze(1)
        weight = 1.0 + lambdas.get("bd", 2.0) * torch.exp(-torch.abs(s_star) / max(lambdas.get("sigma_bd", 0.5), 1e-6))
        cert_loss = (weight * F.smooth_l1_loss(matched_margins, r_star, reduction="none").sum(dim=-1)).mean()
        # Every program receives a label pressure, but the best/matched program is
        # not the only branch with gradients.
        all_cert_loss = F.smooth_l1_loss(out["certificate_margins"], r_star.unsqueeze(1).expand_as(out["certificate_margins"]))
        active_loss = F.cross_entropy(out["active_logits"], batch["b_star"].long().clamp(0, self.bottlenecks - 1))
        u_tgt = batch.get("teacher_u")
        x_tgt = batch.get("teacher_traj")
        if u_tgt is not None and x_tgt is not None:
            gather_u = match.view(bsz, 1, 1, 1).expand(-1, 1, self.recovery_horizon, self.control_dim)
            gather_x = match.view(bsz, 1, 1, 1).expand(-1, 1, self.recovery_horizon + 1, self.state_dim)
            matched_u = torch.gather(out["program_controls"], 1, gather_u).squeeze(1)
            matched_traj = torch.gather(out["program_rollouts"], 1, gather_x).squeeze(1)
            u_ref = u_tgt.float()
            x_ref = x_tgt.float()
            hu = min(matched_u.shape[1], u_ref.shape[1])
            hx = min(matched_traj.shape[1], x_ref.shape[1])
            exec_loss = F.smooth_l1_loss(matched_u[:, :hu, :], u_ref[:, :hu, :]) + lambdas.get("lambda_xi", 0.25) * F.smooth_l1_loss(matched_traj[:, :hx, :], x_ref[:, :hx, :])
        else:
            exec_loss = torch.tensor(0.0, device=r_star.device)
        goal_trace = out["margin_traces"][..., 4]
        funnel_loss = F.relu(-out["funnel_values"]).mean() + F.relu(-goal_trace).mean()
        dyn_loss = self.rollout_regularizer(out, batch)
        ctrl_loss = self.ctrl_violation_loss(out["program_controls"], batch["degradation"])
        mono_loss = self.monotonicity_penalty(batch, matched_margins) if enable_mono else torch.tensor(0.0, device=r_star.device)
        total = (
            lambdas.get("certificate", 1.0) * (cert_loss + 0.25 * all_cert_loss)
            + lambdas.get("execution", lambdas.get("strat", 0.5)) * exec_loss
            + lambdas.get("funnel", 0.2) * funnel_loss
            + lambdas.get("act", 0.0) * active_loss
            + lambdas.get("dyn", 0.02) * dyn_loss
            + lambdas.get("ctrl", 0.05) * ctrl_loss
            + lambdas.get("mono", 0.0) * mono_loss
        )
        return {
            "loss": total,
            "execution_loss": exec_loss.detach(),
            "certificate_loss": cert_loss.detach(),
            "all_program_certificate_loss": all_cert_loss.detach(),
            "funnel_loss": funnel_loss.detach(),
            "active_loss": active_loss.detach(),
            "dyn_loss": dyn_loss.detach(),
            "ctrl_loss": ctrl_loss.detach(),
            "mono_loss": mono_loss.detach(),
            # Legacy aliases.
            "profile_loss": cert_loss.detach(),
            "strategy_loss": exec_loss.detach(),
        }


RPFN = RecoveryProgramFunnelNetwork
RecoveryProfileNetwork = RecoveryProgramFunnelNetwork


def ordering_loss(v_i: torch.Tensor, v_j: torch.Tensor, s_i: torch.Tensor, s_j: torch.Tensor, margin: float = 0.05) -> torch.Tensor:
    sign = torch.sign(s_i - s_j).clamp(min=-1, max=1)
    keep = sign.abs() > 0
    if keep.sum() == 0:
        return torch.tensor(0.0, device=v_i.device)
    return F.relu(margin - sign[keep] * (v_i[keep] - v_j[keep])).mean()
