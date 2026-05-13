from __future__ import annotations

from typing import Tuple

import torch

from .margins import collision_margin, control_margin, goal_margin, road_margin, stability_margin


def compute_program_certificate(
    states: torch.Tensor,
    controls: torch.Tensor,
    funnel_values: torch.Tensor,
    recovery_world: torch.Tensor,
    degradation: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute explicit margin profile and scalar program certificate.

    Returns ``certificate_profile`` [B,L,5], ``program_certificate`` [B,L] and
    ``margin_traces`` [B,L,H+1,5].  Larger values are safer.
    """
    sec = collision_margin(states, recovery_world)
    road = road_margin(states, recovery_world)
    stab = stability_margin(states, degradation)
    ctrl = control_margin(controls, degradation)
    # Pad control trace from H to H+1 by repeating the final control margin.
    ctrl = torch.cat([ctrl, ctrl[..., -1:].clone()], dim=-1)
    goal_final = goal_margin(states[..., -1, :], target, recovery_world)
    goal_trace = goal_final.unsqueeze(-1).expand_as(sec)
    margin_traces = torch.stack([sec, road, stab, ctrl, goal_trace], dim=-1)
    certificate_profile = margin_traces.min(dim=-2).values
    funnel_min = funnel_values.min(dim=-1).values
    certificate_profile = torch.minimum(certificate_profile, funnel_min.unsqueeze(-1).expand_as(certificate_profile))
    program_certificate = certificate_profile.min(dim=-1).values
    return certificate_profile, program_certificate, margin_traces
