from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch

from mrvp.calibration import lower_bounds_for_rows, quantile_for_group
from mrvp.data.schema import BOTTLE_NECKS
from mrvp.models.common import empirical_cvar


def admissible_indices(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    if not rows:
        return []
    min_bin = min(int(r["harm_bin"]) for r in rows)
    return [i for i, r in enumerate(rows) if int(r["harm_bin"]) == min_bin]


def select_from_scores(rows: Sequence[Mapping[str, Any]], lower_bounds: np.ndarray, beta: float = 0.9) -> Dict[str, Any]:
    """Offline selector using one lower-bound profile per action."""
    adm = admissible_indices(rows)
    if not adm:
        raise ValueError("No candidate actions.")
    v = lower_bounds[adm].min(axis=1)
    losses = np.maximum(-v, 0.0)
    # With one score/action CVaR degenerates to violation depth; multi-sample path is below.
    best_local = int(np.argmin(losses))
    best_idx = adm[best_local]
    return {
        "selected_local_index": best_idx,
        "selected_action_id": int(rows[best_idx]["action_id"]),
        "tail_risk": float(losses[best_local]),
        "admissible_indices": adm,
        "lower_V": float(v[best_local]),
        "lower_bounds": lower_bounds[best_idx].tolist(),
    }


@torch.no_grad()
def select_action_with_models(
    root_batch: Dict[str, torch.Tensor],
    root_rows: Sequence[Mapping[str, Any]],
    msrt: torch.nn.Module,
    rpn: torch.nn.Module,
    calibration_table: Mapping[str, Any],
    num_samples: int = 32,
    beta: float = 0.9,
    device: str | torch.device = "cpu",
) -> Dict[str, Any]:
    """Algorithm 1: harm gate -> MSRT samples -> RPN -> group calibration -> CVaR."""
    device = torch.device(device)
    msrt.to(device).eval()
    rpn.to(device).eval()
    # move tensors.
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in root_batch.items()}
    adm = admissible_indices(root_rows)
    if not adm:
        raise ValueError("No candidate actions for root.")
    risks = []
    summaries = []
    for idx in adm:
        single = {}
        for k, v in batch.items():
            if torch.is_tensor(v):
                single[k] = v[idx : idx + 1]
            else:
                single[k] = [v[idx]] if isinstance(v, list) else v
        samples = msrt.sample(single, num_samples=num_samples, deterministic=False)
        # Repeat context/action-independent fields for RPN.
        rep = {
            "o_hist": single["o_hist"].repeat_interleave(num_samples, dim=0),
            "h_ctx": single["h_ctx"].repeat_interleave(num_samples, dim=0),
            "x_plus": samples["x_plus"],
            "d_deg": samples["d_deg"],
            "z_mech": samples["z_mech"],
        }
        pred = rpn(rep)["r_hat"]
        q = torch.as_tensor(quantile_for_group(root_rows[idx].get("calib_group", {}), calibration_table), device=device, dtype=pred.dtype)
        lower = pred - q[None, :]
        v_lower = lower.min(dim=-1).values
        losses = torch.clamp(-v_lower, min=0.0)
        risk = empirical_cvar(losses[None, :], beta=beta, dim=-1).squeeze(0)
        risks.append(risk)
        summaries.append({
            "candidate_index": idx,
            "action_id": int(root_rows[idx]["action_id"]),
            "risk": float(risk.detach().cpu()),
            "mean_lower_V": float(v_lower.mean().detach().cpu()),
            "p_violation": float((losses > 0).float().mean().detach().cpu()),
        })
    risk_tensor = torch.stack(risks)
    best_local = int(torch.argmin(risk_tensor).detach().cpu())
    best_idx = adm[best_local]
    return {
        "selected_local_index": best_idx,
        "selected_action_id": int(root_rows[best_idx]["action_id"]),
        "tail_risk": float(risk_tensor[best_local].detach().cpu()),
        "admissible_indices": adm,
        "candidate_summaries": summaries,
    }
