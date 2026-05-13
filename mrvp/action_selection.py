from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch


def harm_comparable_indices(rows: Sequence[Mapping[str, Any]]) -> List[int]:
    if not rows:
        return []
    min_bin = min(int(r.get("harm_bin", 0)) for r in rows)
    return [i for i, r in enumerate(rows) if int(r.get("harm_bin", 0)) == min_bin]


# Legacy name retained for offline evaluation helpers.
admissible_indices = harm_comparable_indices


def lower_tail_cvar(values: torch.Tensor, beta: float, dim: int = -1) -> torch.Tensor:
    """Lower-tail CVaR for larger-is-better certificates."""
    if values.shape[dim] <= 0:
        raise ValueError("lower_tail_cvar requires at least one value")
    k = max(1, int(math.ceil(float(beta) * values.shape[dim])))
    sorted_v = values.sort(dim=dim).values
    return sorted_v.narrow(dim, 0, k).mean(dim=dim)


def _slice_single(batch: Mapping[str, Any], idx: int) -> Dict[str, Any]:
    single: Dict[str, Any] = {}
    for k, v in batch.items():
        if torch.is_tensor(v):
            single[k] = v[idx : idx + 1]
        elif isinstance(v, list):
            single[k] = [v[idx]]
        else:
            single[k] = v
    return single


def _repeat_context(single: Mapping[str, Any], samples: Mapping[str, torch.Tensor], num_samples: int) -> Dict[str, Any]:
    rep: Dict[str, Any] = {
        "reset_state": samples["reset_state"],
        "reset_slots": samples["reset_slots"],
        "recovery_world_vec": samples["recovery_world_vec"],
        "degradation": samples["degradation"],
        "reset_time": samples["reset_time"],
        # Legacy aliases for compatible modules.
        "x_plus": samples["reset_state"],
        "event_tokens": samples["reset_slots"],
        "world_plus": samples["recovery_world_vec"],
        "deg": samples["degradation"],
        "d_deg": samples["degradation"],
    }
    for key in ["o_hist", "h_ctx", "x_t", "action_id", "action_vec"]:
        if key in single and torch.is_tensor(single[key]):
            rep[key] = single[key].repeat_interleave(num_samples, dim=0)
    return rep


def select_from_scores(rows: Sequence[Mapping[str, Any]], lower_bounds: np.ndarray, beta: float = 0.2) -> Dict[str, Any]:
    """Offline selector that maximizes single-sample certificate score."""
    adm = harm_comparable_indices(rows)
    if not adm:
        raise ValueError("No candidate actions.")
    v = lower_bounds[adm].min(axis=1)
    best_local = int(np.argmax(v))
    best_idx = adm[best_local]
    return {
        "selected_local_index": best_idx,
        "selected_action_id": int(rows[best_idx]["action_id"]),
        "score": float(v[best_local]),
        "score_lcvar": float(v[best_local]),
        "admissible_indices": adm,
        "lower_bounds": lower_bounds[best_idx].tolist(),
    }


@torch.no_grad()
def select_tail_consistent_action(
    root_batch: Dict[str, Any],
    root_rows: Sequence[Mapping[str, Any]],
    cmrt: torch.nn.Module,
    rpfn: torch.nn.Module,
    num_samples: int = 32,
    beta: float = 0.2,
    device: str | torch.device = "cpu",
    certificate_correction: Optional[Callable[[torch.Tensor, Mapping[str, Any]], torch.Tensor] | torch.Tensor | float] = None,
) -> Dict[str, Any]:
    """MRVP action selection: CMRT samples -> RPFN certificates -> LCVaR max."""
    device = torch.device(device)
    cmrt.to(device).eval()
    rpfn.to(device).eval()
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in root_batch.items()}
    adm = harm_comparable_indices(root_rows)
    if not adm:
        raise ValueError("No candidate actions for root.")
    scores = []
    summaries = []
    cache: Dict[int, Dict[str, torch.Tensor]] = {}
    for idx in adm:
        single = _slice_single(batch, idx)
        if hasattr(cmrt, "sample_reset_problems"):
            samples = cmrt.sample_reset_problems(single, num_samples=num_samples, deterministic=False)
        else:
            samples = cmrt.sample(single, num_samples=num_samples, deterministic=False)
            samples = {
                "reset_state": samples.get("reset_state", samples["x_plus"]),
                "reset_slots": samples.get("reset_slots", samples.get("event_tokens")),
                "recovery_world_vec": samples.get("recovery_world_vec", samples.get("world_plus")),
                "degradation": samples.get("degradation", samples.get("deg", samples.get("d_deg"))),
                "reset_time": samples.get("reset_time", samples.get("event_time")),
                **samples,
            }
        rep = _repeat_context(single, samples, num_samples)
        pred = rpfn(rep)
        certs = pred["program_certificates"]
        if certificate_correction is not None:
            if callable(certificate_correction):
                corr = certificate_correction(certs, root_rows[idx])
                certs = certs - corr
            else:
                certs = certs - torch.as_tensor(certificate_correction, device=device, dtype=certs.dtype)
        best_per_sample = certs.max(dim=1).values
        score = lower_tail_cvar(best_per_sample, beta=beta, dim=0)
        scores.append(score)
        cache[idx] = {"certs": certs.detach(), "best_per_sample": best_per_sample.detach()}
        summaries.append(
            {
                "candidate_index": int(idx),
                "action_id": int(root_rows[idx]["action_id"]),
                "score_lcvar": float(score.detach().cpu()),
                "mean_certificate": float(best_per_sample.mean().detach().cpu()),
                "p_negative_certificate": float((best_per_sample < 0).float().mean().detach().cpu()),
                "best_certificate_mean": float(certs.max(dim=1).values.mean().detach().cpu()),
            }
        )
    score_tensor = torch.stack(scores)
    best_local = int(torch.argmax(score_tensor).detach().cpu())
    best_idx = adm[best_local]
    selected_cache = cache[best_idx]
    best_values = selected_cache["best_per_sample"]
    k = max(1, int(math.ceil(float(beta) * best_values.numel())))
    tail_idx = torch.argsort(best_values)[:k]
    tail_certs = selected_cache["certs"][tail_idx]
    flat_best = int(torch.argmax(tail_certs.reshape(-1)).detach().cpu())
    tail_local = flat_best // tail_certs.shape[1]
    program_index = flat_best % tail_certs.shape[1]
    tail_sample_index = int(tail_idx[tail_local].detach().cpu())
    return {
        "selected_action_id": int(root_rows[best_idx]["action_id"]),
        "selected_local_index": int(best_idx),
        "score": float(score_tensor[best_local].detach().cpu()),
        "score_lcvar": float(score_tensor[best_local].detach().cpu()),
        "admissible_indices": [int(i) for i in adm],
        "tail_sample_index": tail_sample_index,
        "program_index": int(program_index),
        "candidate_summaries": summaries,
    }


# Legacy wrapper.  The fourth positional argument used to be a calibration table;
# it is accepted but ignored unless passed as certificate_correction by keyword.
def select_action_with_models(
    root_batch: Dict[str, Any],
    root_rows: Sequence[Mapping[str, Any]],
    msrt: torch.nn.Module,
    rpn: torch.nn.Module,
    calibration_table: Optional[Mapping[str, Any]] = None,
    num_samples: int = 32,
    beta: float = 0.2,
    device: str | torch.device = "cpu",
    certificate_correction: Optional[Callable[[torch.Tensor, Mapping[str, Any]], torch.Tensor] | torch.Tensor | float] = None,
) -> Dict[str, Any]:
    return select_tail_consistent_action(
        root_batch,
        root_rows,
        msrt,
        rpn,
        num_samples=num_samples,
        beta=beta,
        device=device,
        certificate_correction=certificate_correction,
    )
