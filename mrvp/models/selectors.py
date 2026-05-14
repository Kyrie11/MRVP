from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from mrvp.data.collate import collate_rows
from mrvp.sim.harm import construct_harm_comparable_set


@dataclass
class SelectionConfig:
    mode: str = "lcvar"
    M: int = 16
    beta: float = 0.2
    use_harm_filter: bool = True


def lower_tail_cvar(values, beta: float) -> float:
    vals = sorted(float(v) for v in values)
    if not vals:
        return 0.0
    k = max(1, int(math.ceil(float(beta) * len(vals))))
    return float(np.mean(vals[:k]))


def aggregate_certificates(values, mode: str, beta: float) -> float:
    vals = [float(v) for v in values]
    if not vals:
        return 0.0
    if mode == "max":
        return float(np.max(vals))
    if mode == "mean":
        return float(np.mean(vals))
    if mode == "worst":
        return float(np.min(vals))
    if mode == "deterministic":
        return vals[0]
    return lower_tail_cvar(vals, beta)


def select_tail_program(cert_matrix: np.ndarray, beta: float) -> tuple[int, int]:
    best_by_sample = cert_matrix.max(axis=1)
    k = max(1, int(math.ceil(beta * len(best_by_sample))))
    tail_indices = np.argsort(best_by_sample)[:k]
    best = None
    for m in tail_indices:
        l = int(np.argmax(cert_matrix[m]))
        val = float(cert_matrix[m, l])
        if best is None or val > best[0]:
            best = (val, int(m), l)
    assert best is not None
    return best[1], best[2]


def infer_mrvp(root_rows: list[dict], cmrt, rpfn, cfg: SelectionConfig | dict | None = None, device: str = "cpu") -> dict[str, Any]:
    if isinstance(cfg, dict):
        cfg = SelectionConfig(**{k: v for k, v in cfg.items() if k in SelectionConfig.__annotations__})
    cfg = cfg or SelectionConfig()
    rows = construct_harm_comparable_set(root_rows) if cfg.use_harm_filter else list(root_rows)
    if not rows:
        raise ValueError("empty root rows")
    scores: dict[str, float] = {}
    certs: dict[str, list[float]] = {}
    programs: dict[str, np.ndarray] = {}
    cmrt.eval()
    rpfn.eval()
    with torch.no_grad():
        for row in rows:
            batch = collate_rows([row])
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            sample_count = max(1, int(cfg.M)) if cfg.mode != "deterministic" else 1
            vals = []
            mat = []
            for reset in cmrt.sample(batch, num_samples=sample_count):
                out = rpfn(reset)
                cert = out["cert_pred"].detach().cpu().numpy()[0]
                vals.append(float(cert.max()))
                mat.append(cert)
            score = aggregate_certificates(vals, cfg.mode, cfg.beta)
            scores[str(row["action_id"])] = score
            certs[str(row["action_id"])] = vals
            programs[str(row["action_id"])] = np.stack(mat, axis=0)
    selected = max(scores.items(), key=lambda kv: kv[1])[0]
    m_idx, l_idx = select_tail_program(programs[selected], cfg.beta)
    return {"selected_action": selected, "score": scores[selected], "scores": scores, "cert_samples": certs, "tail_sample": int(m_idx), "program_index": int(l_idx)}
