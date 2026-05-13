from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from mrvp.data.dataset import MRVPDataset, mrvp_collate
from mrvp.data.schema import BOTTLE_NECKS, group_dict_to_key
from mrvp.training.loops import to_device


def conformal_quantile(values: Sequence[float], delta: float) -> float:
    vals = np.asarray(list(values), dtype=np.float32)
    if vals.size == 0:
        return 0.0
    vals = np.sort(vals)
    # Split conformal finite-sample correction: ceil((n+1)*(1-delta)).
    k = int(np.ceil((vals.size + 1) * (1.0 - delta))) - 1
    k = int(np.clip(k, 0, vals.size - 1))
    return float(vals[k])


@torch.no_grad()
def predict_rpn(model: torch.nn.Module, dataset: MRVPDataset, batch_size: int = 256, device: str | torch.device = "cpu") -> np.ndarray:
    device = torch.device(device)
    model.to(device).eval()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0, collate_fn=mrvp_collate)
    preds = []
    for batch in tqdm(loader, desc="predict_rpn", leave=False):
        batch = to_device(batch, device)
        out = model(batch)
        preds.append(out["r_hat"].detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def _group_rows_by_root(rows: Sequence[Mapping[str, Any]]) -> Dict[str, List[int]]:
    by_root: Dict[str, List[int]] = {}
    for i, row in enumerate(rows):
        by_root.setdefault(str(row["root_id"]), []).append(i)
    return by_root


def fit_calibration_table(
    rows: Sequence[Mapping[str, Any]],
    r_hat: np.ndarray,
    delta_b: float | Sequence[float] = 0.02,
    n_min: int = 20,
) -> Dict[str, Any]:
    if isinstance(delta_b, (float, int)):
        deltas = [float(delta_b)] * len(BOTTLE_NECKS)
    else:
        deltas = list(map(float, delta_b))
    residuals: Dict[str, Dict[str, Dict[int, List[float]]]] = {level: {} for level in ["full", "medium", "coarse", "global"]}
    by_root = _group_rows_by_root(rows)
    for root, idxs in by_root.items():
        min_bin = min(int(rows[i]["harm_bin"]) for i in idxs)
        adm = [i for i in idxs if int(rows[i]["harm_bin"]) == min_bin]
        if not adm:
            continue
        # root-level worst optimistic residual for every bottleneck.
        residual_by_b = []
        for b in range(len(BOTTLE_NECKS)):
            vals = [max(0.0, float(r_hat[i, b]) - float(rows[i]["r_star"][b])) for i in adm]
            residual_by_b.append(max(vals) if vals else 0.0)
        group = rows[adm[0]].get("calib_group", {})
        for level in residuals.keys():
            key = group_dict_to_key(group, level)
            residuals[level].setdefault(key, {b: [] for b in range(len(BOTTLE_NECKS))})
            for b, val in enumerate(residual_by_b):
                residuals[level][key][b].append(val)
    quantiles: Dict[str, Dict[str, List[float]]] = {level: {} for level in residuals.keys()}
    counts: Dict[str, Dict[str, int]] = {level: {} for level in residuals.keys()}
    for level, groups in residuals.items():
        for key, by_b in groups.items():
            counts[level][key] = min(len(v) for v in by_b.values()) if by_b else 0
            quantiles[level][key] = [conformal_quantile(by_b[b], deltas[b]) for b in range(len(BOTTLE_NECKS))]
    return {
        "version": 1,
        "bottlenecks": BOTTLE_NECKS,
        "delta_b": deltas,
        "n_min": int(n_min),
        "quantiles": quantiles,
        "counts": counts,
        "fallback_order": ["full", "medium", "coarse", "global"],
    }


def save_calibration_table(table: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(table, indent=2), encoding="utf-8")


def load_calibration_table(path: str | Path | None) -> Dict[str, Any]:
    if path is None:
        return {"quantiles": {"global": {"global": [0.0] * len(BOTTLE_NECKS)}}, "counts": {"global": {"global": 999999}}, "n_min": 1, "fallback_order": ["global"]}
    return json.loads(Path(path).read_text(encoding="utf-8"))


def quantile_for_group(group: Mapping[str, Any], table: Mapping[str, Any]) -> np.ndarray:
    n_min = int(table.get("n_min", 1))
    for level in table.get("fallback_order", ["full", "medium", "coarse", "global"]):
        key = group_dict_to_key(group, level)
        q = table.get("quantiles", {}).get(level, {}).get(key)
        count = int(table.get("counts", {}).get(level, {}).get(key, 0))
        if q is not None and count >= n_min:
            return np.asarray(q, dtype=np.float32)
    q = table.get("quantiles", {}).get("global", {}).get("global", [0.0] * len(BOTTLE_NECKS))
    return np.asarray(q, dtype=np.float32)


def lower_bounds_for_rows(r_hat: np.ndarray, rows: Sequence[Mapping[str, Any]], table: Mapping[str, Any]) -> np.ndarray:
    lowers = np.zeros_like(r_hat, dtype=np.float32)
    for i, row in enumerate(rows):
        q = quantile_for_group(row.get("calib_group", {}), table)
        lowers[i] = r_hat[i] - q
    return lowers
