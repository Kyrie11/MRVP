from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mrvp.common.serialization import write_json
from mrvp.sim.harm import construct_harm_comparable_set


def ensure_out(path: str | Path) -> Path:
    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def write_table(rows: list[dict[str, Any]], output: str | Path, name: str) -> None:
    out = ensure_out(output)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{name}.csv", index=False)
    write_json(out / f"{name}.json", rows)
    with (out / f"{name}.tex").open("w", encoding="utf-8") as f:
        f.write(df.to_latex(index=False, escape=False))


def selected_row(root_rows: list[dict], action_id: str) -> dict:
    for row in root_rows:
        if str(row["action_id"]) == str(action_id):
            return row
    raise KeyError(f"action not found: {action_id}")


def pair_accuracy(root_rows: list[dict], pred_scores: dict[str, float], epsilon_c: float = 0.2) -> tuple[int, int]:
    rows = construct_harm_comparable_set(root_rows)
    ok = 0
    total = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            ti = float(rows[i]["score_star"])
            tj = float(rows[j]["score_star"])
            if abs(ti - tj) >= epsilon_c:
                pi = float(pred_scores.get(str(rows[i]["action_id"]), 0.0))
                pj = float(pred_scores.get(str(rows[j]["action_id"]), 0.0))
                ok += int(np.sign(pi - pj) == np.sign(ti - tj))
                total += 1
    return ok, total


def action_metrics(root_rows: list[dict], action_id: str, pred_score: float, pred_scores: dict[str, float], epsilon_c: float = 0.2) -> dict[str, float]:
    safe = construct_harm_comparable_set(root_rows)
    row = selected_row(root_rows, action_id)
    best_teacher = max(float(r["score_star"]) for r in safe)
    exec_score = float(row["score_star"])
    cert = np.asarray(row["cert_star"], dtype=np.float32)
    ok, total = pair_accuracy(root_rows, pred_scores, epsilon_c)
    return {
        "pair_ok": ok,
        "pair_total": total,
        "regret": best_teacher - exec_score,
        "frr_num": float(pred_score > 0 and exec_score < 0),
        "frr_den": float(pred_score > 0),
        "tail_cert": float(pred_score),
        "violation_depth": float(max(0.0, -exec_score)),
        "recovery_success": float(exec_score > 0),
        "secondary_collision": float(cert[1] < 0),
        "road_departure": float(cert[0] < 0),
        "stability_violation": float(cert[3] < 0),
        "route_refuge_return": float(cert[4] > 0),
    }


def reduce_metrics(items: list[dict[str, float]]) -> dict[str, float]:
    if not items:
        return {}
    pair_ok = sum(x["pair_ok"] for x in items)
    pair_total = sum(x["pair_total"] for x in items)
    frr_num = sum(x["frr_num"] for x in items)
    frr_den = sum(x["frr_den"] for x in items)
    out = {
        "Pair Acc. ↑": float(pair_ok / max(1, pair_total)),
        "Regret ↓": float(np.mean([x["regret"] for x in items])),
        "FRR ↓": float(frr_num / max(1, frr_den)),
        "Tail Cert. ↑": float(np.mean([x["tail_cert"] for x in items])),
        "Viol. Depth ↓": float(np.mean([x["violation_depth"] for x in items])),
        "Recovery Succ. ↑": float(np.mean([x["recovery_success"] for x in items])),
        "Sec. Coll. ↓": float(np.mean([x["secondary_collision"] for x in items])),
        "Road Depart. ↓": float(np.mean([x["road_departure"] for x in items])),
        "Stab. Viol. ↓": float(np.mean([x["stability_violation"] for x in items])),
        "Route/Refuge ↑": float(np.mean([x["route_refuge_return"] for x in items])),
    }
    return out
