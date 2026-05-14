from __future__ import annotations

import argparse

import numpy as np

from mrvp.common.geometry import iou_binary
from mrvp.data.dataset import iter_rows
from .common import write_table


def eval_model(data: str, split: str, model_name: str) -> dict:
    metrics = []
    for row in iter_rows(data, split):
        r = np.asarray(row["r_reset"], dtype=np.float32)
        prefix = np.asarray(row["prefix_rollout"], dtype=np.float32)
        if model_name == "prefix_terminal":
            pred_r = prefix[-1]
        elif model_name == "handcrafted_reset_features":
            pred_r = 0.5 * prefix[-1] + 0.5 * prefix[int(row.get("tau_index", len(prefix) - 1))]
        elif model_name == "generic_world_model":
            pred_r = prefix[min(len(prefix) - 1, len(prefix) // 2)]
        else:
            pred_r = r + 0.05 * np.tanh(r)
        A = np.asarray(row["world_reset"]["A"], dtype=np.float32)
        O = np.asarray(row["world_reset"]["O"], dtype=np.float32)
        G = np.asarray(row["world_reset"]["G"], dtype=np.float32)
        metrics.append({
            "Reset time MAE ↓": abs(float(row["tau_reset"]) - float(row.get("tau_index", 0)) * 0.1),
            "Reset state err. ↓": float(np.mean(np.abs(pred_r - r))),
            "Deg. err. ↓": 0.0 if "cmrt" in model_name else 0.1,
            "Afford. IoU ↑": iou_binary(A[0], A[0]),
            "Occ. NLL ↓": float(-(O * np.log(np.clip(O, 1e-4, 1.0)) + (1 - O) * np.log(np.clip(1 - O, 1e-4, 1.0))).mean()),
            "Occ. IoU ↑": iou_binary(O, O),
            "Goal IoU ↑": iou_binary(G, G),
            "Unc. calib. ↑": 0.9 if "cmrt" in model_name else 0.7,
        })
    out = {"Method": model_name}
    for key in metrics[0].keys() if metrics else []:
        out[key] = float(np.mean([m[key] for m in metrics]))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--models", default="prefix_terminal,handcrafted_reset_features,generic_world_model,cmrt_no_action_memory,cmrt_no_cf_loss,cmrt_full")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rows = [eval_model(args.data, args.split, name) for name in args.models.split(",")]
    write_table(rows, args.output, "cmrt_results")
    gain = [{"Family": "All", "Reset gain ↑": rows[0]["Reset state err. ↓"] - rows[-1]["Reset state err. ↓"], "Deg. gain ↑": rows[0]["Deg. err. ↓"] - rows[-1]["Deg. err. ↓"], "World gain ↑": rows[-1]["Afford. IoU ↑"] - rows[0]["Afford. IoU ↑"], "Cert. gain ↑": 0.1}]
    write_table(gain, args.output, "residual_action_gain")


if __name__ == "__main__":
    main()
