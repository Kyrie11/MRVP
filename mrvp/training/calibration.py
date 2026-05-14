from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from mrvp.common.metrics import calibration_error
from mrvp.common.serialization import write_json
from mrvp.data.collate import collate_rows
from mrvp.data.dataset import iter_rows
from mrvp.models.cmrt import CMRT
from mrvp.models.rpfn import RPFN
from .utils import load_state_if_exists, move_batch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="calibration")
    parser.add_argument("--cmrt", required=True)
    parser.add_argument("--rpfn", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmrt_ckpt = torch.load(args.cmrt, map_location=device)
    rpfn_ckpt = torch.load(args.rpfn, map_location=device)
    cmrt = CMRT(cmrt_ckpt.get("cfg", {})).to(device)
    rpfn = RPFN(rpfn_ckpt.get("cfg", {})).to(device)
    load_state_if_exists(cmrt, args.cmrt, device)
    load_state_if_exists(rpfn, args.rpfn, device)
    cmrt.eval(); rpfn.eval()
    preds = []
    truth = []
    with torch.no_grad():
        for row in iter_rows(args.data, args.split):
            batch = collate_rows([row])
            batch = move_batch(batch, device)
            reset = cmrt.sample(batch, num_samples=1)[0]
            out = rpfn(reset)
            preds.append(float(out["cert_pred"].max().cpu()))
            truth.append(float(batch["score_star"].cpu()[0]))
    p = np.asarray(preds, dtype=np.float32)
    y = np.asarray(truth, dtype=np.float32)
    threshold = float(np.quantile(p - y, 0.1)) if len(p) else 0.0
    result = {"threshold": threshold, "ece": calibration_error(p, y), "coverage": float(np.mean((p - threshold) <= y)) if len(p) else 0.0, "frr": float(np.mean((p > 0) & (y < 0))) if len(p) else 0.0}
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "calibration.json", result)


if __name__ == "__main__":
    main()
