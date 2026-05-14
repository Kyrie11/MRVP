from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mrvp.common.config import get_cfg, load_config
from mrvp.data.dataset import iter_roots
from mrvp.models.baselines import method_to_selector_mode, select_by_heuristic
from mrvp.models.cmrt import CMRT
from mrvp.models.rpfn import RPFN
from mrvp.models.selectors import SelectionConfig, infer_mrvp
from mrvp.training.utils import load_state_if_exists
from .common import action_metrics, reduce_metrics, write_table


def load_models(cmrt_path: str | None, rpfn_path: str | None, device: str):
    if cmrt_path and rpfn_path and Path(cmrt_path).exists() and Path(rpfn_path).exists():
        c_ckpt = torch.load(cmrt_path, map_location=device)
        r_ckpt = torch.load(rpfn_path, map_location=device)
        cmrt = CMRT(c_ckpt.get("cfg", {})).to(device)
        rpfn = RPFN(r_ckpt.get("cfg", {})).to(device)
        load_state_if_exists(cmrt, cmrt_path, device)
        load_state_if_exists(rpfn, rpfn_path, device)
        return cmrt, rpfn
    return None, None


def evaluate_methods(data: str, split: str, methods: list[str], cfg: dict, cmrt_path: str | None, rpfn_path: str | None, output: str) -> list[dict]:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmrt, rpfn = load_models(cmrt_path, rpfn_path, device)
    rows_out = []
    for method in methods:
        per_root = []
        for root_rows in iter_roots(data, split):
            if method.startswith("mrvp") and cmrt is not None and rpfn is not None:
                sel_cfg = SelectionConfig(
                    mode=method_to_selector_mode(method),
                    M=int(get_cfg(cfg, "selection.M", 16)),
                    beta=float(get_cfg(cfg, "selection.beta", 0.2)),
                    use_harm_filter=bool(get_cfg(cfg, "selection.use_harm_filter", True)),
                )
                result = infer_mrvp(root_rows, cmrt, rpfn, sel_cfg, device=device)
            else:
                result = select_by_heuristic(root_rows, method, use_harm_filter=bool(get_cfg(cfg, "selection.use_harm_filter", True)))
            selected = result["selected_action"]
            scores = result.get("scores", {selected: result["score"]})
            per_root.append(action_metrics(root_rows, selected, float(result["score"]), scores, float(get_cfg(cfg, "metrics.epsilon_c", 0.2))))
        agg = reduce_metrics(per_root)
        rows_out.append({"Method": method, **agg})
    write_table(rows_out, output, "main_results")
    return rows_out


def harm_spread(data: str, split: str, output: str) -> list[dict]:
    from collections import defaultdict
    from mrvp.sim.harm import construct_harm_comparable_set
    by_family = defaultdict(list)
    for root_rows in iter_roots(data, split):
        safe = construct_harm_comparable_set(root_rows)
        scores = [float(r["score_star"]) for r in safe]
        family = str(root_rows[0].get("scenario_family", "unknown"))
        by_family[family].append((len(safe), max(scores) - min(scores) if scores else 0.0))
    rows = []
    for family, vals in sorted(by_family.items()):
        rows.append({"Family": family, "Roots": len(vals), "|A_safe|": sum(v[0] for v in vals) / max(1, len(vals)), "Mean Delta C*": sum(v[1] for v in vals) / max(1, len(vals)), "% roots gap > eps": sum(v[1] > 0.2 for v in vals) / max(1, len(vals))})
    all_vals = [v for vals in by_family.values() for v in vals]
    rows.append({"Family": "All", "Roots": len(all_vals), "|A_safe|": sum(v[0] for v in all_vals) / max(1, len(all_vals)), "Mean Delta C*": sum(v[1] for v in all_vals) / max(1, len(all_vals)), "% roots gap > eps": sum(v[1] > 0.2 for v in all_vals) / max(1, len(all_vals))})
    write_table(rows, output, "harm_comparable_spread")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--analysis", default="main")
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cmrt", default=None)
    parser.add_argument("--rpfn", default=None)
    parser.add_argument("--methods", default="severity_only,mrvp_full")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.analysis == "harm_spread":
        harm_spread(args.data, args.split, args.output)
    else:
        evaluate_methods(args.data, args.split, args.methods.split(","), cfg, args.cmrt, args.rpfn, args.output)


if __name__ == "__main__":
    main()
