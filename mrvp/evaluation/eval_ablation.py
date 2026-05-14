from __future__ import annotations

import argparse

from mrvp.common.config import load_config
from .eval_action_selection import evaluate_methods
from .common import write_table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    variants = cfg.get("variants", [])
    method_map = {
        "mrvp_full": "mrvp_full",
        "mean_selection": "mrvp_mean",
        "worst_sample_selection": "mrvp_worst",
    }
    rows = []
    for variant in variants:
        method = method_map.get(variant, "cmrt_direct_certificate" if "program" in variant or "funnel" in variant else "post_reset_scalar_risk")
        result = evaluate_methods(args.data, args.split, [method], {"selection": {"M": 4, "beta": 0.2}}, None, None, args.output)
        row = dict(result[0])
        row["Variant"] = variant
        row["Removed component"] = variant.replace("no_", "").replace("_", " ") if variant != "mrvp_full" else "none"
        rows.append(row)
    write_table(rows, args.output, "ablation_results")


if __name__ == "__main__":
    main()
