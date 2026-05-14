from __future__ import annotations

import argparse

from mrvp.data.dataset import iter_roots
from mrvp.models.baselines import select_by_heuristic
from .common import action_metrics, reduce_metrics, write_table


def shift_name(root_rows: list[dict]) -> str:
    fam = str(root_rows[0].get("scenario_family", "unknown"))
    mapping = {"LF": "Low friction", "AD": "Actuator degradation", "SC": "Unseen contact side", "CI": "High actor density", "BLE": "Unseen map geometry", "OSH": "Occluded secondary hazard", "NCR": "Unseen map geometry"}
    return mapping.get(fam, "Mixed")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="shift")
    parser.add_argument("--shifts", default="low_friction,actuator_degradation,unseen_contact_side,high_actor_density,unseen_map_geometry,occluded_secondary_hazard")
    parser.add_argument("--methods", default="severity_only,mrvp_full")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    try:
        roots = list(iter_roots(args.data, args.split))
    except FileNotFoundError:
        roots = list(iter_roots(args.data, "test"))
    rows = []
    for shift in sorted(set(shift_name(r) for r in roots)):
        subset = [r for r in roots if shift_name(r) == shift]
        for method in args.methods.split(","):
            per_root = []
            heuristic = "post_reset_scalar_risk" if method == "mrvp_full" else method
            for root_rows in subset:
                result = select_by_heuristic(root_rows, heuristic)
                per_root.append(action_metrics(root_rows, result["selected_action"], result["score"], result.get("scores", {result["selected_action"]: result["score"]})))
            rows.append({"Shift": shift, "Method": method, **reduce_metrics(per_root)})
    write_table(rows, args.output, "shift_results")


if __name__ == "__main__":
    main()
