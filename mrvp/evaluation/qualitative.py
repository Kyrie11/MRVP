from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from mrvp.data.dataset import iter_roots
from mrvp.models.baselines import select_by_heuristic
from .common import write_table


def save_case(root_rows: list[dict], output: Path, idx: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    sev = select_by_heuristic(root_rows, "severity_only")
    mrvp = select_by_heuristic(root_rows, "post_reset_scalar_risk")
    fig, ax = plt.subplots(figsize=(5, 5))
    for row in root_rows:
        traj = row["prefix_rollout"]
        ax.plot(traj[:, 0], traj[:, 1], label=str(row["action_id"]))
    ax.set_title(f"root {root_rows[0]['root_id']}")
    ax.set_aspect("equal", adjustable="box")
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / f"case_{idx:03d}_prefixes.png", dpi=150)
    plt.close(fig)
    return {"Case": f"Case {idx}", "Scenario": str(root_rows[0].get("scenario_family", "unknown")), "Severity-only choice": sev["selected_action"], "MRVP choice": mrvp["selected_action"], "Mechanism": "post-reset recoverability score"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--cases", default="boundary_limited_evasion,low_friction_side_conflict,actuator_degradation,occluded_secondary_hazard,narrow_corridor_recovery")
    parser.add_argument("--cmrt", default=None)
    parser.add_argument("--rpfn", default=None)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    out = Path(args.output)
    rows = []
    for idx, root_rows in enumerate(iter_roots(args.data, args.split)):
        if idx >= 5:
            break
        rows.append(save_case(root_rows, out, idx + 1))
    write_table(rows, out, "qualitative_cases")


if __name__ == "__main__":
    main()
