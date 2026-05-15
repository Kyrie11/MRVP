from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mrvp.data.dataset import iter_roots
from mrvp.models.baselines import select_by_heuristic


def _pick_roots(data: str, split: str, root_id: str | None, case_index: int | None, max_cases: int) -> list[list[dict]]:
    picked: list[list[dict]] = []
    for idx, root in enumerate(iter_roots(data, split)):
        rid = str(root[0]["root_id"])
        if root_id is not None and rid != root_id:
            continue
        if case_index is not None and idx != case_index:
            continue
        picked.append(root)
        if len(picked) >= max_cases:
            break
    if not picked:
        raise ValueError("no matching root found")
    return picked


def _save_root(root: list[dict], out: Path, case_no: int) -> None:
    rid = str(root[0]["root_id"])
    sev = select_by_heuristic(root, "severity_only")
    rec = select_by_heuristic(root, "post_reset_scalar_risk")

    fig, ax = plt.subplots(figsize=(7, 7))
    for row in root:
        traj = np.asarray(row["prefix_rollout"])
        action = str(row["action_id"])
        label = f"{action} bin={int(row['harm_bin'])} C={float(row['score_star']):.2f}"
        ax.plot(traj[:, 0], traj[:, 1], label=label)
        rr = np.asarray(row["r_reset"])
        ax.scatter([rr[0]], [rr[1]], s=12)
    ax.set_title(f"{rid}: prefixes and reset states\nseverity={sev['selected_action']} recoverability={rec['selected_action']}")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(out / f"case_{case_no:03d}_{rid}_prefixes.png", dpi=180)
    plt.close(fig)

    best_row = max(root, key=lambda r: float(r["score_star"]))
    world = best_row["world_reset"]
    panels = [("A", "affordance"), ("G", "goal"), ("O", "occupancy_t0"), ("Y", "actor_response_t0")]
    for key, title in panels:
        arr = np.asarray(world[key])
        if arr.ndim == 4:  # time, channel, height, width
            arr = arr[0, 0]
        elif arr.ndim == 3:  # channel, height, width
            arr = arr[0]
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(arr, origin="lower")
        ax.set_title(f"{rid}: {title} ({best_row['action_id']})")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out / f"case_{case_no:03d}_{rid}_{key}.png", dpi=180)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 7))
    for row in root:
        traj = np.asarray(row["teacher_traj"])
        ax.plot(traj[:, 0], traj[:, 1], label=f"{row['action_id']} C={float(row['score_star']):.2f}")
    ax.set_title(f"{rid}: teacher recovery trajectories")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=6, loc="best")
    fig.tight_layout()
    fig.savefig(out / f"case_{case_no:03d}_{rid}_teacher_recovery.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    parser.add_argument("--root-id", default=None)
    parser.add_argument("--case-index", type=int, default=None, help="Zero-based root index inside the split.")
    parser.add_argument("--max-cases", type=int, default=1)
    args = parser.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    roots = _pick_roots(args.data, args.split, args.root_id, args.case_index, args.max_cases)
    for i, root in enumerate(roots, start=1):
        _save_root(root, out, i)


if __name__ == "__main__":
    main()
