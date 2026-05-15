from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mrvp.data.dataset import iter_roots
from mrvp.sim.harm import construct_harm_comparable_set


def _pick_roots(data: str, split: str, root_id: str | None, max_cases: int) -> list[list[dict]]:
    roots = list(iter_roots(data, split))
    if root_id is not None:
        roots = [r for r in roots if r and str(r[0].get("root_id")) == root_id]
    return roots[:max_cases]


def _selector_summary(root: list[dict]) -> dict:
    safe = construct_harm_comparable_set(root)
    if not safe:
        safe = root
    severity = min(safe, key=lambda r: (float(r.get("rho_imp", 0.0)), -float(r.get("score_star", 0.0))))
    recovery = max(safe, key=lambda r: float(r.get("score_star", 0.0)))
    return {"safe": safe, "severity": severity, "recovery": recovery}


def _imshow(path: Path, arr: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 5))
    im = ax.imshow(arr, origin="lower")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _save_case(root: list[dict], out: Path, idx: int) -> None:
    rid = str(root[0]["root_id"])
    sel = _selector_summary(root)
    sev_action = str(sel["severity"]["action_id"])
    rec_action = str(sel["recovery"]["action_id"])
    safe_ids = {str(r["action_id"]) for r in sel["safe"]}

    fig, ax = plt.subplots(figsize=(8, 7))
    for row in root:
        action = str(row["action_id"])
        traj = np.asarray(row["prefix_rollout"], dtype=np.float32)
        reset = np.asarray(row["r_reset"], dtype=np.float32)
        flag = ""
        if action == sev_action and action == rec_action:
            flag = "*"
        elif action == sev_action:
            flag = "S"
        elif action == rec_action:
            flag = "R"
        label = f"{flag}{action} B{int(row['harm_bin'])} rho={float(row['rho_imp']):.2f} C={float(row['score_star']):.2f}"
        lw = 3.0 if flag else 1.2
        alpha = 1.0 if action in safe_ids else 0.35
        ax.plot(traj[:, 0], traj[:, 1], linewidth=lw, alpha=alpha, label=label)
        ax.scatter([reset[0]], [reset[1]], s=60 if flag else 20, alpha=alpha)
    ax.set_title(f"{rid}: candidate prefixes and reset states\nS={sev_action}, R={rec_action}; faded=outside harm-comparable set")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / f"case_{idx:03d}_{rid}_prefix_reset.png", dpi=180)
    plt.close(fig)

    actions = [str(r["action_id"]) for r in root]
    rho = np.array([float(r["rho_imp"]) for r in root])
    score = np.array([float(r["score_star"]) for r in root])
    bins = np.array([int(r["harm_bin"]) for r in root])
    x = np.arange(len(root))
    fig, ax1 = plt.subplots(figsize=(11, 4.5))
    ax1.bar(x - 0.18, rho, width=0.36, label="first-impact harm rho")
    ax1.set_ylabel("rho_imp")
    ax2 = ax1.twinx()
    ax2.bar(x + 0.18, score, width=0.36, alpha=0.55, label="teacher certificate C*")
    ax2.set_ylabel("score_star / C*")
    for i, (a, b) in enumerate(zip(actions, bins)):
        marker = ""
        if a == sev_action: marker += "S"
        if a == rec_action: marker += "R"
        ax1.text(i, max(float(rho.max()), 1e-6) * 1.04, f"B{b} {marker}", ha="center", fontsize=8)
    ax1.set_xticks(x)
    ax1.set_xticklabels(actions, rotation=35, ha="right")
    ax1.set_title(f"{rid}: severity-only vs recoverability evidence")
    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out / f"case_{idx:03d}_{rid}_severity_vs_recovery.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 7))
    for row in root:
        action = str(row["action_id"])
        traj = np.asarray(row["teacher_traj"], dtype=np.float32)
        flag = "R" if action == rec_action else ("S" if action == sev_action else "")
        ax.plot(traj[:, 0], traj[:, 1], linewidth=3.0 if flag else 1.2, label=f"{flag}{action} C={float(row['score_star']):.2f}")
    ax.set_title(f"{rid}: teacher recovery programs after reset")
    ax.set_xlabel("world x [m]")
    ax.set_ylabel("world y [m]")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.3)
    ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(out / f"case_{idx:03d}_{rid}_teacher_recovery.png", dpi=180)
    plt.close(fig)

    best = sel["recovery"]
    world = best["world_reset"]
    A = np.asarray(world["A"])
    O = np.asarray(world["O"])
    G = np.asarray(world["G"])
    Y = np.asarray(world["Y"])
    _imshow(out / f"case_{idx:03d}_{rid}_A_drivable_signed.png", A[0] + A[3], f"{rid}: recovery action {rec_action}, drivable + signed road")
    _imshow(out / f"case_{idx:03d}_{rid}_O_actor_occupancy.png", O[0, 0], f"{rid}: actor occupancy at reset")
    _imshow(out / f"case_{idx:03d}_{rid}_G_goal_fields.png", np.max(G, axis=0), f"{rid}: route/refuge/stop goal fields")
    _imshow(out / f"case_{idx:03d}_{rid}_Y_response.png", np.max(Y[:, 0], axis=0), f"{rid}: actor response / closing risk")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--split", default="all")
    p.add_argument("--output", required=True)
    p.add_argument("--root-id", default=None)
    p.add_argument("--max-cases", type=int, default=5)
    args = p.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    for i, root in enumerate(_pick_roots(args.data, args.split, args.root_id, args.max_cases), start=1):
        _save_case(root, out, i)


if __name__ == "__main__":
    main()
