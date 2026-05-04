from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np

from .schema import (
    ACTION_NAMES,
    ACTION_VECTORS,
    BOTTLE_NECKS,
    CTX_DIM,
    DEG_DIM,
    HIST_LEN,
    MAX_AGENTS,
    ACTOR_FEAT_DIM,
    MECH_DIM,
    NUM_ACTIONS,
    STATE_DIM,
    SIDE_NAMES,
    harm_bin_from_rho,
)

FAMILIES = [
    "rear_end_blocked_forward_corridor",
    "side_swipe_near_boundary",
    "oblique_intersection_impact",
    "cut_in_unavoidable_contact",
    "boundary_critical_non_contact",
    "low_friction_recovery",
    "actuator_degradation_after_impact",
    "dense_agent_secondary_exposure",
]


def _split_for_root(i: int, n_roots: int) -> str:
    frac = i / max(1, n_roots)
    if frac < 0.70:
        return "train"
    if frac < 0.82:
        return "val"
    if frac < 0.90:
        return "cal"
    return "test"


def make_synthetic_rows(
    n_roots: int = 240,
    actions_per_root: int = NUM_ACTIONS,
    seed: int = 7,
    harm_thresholds: Sequence[float] = (0.5, 2.0, 4.0, 7.0, 11.0),
) -> List[Dict]:
    """Generate a synthetic MRVP dataset for smoke tests.

    This does not replace CARLA. It creates same-root counterfactuals with
    tied harm bins and different downstream bottlenecks so the whole pipeline
    can be executed without a simulator.
    """
    rng = np.random.default_rng(seed)
    rows: List[Dict] = []
    for r in range(n_roots):
        family_id = r % len(FAMILIES)
        family = FAMILIES[family_id]
        split = _split_for_root(r, n_roots)
        contact = family_id != 4
        base_speed = rng.uniform(5.0, 18.0)
        friction = {0: 0.95, 1: 0.75, 2: 0.35}.get(int(rng.choice([0, 1, 2], p=[0.65, 0.25, 0.10])), 0.8)
        if family_id == 5:
            friction = rng.uniform(0.25, 0.55)
        actor_density = rng.uniform(0.1, 1.0) if family_id in (0, 7, 2) else rng.uniform(0.0, 0.5)
        lane_width = rng.uniform(2.8, 4.2)
        boundary_side = int(rng.choice([-1, 1]))
        damage_class = int(rng.integers(0, 4)) if family_id in (0, 2, 6) else int(rng.integers(0, 2))
        town = f"Town{int(rng.integers(1, 8)):02d}"
        root_id = f"syn_{r:06d}"
        base_x = rng.normal(0, 4)
        base_y = rng.normal(0, 2)
        base_yaw = rng.normal(0, 0.15)
        route_dx = rng.uniform(20, 60)
        route_dy = rng.normal(0, 3)
        for a in range(actions_per_root):
            action_name = ACTION_NAMES[a % len(ACTION_NAMES)]
            action_vec = ACTION_VECTORS[a % len(ACTION_VECTORS)].astype(float)
            steer, throttle, brake, duration = action_vec
            steer_effect = float(steer) * rng.uniform(0.6, 1.4)
            brake_effect = float(brake) * rng.uniform(0.7, 1.3)
            # First-impact harm: same-root actions are intentionally close.
            harm_base = max(0.0, base_speed * (0.18 + 0.04 * family_id) + rng.normal(0, 0.25))
            harm_action = harm_base - 0.35 * brake_effect + 0.12 * abs(steer_effect) + rng.normal(0, 0.18)
            if not contact:
                harm_action = rng.uniform(0.0, 0.4)  # non-contact boundary-critical.
            rho_imp = float(max(0.0, harm_action))
            harm_bin = harm_bin_from_rho(rho_imp, harm_thresholds)
            # State before and after transition.
            x_minus = np.zeros(STATE_DIM, dtype=np.float32)
            x_minus[:10] = [
                base_x,
                base_y,
                base_yaw,
                base_speed * np.cos(base_yaw),
                base_speed * np.sin(base_yaw),
                base_speed,
                rng.normal(0, 0.05),
                rng.normal(0, 0.03),
                steer,
                0.0,
            ]
            yaw_reset = (0.15 + 0.05 * family_id) * steer_effect + (0.08 * damage_class) * rng.normal()
            speed_drop = (2.0 * brake_effect + 0.25 * rho_imp + rng.normal(0, 0.25)) if contact else 1.0 * brake_effect
            lat_shift = 0.4 * steer_effect + rng.normal(0, 0.15)
            x_plus = x_minus.copy()
            x_plus[0] += duration * base_speed * np.cos(base_yaw) * 0.65
            x_plus[1] += lat_shift
            x_plus[2] += yaw_reset
            x_plus[3] *= max(0.05, 1.0 - speed_drop / (base_speed + 1e-3))
            x_plus[4] += steer_effect * 0.8
            x_plus[5] = float(np.hypot(x_plus[3], x_plus[4]))
            x_plus[6] += yaw_reset / max(duration, 0.05)
            x_plus[7] += np.arctan2(x_plus[4], max(abs(x_plus[3]), 1e-3))
            x_plus[8] = steer
            x_plus[9] = duration
            # Degradation vector: steering/brake/throttle scale, delay, friction, damage norm.
            steering_scale = max(0.25, 1.0 - 0.12 * damage_class - rng.uniform(0, 0.08))
            brake_scale = max(0.25, 1.0 - 0.10 * damage_class - rng.uniform(0, 0.08))
            throttle_scale = max(0.25, 1.0 - 0.07 * damage_class)
            delay = rng.uniform(0.02, 0.18) + 0.03 * damage_class
            d_deg = np.asarray([steering_scale, brake_scale, throttle_scale, delay, friction, damage_class / 3.0], dtype=np.float32)
            # Mechanism descriptor.
            z = np.zeros(MECH_DIM, dtype=np.float32)
            z[0] = 1.0 if contact else 0.0
            side_id = int(np.clip(np.round((steer > 0) * 3 + (steer < 0) * 2), 0, 4))
            if abs(steer) < 0.2:
                side_id = int(rng.choice([0, 1, 4]))
            z[1 + side_id] = 1.0
            normal = np.asarray([-np.sign(steer) if abs(steer) > 0.1 else 1.0, -boundary_side], dtype=np.float32)
            normal = normal / (np.linalg.norm(normal) + 1e-6)
            z[6:8] = normal
            clearance = lane_width / 2.0 - abs(base_y + lat_shift) - 0.9
            z[8] = clearance if not contact else -0.1 * rho_imp + rng.normal(0, 0.05)
            z[9] = rng.normal(0.0, 0.4) + 0.2 * steer_effect
            z[10] = rho_imp if contact else 0.0
            reset = np.asarray([
                x_plus[3] - x_minus[3],
                x_plus[4] - x_minus[4],
                x_plus[5] - x_minus[5],
                x_plus[2] - x_minus[2],
                x_plus[6] - x_minus[6],
                x_plus[7] - x_minus[7],
                x_plus[1] - x_minus[1],
            ], dtype=np.float32)
            z[11:18] = reset
            return_length = max(0.0, route_dx - 3.0 * actor_density - 2.0 * abs(lat_shift))
            secondary_clear = max(0.0, 12.0 - 7.0 * actor_density - 0.4 * rho_imp + rng.normal(0, 1.0))
            z[18:24] = [lane_width, clearance, return_length, secondary_clear, np.sign(route_dy), 1.0]
            z[24:29] = [steering_scale, brake_scale, delay, friction, damage_class / 3.0]
            z[29:32] = [rng.uniform(0.02, 0.35), 0.05 + 0.15 * contact, actor_density]
            # Context vector.
            h_ctx = np.zeros(CTX_DIM, dtype=np.float32)
            h_ctx[:16] = [
                rng.normal(0, 0.03),
                -base_yaw,
                lane_width,
                lane_width / 2 + base_y,
                lane_width / 2 - base_y,
                secondary_clear,
                8.0,
                secondary_clear,
                max(0.1, secondary_clear / max(base_speed, 0.1)),
                return_length,
                actor_density,
                13.9,
                friction,
                rng.uniform(0, 1),
                rng.normal(0, 0.02),
                lane_width,
            ]
            h_ctx[16:24] = [0.0, -base_y, route_dx, route_dy, 0.0, actor_density * 0.2, 1.0, boundary_side]
            h_ctx[24:29] = [int(town[-2:]) / 10.0, family_id / max(1, len(FAMILIES)-1), 1.0 if contact else 0.0, damage_class / 3.0, 0.5]
            # History [T,A,F]. Ego plus fake surrounding actors.
            o_hist = np.zeros((HIST_LEN, MAX_AGENTS, ACTOR_FEAT_DIM), dtype=np.float32)
            for t in range(HIST_LEN):
                dt = (t - HIST_LEN + 1) * 0.1
                o_hist[t, 0, :] = [base_x + base_speed * dt, base_y, base_yaw, base_speed, 0.0, 4.6, 1.9, 1.0, 1.0]
                for ag in range(1, min(MAX_AGENTS, 1 + int(2 + actor_density * 8))):
                    o_hist[t, ag, :] = [
                        base_x + rng.uniform(6, 35) + base_speed * dt * rng.uniform(0.5, 1.1),
                        base_y + rng.normal(0, 3.2),
                        rng.normal(0, 0.2),
                        base_speed * rng.uniform(0.3, 1.2),
                        rng.normal(0, 0.5),
                        4.5,
                        1.8,
                        1.0,
                        1.0,
                    ]
            # Teacher margins: positive safe. Constructed from actual mechanism fields.
            r_sec = secondary_clear - 2.0 - 0.4 * actor_density - 0.2 * abs(yaw_reset) + rng.normal(0, 0.3)
            r_road = clearance + 0.7 - 0.2 * abs(steer_effect) + rng.normal(0, 0.15)
            r_stab = 1.6 * friction - abs(x_plus[6]) * 0.55 - abs(x_plus[7]) * 0.35 - 0.1 * base_speed / 10 + rng.normal(0, 0.15)
            r_ctrl = min(steering_scale, brake_scale) - 0.35 - 0.25 * abs(steer) - 0.2 * delay + rng.normal(0, 0.08)
            r_return = 0.08 * return_length + 0.2 * r_road + 0.15 * r_sec - 1.2 * actor_density + rng.normal(0, 0.25)
            if family_id == 0:
                r_sec -= 1.2 * actor_density
                r_return -= 0.8
            elif family_id == 1:
                r_road -= 0.7
            elif family_id == 2:
                r_stab -= 0.5 * abs(yaw_reset)
            elif family_id == 4:
                r_road -= 0.8
                r_return -= 0.4
            elif family_id == 5:
                r_stab -= 0.9
            elif family_id == 6:
                r_ctrl -= 0.7
            elif family_id == 7:
                r_sec -= 0.9
            r_star = np.asarray([r_sec, r_road, r_stab, r_ctrl, r_return], dtype=np.float32)
            row = {
                "root_id": root_id,
                "split": split,
                "family": family,
                "action_id": int(a),
                "action_name": action_name,
                "action_vec": action_vec.tolist(),
                "o_hist": o_hist.tolist(),
                "h_ctx": h_ctx.tolist(),
                "rho_imp": rho_imp,
                "harm_bin": int(harm_bin),
                "x_minus": x_minus.tolist(),
                "x_plus": x_plus.tolist(),
                "d_deg": d_deg.tolist(),
                "z_mech": z.tolist(),
                "r_star": r_star.tolist(),
                "b_star": int(np.argmin(r_star)),
                "s_star": float(np.min(r_star)),
                "calib_group": {
                    "contact_type": "contact" if contact else "boundary",
                    "contact_side": SIDE_NAMES[side_id],
                    "boundary_side": "right" if boundary_side > 0 else "left",
                    "friction_bin": "low" if friction < 0.55 else ("mid" if friction < 0.85 else "high"),
                    "damage_class": str(damage_class),
                    "density_bin": "dense" if actor_density > 0.65 else "normal",
                    "town": town,
                    "family": family,
                },
            }
            rows.append(row)
    return rows


def write_jsonl(rows: Sequence[Dict], output: str | Path) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic MRVP dataset for smoke tests.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-roots", type=int, default=240)
    parser.add_argument("--actions-per-root", type=int, default=NUM_ACTIONS)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    rows = make_synthetic_rows(args.n_roots, args.actions_per_root, args.seed)
    write_jsonl(rows, args.output)
    splits: Dict[str, int] = {}
    for row in rows:
        splits[row["split"]] = splits.get(row["split"], 0) + 1
    print(json.dumps({"output": args.output, "rows": len(rows), "splits": splits}, indent=2))


if __name__ == "__main__":
    main()
