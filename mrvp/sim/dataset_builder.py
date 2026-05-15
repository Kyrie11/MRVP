from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

from mrvp.common.config import load_config
from mrvp.common.serialization import load_root_rows, save_root_rows, write_json
from mrvp.data.schema import ACTION_IDS, CERT_NAMES, SCENARIO_FAMILIES
from mrvp.data.split import assert_no_leakage, parse_split_spec, split_root_ids
from .action_library import default_action_library
from .degradation import sample_degradation
from .harm import HarmBinner, compute_rho
from .recovery_world import make_recovery_world
from .reset_targets import extract_reset
from .teacher_mpc import rollout_bicycle, solve_teacher


def _ctx(size: int, friction: float) -> dict:
    yy, xx = np.mgrid[0:size, 0:size]
    center = size / 2
    drivable = (np.abs(yy - center) < size * 0.35).astype(np.float32)
    route = np.exp(-((yy - center) ** 2) / (2 * (size * 0.06) ** 2)).astype(np.float32)
    return {"drivable": drivable, "route": route, "friction": float(friction)}


def _root_rows(root_id: str, family: str, seed: int, sim_source: str, cfg: dict) -> list[dict]:
    rng = np.random.default_rng(seed)
    size = int(cfg.get("world", {}).get("bev_size", 64))
    hist_steps = int(cfg.get("history_steps", 5))
    actors = int(cfg.get("actors", 6))
    actor_features = int(cfg.get("actor_features", 8))
    dt = float(cfg.get("prefix_dt", 0.1))
    x_t = np.zeros(12, dtype=np.float32)
    x_t[3] = rng.uniform(10.0, 22.0)
    x_t[1] = rng.normal(0, 0.4)
    x_t[2] = rng.normal(0, 0.04)
    deg = sample_degradation(rng, family)
    h_ctx = _ctx(size, float(deg[0]))
    o_hist = rng.normal(0, 0.3, size=(hist_steps, actors, actor_features)).astype(np.float32)
    o_hist[:, 0, 0] = np.linspace(-hist_steps * x_t[3] * dt, 0, hist_steps)
    boundary_dir = -1.0 if rng.random() < 0.5 else 1.0
    corridor_dir = 1.0 if family in {"NCR", "BLE"} else -boundary_dir
    lib = default_action_library(duration=float(cfg.get("prefix_horizon_s", 1.0)), dt=dt, boundary_direction=boundary_dir, corridor_direction=corridor_dir)
    rows = []
    family_risk = {"SC": 0.45, "LHB": 0.40, "CI": 0.43, "BLE": 0.38, "LF": 0.35, "AD": 0.33, "OSH": 0.42, "NCR": 0.36}[family]
    for action_index, action in enumerate(lib):
        prefix_rollout = rollout_bicycle(x_t, action.controls, dt=dt)
        steer_energy = float(np.mean(np.abs(action.controls[:, 0])))
        brake_energy = float(np.mean(action.controls[:, 1]))
        action_bias = {"hard_brake": -0.08, "maintain": 0.15, "brake_left": 0.05, "brake_right": 0.05, "mild_left": 0.02, "mild_right": 0.02, "boundary_away": -0.03, "corridor_seeking": -0.04}[action.action_id]
        contact_prob = np.clip(family_risk + action_bias - 0.12 * brake_energy + 0.03 * steer_energy, 0.08, 0.85)
        contact = bool(rng.random() < contact_prob)
        contact_time = float(rng.uniform(0.25, 0.85)) if contact else -1.0
        impact_speed = float(max(0.0, x_t[3] * (0.4 + 0.3 * rng.random()) - 4.0 * brake_energy)) if contact else 0.0
        impulse = float(1200.0 * impact_speed * rng.uniform(0.15, 0.35)) if contact else 0.0
        delta_v = float(impact_speed * rng.uniform(0.18, 0.45)) if contact else 0.0
        rho = compute_rho(impact_speed, impulse, delta_v, cfg.get("harm", {}).get("type", "delta_v"))
        idx, tau, r_reset = extract_reset(prefix_rollout, contact, contact_time if contact else 0.0, dt, cfg.get("reset", {}))
        world = make_recovery_world(size, rng, family, r_reset, horizon=int(cfg.get("world", {}).get("steps_O", 15)))
        teacher = solve_teacher(r_reset, world, deg, horizon=int(cfg.get("recovery_horizon", 30)), dt=float(cfg.get("recovery_dt", 0.1)), seed=seed + action_index)
        cert = np.array([teacher.margins[name] for name in CERT_NAMES], dtype=np.float32)
        score_adjust = 0.08 * (action.action_id == "corridor_seeking") + 0.06 * (action.action_id == "boundary_away") - 0.04 * (action.action_id == "maintain")
        score_star = float(teacher.score_star + score_adjust + rng.normal(0, 0.03))
        row = {
            "root_id": root_id,
            "action_id": action.action_id,
            "sim_source": sim_source,
            "scenario_family": family,
            "seed": int(seed),
            "trigger_time": 0.0,
            "o_hist": o_hist,
            "h_ctx": h_ctx,
            "x_t": x_t,
            "action_params": action.params,
            "prefix_rollout": prefix_rollout.astype(np.float32),
            "prefix_controls": action.controls.astype(np.float32),
            "contact": contact,
            "contact_time": contact_time,
            "contact_actor_id": int(rng.integers(1, actors)) if contact else -1,
            "contact_normal": np.array([rng.choice([-1.0, 1.0]), rng.normal(0, 0.2)], dtype=np.float32),
            "impulse": impulse,
            "delta_v": delta_v,
            "rho_imp": rho,
            "harm_bin": 0,
            "tau_reset": tau,
            "tau_index": idx,
            "r_reset": r_reset.astype(np.float32),
            "deg": deg.astype(np.float32),
            "world_reset": world,
            "teacher_u": teacher.controls.astype(np.float32),
            "teacher_traj": teacher.trajectory.astype(np.float32),
            "cert_star": cert,
            "score_star": score_star,
            "audit": {"target_type": teacher.target_type, "route_progress": float(r_reset[0]), "teacher_success": bool(teacher.success), "impulse_estimated": True},
        }
        rows.append(row)
    return rows


def _existing_root_indices(all_dir: Path, sim_source: str) -> list[int]:
    pattern = re.compile(rf"^root_{re.escape(sim_source)}_(\d{{6}})\.h5$")
    indices: list[int] = []
    for path in all_dir.glob(f"root_{sim_source}_*.h5"):
        match = pattern.match(path.name)
        if match:
            indices.append(int(match.group(1)))
    return sorted(indices)


def build_synthetic_dataset(
    output: str | Path,
    num_roots: int,
    families: Iterable[str],
    seed: int,
    sim_source: str,
    cfg: dict,
    *,
    append: bool = False,
    target_total: bool = False,
) -> Path:
    """Build root shards.

    Default behavior is unchanged: an existing output directory is removed.
    With append=True, existing shards are preserved and new root ids continue
    from the largest existing index. If target_total=True, num_roots is
    interpreted as the desired total number of roots after the run rather
    than the number of new roots to add.
    """
    out = Path(output)
    all_dir = out / "all"
    if out.exists() and not append:
        shutil.rmtree(out)
    all_dir.mkdir(parents=True, exist_ok=True)

    existing = _existing_root_indices(all_dir, sim_source) if append else []
    start_idx = (max(existing) + 1) if existing else 0
    roots_to_write = max(0, int(num_roots) - len(existing)) if target_total else int(num_roots)

    fams = list(families)
    if not fams:
        fams = SCENARIO_FAMILIES
    for offset in range(roots_to_write):
        i = start_idx + offset
        family = fams[i % len(fams)]
        rows = _root_rows(f"{sim_source}_{i:06d}", family, int(seed) + i, sim_source, cfg)
        save_root_rows(all_dir / f"root_{sim_source}_{i:06d}.h5", rows)
    final_indices = _existing_root_indices(all_dir, sim_source)
    meta = {
        "sim_source": sim_source,
        "num_roots": len(final_indices),
        "num_roots_written_this_run": roots_to_write,
        "append": bool(append),
        "target_total": bool(target_total),
        "families": fams,
        "schema": "MRVP-CF",
        "action_ids": ACTION_IDS,
        "seed": int(seed),
    }
    write_json(out / "meta.json", meta)
    return out


def _read_all_roots(inputs: list[str]) -> dict[str, list[dict]]:
    roots: dict[str, list[dict]] = {}
    for item in inputs:
        base = Path(item)
        files = sorted((base / "all").glob("root_*.h5")) or sorted(base.glob("**/root_*.h5"))
        for f in files:
            rows = load_root_rows(f)
            if rows:
                roots[str(rows[0]["root_id"])] = rows
    return roots


def merge_datasets(inputs: list[str], output: str | Path, split_spec: str, seed: int, num_contact_bins: int = 5) -> Path:
    roots = _read_all_roots(inputs)
    if not roots:
        raise FileNotFoundError("no input root shards found")
    splits = split_root_ids(list(roots.keys()), parse_split_spec(split_spec), seed)
    assert_no_leakage(splits)
    train_rows = [row for rid in splits.get("train", []) for row in roots[rid]]
    binner = HarmBinner(num_contact_bins=num_contact_bins).fit(
        np.array([float(r["rho_imp"]) for r in train_rows], dtype=np.float32),
        np.array([bool(r.get("contact", False)) for r in train_rows], dtype=bool),
    )
    out = Path(output)
    if out.exists():
        shutil.rmtree(out)
    for split, ids in splits.items():
        split_dir = out / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for rid in ids:
            rows = roots[rid]
            for row in rows:
                row["harm_bin"] = binner.assign(float(row["rho_imp"]), bool(row.get("contact", False)))
            save_root_rows(split_dir / f"root_{rid}.h5", rows)
    write_json(out / "splits.json", splits)
    write_json(out / "meta.json", {"schema": "MRVP-CF", "harm_binner": binner.to_dict(), "inputs": inputs})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    syn = sub.add_parser("synthetic")
    syn.add_argument("--config", default=None)
    syn.add_argument("--output", required=True)
    syn.add_argument("--num-roots", type=int, default=32)
    syn.add_argument("--families", default=",".join(SCENARIO_FAMILIES))
    syn.add_argument("--seed", type=int, default=42)
    syn.add_argument("--sim-source", default="synthetic")
    merge = sub.add_parser("merge")
    merge.add_argument("--inputs", required=True)
    merge.add_argument("--output", required=True)
    merge.add_argument("--split", default="train:0.70,val:0.10,calibration:0.10,test:0.10")
    merge.add_argument("--root-level", action="store_true")
    merge.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()
    if args.cmd == "synthetic":
        cfg = load_config(args.config)
        build_synthetic_dataset(args.output, args.num_roots, args.families.split(","), args.seed, args.sim_source, cfg)
    if args.cmd == "merge":
        merge_datasets(args.inputs.split(","), args.output, args.split, args.seed)


if __name__ == "__main__":
    main()
