from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path
from typing import Iterable

import numpy as np

from mrvp.common.config import load_config
from mrvp.common.serialization import load_root_rows, save_root_rows, write_json
from mrvp.data.schema import ACTION_IDS, CERT_NAMES, SCENARIO_FAMILIES
from mrvp.sim.action_library import default_action_library
from mrvp.sim.degradation import sample_degradation
from mrvp.sim.harm import HarmBinner, compute_rho
from mrvp.sim.metadrive_cf_adapter import MetaDriveCounterfactualAdapter
from mrvp.sim.reset_targets import extract_reset
from mrvp.sim.teacher_mpc import solve_teacher


def _existing_indices(all_dir: Path) -> list[int]:
    pat = re.compile(r"^root_metadrive_(\d{6})\.h5$")
    out: list[int] = []
    for p in all_dir.glob("root_metadrive_*.h5"):
        m = pat.match(p.name)
        if m:
            out.append(int(m.group(1)))
    return sorted(out)


def _directions_from_root(root_state: np.ndarray, h_ctx: dict) -> tuple[float, float]:
    left = float(h_ctx.get("dist_to_left_side", 3.5))
    right = float(h_ctx.get("dist_to_right_side", 3.5))
    boundary_dir = -1.0 if left < right else 1.0
    corridor_dir = 1.0 if right < left else -1.0
    return boundary_dir, corridor_dir


def _fit_and_write_harm_bins(files: list[Path], cfg: dict) -> dict:
    rows = []
    for f in files:
        rows.extend(load_root_rows(f))
    harm_cfg = cfg.get("harm", {}) or {}
    binner = HarmBinner(
        harm_type=str(harm_cfg.get("type", "delta_v")),
        num_contact_bins=int(harm_cfg.get("num_contact_bins", 5)),
        no_contact_bin=int(harm_cfg.get("no_contact_bin", 0)),
    ).fit(
        np.array([float(r["rho_imp"]) for r in rows], dtype=np.float32),
        np.array([bool(r.get("contact", False)) for r in rows], dtype=bool),
    )
    for f in files:
        root_rows = load_root_rows(f)
        for r in root_rows:
            r["harm_bin"] = binner.assign(float(r["rho_imp"]), bool(r.get("contact", False)))
        save_root_rows(f, root_rows)
    return binner.to_dict()


def build_metadrive_cf_dataset(
    output: str | Path,
    num_roots: int,
    families: Iterable[str],
    seed: int,
    cfg: dict,
    *,
    append: bool = False,
    target_total: bool = False,
    max_attempts_per_root: int = 6,
) -> Path:
    out = Path(output)
    all_dir = out / "all"
    if out.exists() and not append:
        shutil.rmtree(out)
    all_dir.mkdir(parents=True, exist_ok=True)

    existing = _existing_indices(all_dir) if append else []
    start_idx = (max(existing) + 1) if existing else 0
    roots_to_write = max(0, int(num_roots) - len(existing)) if target_total else int(num_roots)
    fams = list(families) or SCENARIO_FAMILIES
    adapter = MetaDriveCounterfactualAdapter(cfg)
    written = 0
    skipped: list[dict] = []
    try:
        i = start_idx
        while written < roots_to_write:
            family = fams[i % len(fams)]
            root_id = f"metadrive_{i:06d}"
            root_seed = int(seed) + i
            success = False
            last_error = None
            for attempt in range(max_attempts_per_root):
                attempt_seed = root_seed + attempt * 1000003
                rng = np.random.default_rng(attempt_seed)
                deg = sample_degradation(rng, family)
                try:
                    root = adapter.build_root(root_id, attempt_seed, family, deg)
                    if root is None:
                        last_error = "root rollout terminated before trigger"
                        continue
                    boundary_dir, corridor_dir = _directions_from_root(root.root_state, root.h_ctx)
                    lib = default_action_library(
                        duration=float(cfg.get("prefix_horizon_s", 1.0)),
                        dt=float(cfg.get("prefix_dt", 0.1)),
                        boundary_direction=boundary_dir,
                        corridor_direction=corridor_dir,
                    )
                    rows = []
                    for action_index, action in enumerate(lib):
                        pref = adapter.apply_prefix(root, action.controls)
                        idx, tau, r_reset = extract_reset(
                            pref.prefix_rollout,
                            pref.contact,
                            pref.contact_time if pref.contact else 0.0,
                            float(cfg.get("prefix_dt", 0.1)),
                            cfg.get("reset", {}),
                        )
                        size = int(cfg.get("world", {}).get("bev_size", 64))
                        occ_steps = int(cfg.get("world", {}).get("steps_O", 15))
                        world = adapter.make_world_from_reset(r_reset, deg, size=size, horizon=occ_steps)
                        teacher = solve_teacher(
                            r_reset,
                            world,
                            deg,
                            horizon=int(cfg.get("recovery_horizon", 30)),
                            dt=float(cfg.get("recovery_dt", 0.1)),
                            seed=attempt_seed + action_index,
                        )
                        rho = compute_rho(
                            impact_speed=float(max(0.0, pref.delta_v)),
                            impulse=float(pref.impulse),
                            delta_v=float(pref.delta_v),
                            harm_type=cfg.get("harm", {}).get("type", "delta_v"),
                        )
                        cert = np.array([teacher.margins[name] for name in CERT_NAMES], dtype=np.float32)
                        audit = dict(pref.audit)
                        audit.update({
                            "target_type": teacher.target_type,
                            "teacher_success": bool(teacher.success),
                            "route_progress": float(r_reset[0]),
                            "root_seed": int(attempt_seed),
                            "trigger_step": int(root.trigger_step),
                            "env_config": root.env_config,
                            "metadrive_replayed_root": True,
                        })
                        rows.append({
                            "root_id": root.root_id,
                            "action_id": action.action_id,
                            "sim_source": "metadrive",
                            "scenario_family": family,
                            "seed": int(attempt_seed),
                            "trigger_time": float(root.trigger_step * float(cfg.get("prefix_dt", 0.1))),
                            "o_hist": root.o_hist,
                            "h_ctx": root.h_ctx,
                            "x_t": root.root_state,
                            "action_params": action.params,
                            "prefix_rollout": pref.prefix_rollout,
                            "prefix_controls": pref.prefix_controls,
                            "contact": bool(pref.contact),
                            "contact_time": float(pref.contact_time),
                            "contact_actor_id": int(pref.contact_actor_id),
                            "contact_normal": pref.contact_normal,
                            "impulse": float(pref.impulse),
                            "delta_v": float(pref.delta_v),
                            "rho_imp": float(rho),
                            "harm_bin": 0,
                            "tau_reset": float(tau),
                            "tau_index": int(idx),
                            "r_reset": r_reset.astype(np.float32),
                            "deg": deg.astype(np.float32),
                            "world_reset": world,
                            "teacher_u": teacher.controls.astype(np.float32),
                            "teacher_traj": teacher.trajectory.astype(np.float32),
                            "cert_star": cert,
                            "score_star": float(teacher.score_star),
                            "audit": audit,
                        })
                    save_root_rows(all_dir / f"root_{root_id}.h5", rows)
                    written += 1
                    success = True
                    break
                except Exception as exc:
                    last_error = repr(exc)
                    try:
                        adapter.close()
                    except Exception:
                        pass
            if not success:
                skipped.append({"root_id": root_id, "family": family, "seed": root_seed, "error": str(last_error)})
            i += 1
    finally:
        adapter.close()

    all_files = sorted(all_dir.glob("root_metadrive_*.h5"))
    harm_meta = _fit_and_write_harm_bins(all_files, cfg) if all_files else {}
    write_json(out / "meta.json", {
        "schema": "MRVP-CF",
        "sim_source": "metadrive",
        "counterfactual_mode": "seed_replay_branching",
        "num_roots": len(all_files),
        "num_roots_written_this_run": written,
        "append": bool(append),
        "target_total": bool(target_total),
        "families": fams,
        "action_ids": ACTION_IDS,
        "seed": int(seed),
        "harm_binner": harm_meta,
        "skipped": skipped,
    })
    return out


def main() -> None:
    p = argparse.ArgumentParser(description="Build a real MetaDrive-backed MRVP counterfactual dataset.")
    p.add_argument("--config", default="configs/dataset_metadrive.yaml")
    p.add_argument("--output", required=True)
    p.add_argument("--num-roots", type=int, default=100)
    p.add_argument("--families", default=",".join(SCENARIO_FAMILIES))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--append", action="store_true")
    p.add_argument("--target-total", action="store_true")
    p.add_argument("--max-attempts-per-root", type=int, default=6)
    args = p.parse_args()
    cfg = load_config(args.config)
    build_metadrive_cf_dataset(
        output=args.output,
        num_roots=args.num_roots,
        families=[x.strip() for x in args.families.split(",") if x.strip()],
        seed=args.seed,
        cfg=cfg,
        append=args.append,
        target_total=args.target_total,
        max_attempts_per_root=args.max_attempts_per_root,
    )


if __name__ == "__main__":
    main()
