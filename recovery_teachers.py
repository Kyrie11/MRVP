#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrvp.carla.recovery_teachers import degraded_mpc_teacher, heuristic_post_impact_teacher


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MRVP recovery teacher from transition/mechanism JSON files.")
    parser.add_argument("--transition", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--teacher", choices=["degraded_mpc", "heuristic"], default="degraded_mpc")
    parser.add_argument("--horizon", type=float, default=5.0)
    args = parser.parse_args()
    trans = json.loads(Path(args.transition).read_text(encoding="utf-8"))
    labels = json.loads(Path(args.labels).read_text(encoding="utf-8"))
    fn = degraded_mpc_teacher if args.teacher == "degraded_mpc" else heuristic_post_impact_teacher
    out = fn(trans["x_plus"], labels["d_deg"], labels["h_ctx"], horizon=args.horizon)
    Path(args.output).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "s_star": out["s_star"], "b_star": out["b_star"]}))


if __name__ == "__main__":
    main()
