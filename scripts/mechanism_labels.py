#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json
from pathlib import Path

from mrvp.carla.mechanism_labels import compute_mechanism_labels
from mrvp.carla.transition_extractor import extract_transition


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute MRVP mechanism labels z_mech/h_ctx/d_deg from rollout JSON.")
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--transition", default="")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rollout = json.loads(Path(args.rollout).read_text(encoding="utf-8"))
    transition = json.loads(Path(args.transition).read_text(encoding="utf-8")) if args.transition else extract_transition(rollout)
    labels = compute_mechanism_labels(rollout, transition)
    Path(args.output).write_text(json.dumps(labels, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "group": labels["calib_group"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
