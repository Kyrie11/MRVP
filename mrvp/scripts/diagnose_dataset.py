from __future__ import annotations

import argparse

from mrvp.sim.quality_diagnostics import diagnose_dataset
from mrvp.common.config import load_config
from mrvp.data.schema import ACTION_IDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", default="train,val,calibration,test,shift")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default=None, help="Dataset config used for expected tensor shapes and horizons.")
    parser.add_argument("--expected-actions", default=",".join(ACTION_IDS))
    parser.add_argument("--epsilon-c", type=float, default=0.2)
    parser.add_argument("--relaxed", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config) if args.config else None
    expected_actions = [x for x in args.expected_actions.split(",") if x]
    diagnose_dataset(
        args.data, [x for x in args.splits.split(",") if x], args.output,
        epsilon_c=args.epsilon_c, strict=not args.relaxed, cfg=cfg,
        expected_actions=expected_actions,
    )


if __name__ == "__main__":
    main()
