from __future__ import annotations

import argparse

from mrvp.common.config import load_config
from mrvp.sim.dataset_builder import build_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-roots", type=int, default=10000)
    parser.add_argument("--families", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--append", action="store_true",
                        help="Preserve existing root shards and add new roots after the last index.")
    parser.add_argument(
"--target-total",
            action = "store_true",
            help = "When used with --append, interpret --num-roots as the desired total root count instead of roots to add.",
    )

    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["sim_source"] = "metadrive"
    build_synthetic_dataset(
        args.output,
        args.num_roots,
        args.families.split(","),
        args.seed,
        "metadrive",
        cfg,
        append = args.append,
        target_total = args.target_total,
    )

if __name__ == "__main__":
    main()
