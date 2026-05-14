from __future__ import annotations

import argparse

from mrvp.common.config import load_config
from mrvp.sim.dataset_builder import build_synthetic_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num-roots", type=int, default=2000)
    parser.add_argument("--families", required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    cfg = load_config(args.config)
    cfg["host"] = args.host
    cfg["port"] = args.port
    cfg["sim_source"] = "carla"
    build_synthetic_dataset(args.output, args.num_roots, args.families.split(","), args.seed, "carla", cfg)


if __name__ == "__main__":
    main()
