from __future__ import annotations

import argparse
import json
from pathlib import Path

from .scenario_builder import CarlaMRVPGenerator
from .scenario_templates import generate_root_specs


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate CARLA-MRVP JSONL dataset.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--tm-port", type=int, default=8000)
    parser.add_argument("--output", required=True)
    parser.add_argument("--n-roots", type=int, default=100)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--towns", default="Town03,Town04,Town05,Town10HD")
    parser.add_argument("--fixed-delta-seconds", type=float, default=0.05)
    args = parser.parse_args()
    specs = generate_root_specs(args.n_roots, seed=args.seed, towns=[x.strip() for x in args.towns.split(",") if x.strip()])
    gen = CarlaMRVPGenerator(args.host, args.port, fixed_delta_seconds=args.fixed_delta_seconds, traffic_manager_port=args.tm_port)
    try:
        gen.generate(specs, args.output)
    finally:
        gen.close()
    print(json.dumps({"output": args.output, "roots": len(specs)}))


if __name__ == "__main__":
    main()
