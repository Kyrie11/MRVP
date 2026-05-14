from __future__ import annotations

import argparse
import runpy


def main() -> None:
    import sys
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--output", required=True)
    args, rest = parser.parse_known_args()
    sys.argv = ["mrvp.evaluation.qualitative", "--data", args.data, "--split", args.split, "--output", args.output] + rest
    runpy.run_module("mrvp.evaluation.qualitative", run_name="__main__")


if __name__ == "__main__":
    main()
