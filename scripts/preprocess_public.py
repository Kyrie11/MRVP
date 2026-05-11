#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path as _Path
sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))

import argparse
import json

from mrvp.public_pretraining.preprocessors import PREPROCESSORS


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocess public driving datasets into MRVP context-pretraining JSONL.")
    parser.add_argument("--dataset", required=True, choices=sorted(PREPROCESSORS.keys()))
    parser.add_argument("--input", required=True, help="Dataset root directory.")
    parser.add_argument("--output", required=True, help="Output JSONL.")
    parser.add_argument("--max-records", type=int, default=None)
    parser.add_argument("--version", default="v1.0-mini", help="nuScenes version only.")
    args = parser.parse_args()
    fn = PREPROCESSORS[args.dataset]
    if args.dataset == "nuscenes":
        n = fn(args.input, args.output, version=args.version, max_records=args.max_records)
    else:
        n = fn(args.input, args.output, max_records=args.max_records)
    print(json.dumps({"dataset": args.dataset, "input": args.input, "output": args.output, "records": n}, indent=2))


if __name__ == "__main__":
    main()
