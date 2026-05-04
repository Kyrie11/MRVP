#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from mrvp.carla.transition_extractor import extract_transition


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract MRVP transition target from a saved rollout JSON.")
    parser.add_argument("--rollout", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    rollout = json.loads(Path(args.rollout).read_text(encoding="utf-8"))
    trans = extract_transition(rollout)
    Path(args.output).write_text(json.dumps(trans, indent=2), encoding="utf-8")
    print(json.dumps({"output": args.output, "transition_type": trans["transition_type"], "harm_bin": trans["harm_bin"]}))


if __name__ == "__main__":
    main()
