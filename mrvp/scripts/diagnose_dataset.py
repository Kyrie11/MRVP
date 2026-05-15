from __future__ import annotations

import argparse

from mrvp.sim.quality_diagnostics import diagnose_dataset


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--splits", default="train,val,calibration,test,shift")
    parser.add_argument("--output", required=True)
    parser.add_argument("--relaxed", action="store_true")
    args = parser.parse_args()
    diagnose_dataset(args.data, args.splits.split(","), args.output, strict=not args.relaxed)


if __name__ == "__main__":
    main()
