from __future__ import annotations

import argparse
import runpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["cmrt", "rpfn", "finetune", "ordering", "calibration"])
    args, rest = parser.parse_known_args()
    module = {
        "cmrt": "mrvp.training.train_cmrt",
        "rpfn": "mrvp.training.train_rpfn",
        "finetune": "mrvp.training.finetune_rpfn",
        "ordering": "mrvp.training.train_ordering",
        "calibration": "mrvp.training.calibration",
    }[args.stage]
    import sys
    sys.argv = [module] + rest
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
