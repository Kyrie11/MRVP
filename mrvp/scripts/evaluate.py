from __future__ import annotations

import argparse
import runpy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task", choices=["action", "reset", "program", "ablation", "tail", "shift", "runtime", "qualitative"])
    args, rest = parser.parse_known_args()
    module = {
        "action": "mrvp.evaluation.eval_action_selection",
        "reset": "mrvp.evaluation.eval_reset_prediction",
        "program": "mrvp.evaluation.eval_program_recovery",
        "ablation": "mrvp.evaluation.eval_ablation",
        "tail": "mrvp.evaluation.eval_tail_risk",
        "shift": "mrvp.evaluation.eval_shift",
        "runtime": "mrvp.evaluation.runtime",
        "qualitative": "mrvp.evaluation.qualitative",
    }[args.task]
    import sys
    sys.argv = [module] + rest
    runpy.run_module(module, run_name="__main__")


if __name__ == "__main__":
    main()
