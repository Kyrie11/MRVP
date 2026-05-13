from __future__ import annotations

import argparse
import json
from pathlib import Path


def contains(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8") if path.exists() else False


def main() -> None:
    ap = argparse.ArgumentParser(description="Check code alignment with MRVP CMRT/RPFN main method.")
    ap.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--fail-on-error", action="store_true")
    args = ap.parse_args()
    root = Path(args.root)
    checks = {
        "cmrt_class": contains(root / "mrvp/models/cmrt.py", "class CounterfactualMotionResetTokenizer"),
        "rpfn_class": contains(root / "mrvp/models/rpfn.py", "class RecoveryProgramFunnelNetwork"),
        "selection_lower_tail_cvar": contains(root / "mrvp/action_selection.py", "lower_tail_cvar") and contains(root / "mrvp/action_selection.py", "torch.argmax"),
        "rpfn_rollout_main_path": contains(root / "mrvp/models/rpfn.py", "degraded_bicycle_step") and contains(root / "mrvp/models/rpfn.py", "program_rollouts"),
        "no_direct_strategy_traj_head": not contains(root / "mrvp/models/rpfn.py", "strategy_traj_head"),
        "cmrt_audit_event_default_off": contains(root / "scripts/train.py", "--lambda-audit-event") and contains(root / "scripts/train.py", "default=0.0"),
    }
    report = {"checks": checks, "passed": all(checks.values())}
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.fail_on_error and not report["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
