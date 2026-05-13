from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mrvp.data.dataset import iter_jsonl
from mrvp.data.schema import SchemaDims, audit_vector_from_row, ensure_slots, validate_row_no_leakage


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate MRVP reset schema for reset-slot/recovery-world label leakage.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--fail-on-leakage", action="store_true")
    args = ap.parse_args()

    dims = SchemaDims()
    rows = 0
    missing_reset_slots = 0
    legacy_event_tokens_rows = 0
    audit_token_fallback_rows = 0
    recovery_world_contains_target_rows = 0
    leakage_rows = 0
    warnings_by_type: Dict[str, int] = {}

    for row in iter_jsonl(args.data):
        rows += 1
        reset_slot_source = row.get("reset_slots", row.get("reset_slots_target"))
        if reset_slot_source is None:
            missing_reset_slots += 1
        legacy_token_source = row.get("event_tokens", row.get("tokens", row.get("z_tokens", None)))
        if legacy_token_source is not None:
            legacy_event_tokens_rows += 1
            tok = ensure_slots(legacy_token_source, dims.reset_slot_count, dims.reset_slot_dim).reshape(-1)
            audit = audit_vector_from_row(row, dims)
            n = min(tok.size, audit.size)
            if n and np.allclose(tok[:n], audit[:n], atol=1e-6, rtol=1e-6):
                audit_token_fallback_rows += 1
        warns = validate_row_no_leakage(row, dims)
        if warns:
            leakage_rows += 1
            for w in warns:
                warnings_by_type[w] = warnings_by_type.get(w, 0) + 1
                if w.startswith("recovery_world_contains"):
                    recovery_world_contains_target_rows += 1
                    break

    report: Dict[str, Any] = {
        "rows": rows,
        "leakage_rows": leakage_rows,
        "missing_reset_slots": missing_reset_slots,
        "legacy_event_tokens_rows": legacy_event_tokens_rows,
        "audit_token_fallback_rows": audit_token_fallback_rows,
        "recovery_world_contains_target_rows": recovery_world_contains_target_rows,
        "warnings_by_type": warnings_by_type,
        "fatal": bool(audit_token_fallback_rows > 0 or recovery_world_contains_target_rows > 0),
    }
    text = json.dumps(report, indent=2, ensure_ascii=False)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    if args.fail_on_leakage and report["fatal"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
