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
from mrvp.data.schema import SchemaDims, audit_vector_from_row, ensure_tokens, validate_row_no_leakage


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate MRVP JSONL schema for token/world leakage.")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--fail-on-leakage", action="store_true")
    args = ap.parse_args()

    dims = SchemaDims()
    rows = 0
    missing_event_tokens = 0
    audit_token_fallback_rows = 0
    world_contains_target_rows = 0
    leakage_rows = 0
    warnings_by_type: Dict[str, int] = {}

    for row in iter_jsonl(args.data):
        rows += 1
        token_source = row.get("event_tokens", row.get("tokens", row.get("z_tokens", None)))
        if token_source is None:
            missing_event_tokens += 1
        else:
            tok = ensure_tokens(token_source, dims.token_count, dims.token_dim).reshape(-1)
            audit = audit_vector_from_row(row, dims)
            n = min(tok.size, audit.size)
            if n and np.allclose(tok[:n], audit[:n], atol=1e-6, rtol=1e-6):
                audit_token_fallback_rows += 1
        warns = validate_row_no_leakage(row, dims)
        if warns:
            leakage_rows += 1
            for w in warns:
                warnings_by_type[w] = warnings_by_type.get(w, 0) + 1
                if w.startswith("world_plus_contains"):
                    world_contains_target_rows += 1
                    break

    report: Dict[str, Any] = {
        "rows": rows,
        "leakage_rows": leakage_rows,
        "missing_event_tokens": missing_event_tokens,
        "audit_token_fallback_rows": audit_token_fallback_rows,
        "world_contains_target_rows": world_contains_target_rows,
        "warnings_by_type": warnings_by_type,
        "fatal": bool(audit_token_fallback_rows > 0 or world_contains_target_rows > 0),
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
