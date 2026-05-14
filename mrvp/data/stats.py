from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np


def summarize_rows(rows: Iterable[dict]) -> dict[str, float | int]:
    data = list(rows)
    roots = {str(r["root_id"]) for r in data}
    contacts = [bool(r.get("contact", False)) for r in data]
    scores = np.asarray([float(r["score_star"]) for r in data], dtype=np.float32) if data else np.zeros(0)
    return {
        "roots": len(roots),
        "rows": len(data),
        "contact_rate": float(np.mean(contacts)) if contacts else 0.0,
        "score_mean": float(scores.mean()) if scores.size else 0.0,
        "score_min": float(scores.min()) if scores.size else 0.0,
        "score_max": float(scores.max()) if scores.size else 0.0,
    }


def group_by_root(rows: Iterable[dict]) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        out[str(row["root_id"])].append(row)
    return dict(out)
