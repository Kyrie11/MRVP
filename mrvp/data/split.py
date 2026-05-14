from __future__ import annotations

import random
from collections import Counter


def parse_split_spec(spec: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for item in spec.split(","):
        name, value = item.split(":")
        out[name.strip()] = float(value)
    total = sum(out.values())
    if total <= 0:
        raise ValueError("split weights must sum to a positive value")
    return {k: v / total for k, v in out.items()}


def split_root_ids(root_ids, ratios: dict[str, float], seed: int) -> dict[str, list[str]]:
    ids = [str(r) for r in sorted(set(root_ids))]
    rng = random.Random(seed)
    rng.shuffle(ids)
    names = list(ratios.keys())
    counts = {name: int(len(ids) * ratios[name]) for name in names}
    remaining = len(ids) - sum(counts.values())
    for name in names[:remaining]:
        counts[name] += 1
    out: dict[str, list[str]] = {}
    start = 0
    for name in names:
        end = start + counts[name]
        out[name] = sorted(ids[start:end])
        start = end
    return out


def assert_no_leakage(splits: dict[str, list[str]]) -> None:
    all_ids = []
    for values in splits.values():
        all_ids.extend(str(v) for v in values)
    counts = Counter(all_ids)
    dup = [k for k, v in counts.items() if v > 1]
    if dup:
        raise ValueError(f"root-level leakage detected: {dup[:5]}")
