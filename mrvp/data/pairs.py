from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from .dataset import MRVPDataset


class SameRootPairDataset(Dataset):
    """Pairs mined only within same root and same harm bin, as in Appendix."""

    def __init__(self, base: MRVPDataset, eps_s: float = 0.25, max_pairs_per_root: int = 64) -> None:
        self.base = base
        self.pairs: List[Tuple[int, int]] = []
        for _, indices in base.root_to_indices.items():
            by_bin = {}
            for i in indices:
                hb = int(base.rows[i]["harm_bin"])
                by_bin.setdefault(hb, []).append(i)
            for group in by_bin.values():
                if len(group) < 2:
                    continue
                candidates = []
                for p, i in enumerate(group):
                    for j in group[p + 1 :]:
                        if abs(float(base.rows[i]["s_star"]) - float(base.rows[j]["s_star"])) >= eps_s:
                            candidates.append((i, j))
                if len(candidates) > max_pairs_per_root:
                    # Deterministic sub-sampling to keep loaders manageable.
                    step = max(1, len(candidates) // max_pairs_per_root)
                    candidates = candidates[::step][:max_pairs_per_root]
                self.pairs.extend(candidates)
        if not self.pairs:
            raise ValueError("No same-root severity-equivalent pairs found. Lower eps_s or generate more roots.")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        i, j = self.pairs[idx]
        return self.base[i], self.base[j]


def pair_collate(batch):
    from .dataset import mrvp_collate

    left = [b[0] for b in batch]
    right = [b[1] for b in batch]
    return {"i": mrvp_collate(left), "j": mrvp_collate(right)}
