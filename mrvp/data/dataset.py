from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from .schema import SchemaDims, row_to_numpy


def iter_jsonl(paths: str | Path | Sequence[str | Path]) -> Iterable[Dict[str, Any]]:
    if isinstance(paths, (str, Path)):
        p = Path(paths)
        if p.is_dir():
            files = sorted(list(p.glob("*.jsonl")) + list(p.glob("*.json")))
        else:
            files = [p]
    else:
        files = [Path(x) for x in paths]
    for file in files:
        if file.suffix == ".json":
            obj = json.loads(file.read_text())
            if isinstance(obj, list):
                for row in obj:
                    yield row
            elif isinstance(obj, Mapping) and "rows" in obj:
                for row in obj["rows"]:
                    yield row
            else:
                yield obj
        else:
            with file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        yield json.loads(line)


class MRVPDataset(Dataset):
    """PyTorch dataset for rows following the MRVP appendix schema."""

    def __init__(
        self,
        path: str | Path | Sequence[str | Path],
        split: Optional[str] = None,
        dims: SchemaDims = SchemaDims(),
        keep_rows: bool = True,
    ) -> None:
        self.path = path
        self.split = split
        self.dims = dims
        rows = []
        raw_rows = []
        for row in iter_jsonl(path):
            if split is not None and str(row.get("split", "train")) != split:
                continue
            rows.append(row_to_numpy(row, dims))
            if keep_rows:
                raw_rows.append(row)
        if not rows:
            raise ValueError(f"No MRVP rows found in {path!r} for split={split!r}")
        self.rows = rows
        self.raw_rows = raw_rows if keep_rows else None
        self.root_to_indices: Dict[str, List[int]] = {}
        for i, row in enumerate(rows):
            self.root_to_indices.setdefault(str(row["root_id"]), []).append(i)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        item: Dict[str, Any] = {
            "idx": torch.tensor(idx, dtype=torch.long),
            "action_id": torch.tensor(row["action_id"], dtype=torch.long),
            "action_vec": torch.as_tensor(row["action_vec"], dtype=torch.float32),
            "o_hist": torch.as_tensor(row["o_hist"], dtype=torch.float32),
            "h_ctx": torch.as_tensor(row["h_ctx"], dtype=torch.float32),
            "rho_imp": torch.tensor(row["rho_imp"], dtype=torch.float32),
            "harm_bin": torch.tensor(row["harm_bin"], dtype=torch.long),
            "x_minus": torch.as_tensor(row["x_minus"], dtype=torch.float32),
            "x_plus": torch.as_tensor(row["x_plus"], dtype=torch.float32),
            "d_deg": torch.as_tensor(row["d_deg"], dtype=torch.float32),
            "z_mech": torch.as_tensor(row["z_mech"], dtype=torch.float32),
            "r_star": torch.as_tensor(row["r_star"], dtype=torch.float32),
            "b_star": torch.tensor(row["b_star"], dtype=torch.long),
            "s_star": torch.tensor(row["s_star"], dtype=torch.float32),
            "root_id": row["root_id"],
            "split": row["split"],
            "family": row["family"],
            "calib_group": row["calib_group"],
        }
        return item

    def raw(self, idx: int) -> Optional[Dict[str, Any]]:
        if self.raw_rows is None:
            return None
        return self.raw_rows[idx]


def mrvp_collate(batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    tensor_keys = [
        "idx",
        "action_id",
        "action_vec",
        "o_hist",
        "h_ctx",
        "rho_imp",
        "harm_bin",
        "x_minus",
        "x_plus",
        "d_deg",
        "z_mech",
        "r_star",
        "b_star",
        "s_star",
    ]
    for key in tensor_keys:
        out[key] = torch.stack([x[key] for x in batch], dim=0)
    for key in ["root_id", "split", "family", "calib_group"]:
        out[key] = [x[key] for x in batch]
    return out


def rows_by_root(dataset: MRVPDataset) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for row in dataset.rows:
        grouped.setdefault(str(row["root_id"]), []).append(row)
    return grouped


def root_split_summary(path: str | Path) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    roots: Dict[str, set] = {}
    for row in iter_jsonl(path):
        split = str(row.get("split", "train"))
        counts[split] = counts.get(split, 0) + 1
        roots.setdefault(split, set()).add(str(row.get("root_id", "0")))
    return {f"rows_{k}": v for k, v in counts.items()} | {f"roots_{k}": len(v) for k, v in roots.items()}
