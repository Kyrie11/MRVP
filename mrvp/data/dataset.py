from __future__ import annotations

from pathlib import Path
from typing import Iterable

from torch.utils.data import Dataset

from mrvp.common.serialization import load_root_rows
from .schema import require_row_fields


def find_root_files(data_dir: str | Path, split: str | None = None) -> list[Path]:
    base = Path(data_dir)
    if split is not None:
        base = base / split
    files = sorted(base.glob("root_*.h5"))
    if not files and (base / "all").exists() and split is None:
        files = sorted((base / "all").glob("root_*.h5"))
    return files


class MRVPDataset(Dataset):
    def __init__(self, data_dir: str | Path, split: str = "train", group_by_root: bool = False, require_fields: bool = True):
        self.data_dir = Path(data_dir)
        self.split = split
        self.group_by_root = group_by_root
        self.root_files = find_root_files(self.data_dir, split)
        if not self.root_files:
            raise FileNotFoundError(f"no root shards found in {self.data_dir / split}")
        self._rows: list[tuple[int, int]] = []
        self._cache: dict[int, list[dict]] = {}
        for root_idx, path in enumerate(self.root_files):
            rows = load_root_rows(path)
            if require_fields:
                for row in rows:
                    require_row_fields(row)
            self._cache[root_idx] = rows
            for row_idx in range(len(rows)):
                self._rows.append((root_idx, row_idx))

    def __len__(self) -> int:
        return len(self.root_files) if self.group_by_root else len(self._rows)

    def __getitem__(self, idx: int):
        if self.group_by_root:
            return self._cache[idx]
        root_idx, row_idx = self._rows[idx]
        return self._cache[root_idx][row_idx]


def iter_rows(data_dir: str | Path, split: str) -> Iterable[dict]:
    ds = MRVPDataset(data_dir, split=split, group_by_root=False)
    for i in range(len(ds)):
        yield ds[i]


def iter_roots(data_dir: str | Path, split: str) -> Iterable[list[dict]]:
    ds = MRVPDataset(data_dir, split=split, group_by_root=True)
    for i in range(len(ds)):
        yield ds[i]
