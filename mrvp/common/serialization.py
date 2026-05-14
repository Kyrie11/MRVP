from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _write_value(group: h5py.Group, key: str, value: Any) -> None:
    if isinstance(value, dict):
        sub = group.create_group(key)
        for k, v in value.items():
            _write_value(sub, str(k), v)
        return
    if isinstance(value, (str, bytes)):
        group.attrs[key] = value.decode("utf-8") if isinstance(value, bytes) else value
        return
    if isinstance(value, (bool, int, float, np.integer, np.floating)):
        group.attrs[key] = value
        return
    arr = np.asarray(value)
    if arr.dtype.kind in {"U", "O"}:
        group.attrs[key] = json.dumps(arr.tolist(), ensure_ascii=False)
    else:
        group.create_dataset(key, data=arr, compression="gzip")


def _read_group(group: h5py.Group) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in group.attrs.items():
        if isinstance(value, bytes):
            out[key] = value.decode("utf-8")
        else:
            out[key] = value.item() if hasattr(value, "item") else value
    for key, item in group.items():
        if isinstance(item, h5py.Group):
            out[key] = _read_group(item)
        else:
            out[key] = item[()]
    return out


def save_root_rows(path: str | Path, rows: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("at least one row is required")
    with h5py.File(out, "w") as h5:
        h5.attrs["root_id"] = str(rows[0]["root_id"])
        h5.attrs["scenario_family"] = str(rows[0].get("scenario_family", "unknown"))
        h5.attrs["sim_source"] = str(rows[0].get("sim_source", "unknown"))
        actions = h5.create_group("actions")
        for idx, row in enumerate(rows):
            action_id = str(row.get("action_id", idx))
            g = actions.create_group(action_id)
            for key, value in row.items():
                _write_value(g, key, value)


def load_root_rows(path: str | Path) -> list[dict[str, Any]]:
    with h5py.File(path, "r") as h5:
        rows = []
        for action_id in sorted(h5["actions"].keys()):
            row = _read_group(h5["actions"][action_id])
            rows.append(row)
        return rows


def write_json(path: str | Path, data: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def read_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
