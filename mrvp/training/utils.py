from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from mrvp.common.config import get_cfg, save_config
from mrvp.common.logging import write_jsonl
from mrvp.data.collate import collate_rows
from mrvp.data.dataset import MRVPDataset


def move_batch(batch: dict[str, Any], device: str) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def make_loader(data_dir: str, split: str, batch_size: int, num_workers: int = 0, shuffle: bool = True) -> DataLoader:
    ds = MRVPDataset(data_dir, split=split, group_by_root=False)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate_rows)


def prepare_run(output: str | Path, cfg: dict) -> Path:
    out = Path(output)
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out / "metrics").mkdir(parents=True, exist_ok=True)
    save_config(cfg, out / "config_resolved.yaml")
    return out


def save_checkpoint(path: str | Path, model: torch.nn.Module, cfg: dict, extra: dict | None = None) -> None:
    data = {"model_state": model.state_dict(), "cfg": cfg, "extra": extra or {}}
    torch.save(data, path)


def load_state_if_exists(model: torch.nn.Module, path: str | Path | None, device: str = "cpu") -> dict:
    if path is None:
        return {}
    ckpt_path = Path(path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=False)
    return ckpt if isinstance(ckpt, dict) else {}


def write_metrics(output: Path, split: str, epoch: int, metrics: dict[str, float]) -> None:
    row = {"epoch": int(epoch), **{k: float(v) for k, v in metrics.items()}}
    write_jsonl(output / "metrics" / f"{split}_metrics.jsonl", row)


def train_defaults(cfg: dict) -> tuple[int, int, float, int]:
    return (
        int(get_cfg(cfg, "train.epochs", 1)),
        int(get_cfg(cfg, "train.batch_size", 8)),
        float(get_cfg(cfg, "train.lr", 3e-4)),
        int(get_cfg(cfg, "train.num_workers", 0)),
    )
