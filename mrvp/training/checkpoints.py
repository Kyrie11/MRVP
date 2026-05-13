from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Type

import torch


def save_checkpoint(path: str | Path, model: torch.nn.Module, extra: Dict[str, Any] | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model_state": model.state_dict(), "extra": extra or {}}, path)


def load_state(path: str | Path, map_location: str | torch.device = "cpu") -> Dict[str, Any]:
    return torch.load(Path(path), map_location=map_location)


def load_model(model: torch.nn.Module, path: str | Path, map_location: str | torch.device = "cpu", strict: bool = True) -> torch.nn.Module:
    ckpt = load_state(path, map_location=map_location)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=strict)
    return model
