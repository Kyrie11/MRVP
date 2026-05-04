from __future__ import annotations

from typing import Dict, Iterable, Mapping

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


def to_device(batch, device: torch.device):
    if isinstance(batch, dict):
        return {k: to_device(v, device) for k, v in batch.items()}
    if torch.is_tensor(batch):
        return batch.to(device)
    if isinstance(batch, (list, tuple)):
        return batch
    return batch


def scalar_logs(logs: Mapping[str, torch.Tensor | float]) -> Dict[str, float]:
    out = {}
    for k, v in logs.items():
        if torch.is_tensor(v):
            out[k] = float(v.detach().cpu().mean())
        else:
            out[k] = float(v)
    return out


def train_one_epoch(model, loader: DataLoader, optimizer, device: torch.device, loss_kwargs: dict | None = None, desc: str = "train") -> Dict[str, float]:
    model.train()
    totals = {}
    n = 0
    pbar = tqdm(loader, desc=desc, leave=False)
    for batch in pbar:
        batch = to_device(batch, device)
        optimizer.zero_grad(set_to_none=True)
        logs = model.loss(batch, **(loss_kwargs or {}))
        logs["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        s = scalar_logs(logs)
        for k, v in s.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
        pbar.set_postfix({"loss": s.get("loss", 0.0)})
    return {k: v / max(n, 1) for k, v in totals.items()}


@torch.no_grad()
def eval_loss(model, loader: DataLoader, device: torch.device, loss_kwargs: dict | None = None, desc: str = "val") -> Dict[str, float]:
    model.eval()
    totals = {}
    n = 0
    for batch in tqdm(loader, desc=desc, leave=False):
        batch = to_device(batch, device)
        logs = model.loss(batch, **(loss_kwargs or {}))
        s = scalar_logs(logs)
        for k, v in s.items():
            totals[k] = totals.get(k, 0.0) + v
        n += 1
    return {k: v / max(n, 1) for k, v in totals.items()}
