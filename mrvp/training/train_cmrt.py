from __future__ import annotations

import argparse

import torch

from mrvp.common.config import deep_update, get_cfg, load_config
from mrvp.common.seed import set_seed
from mrvp.models.cmrt import CMRT
from mrvp.models.losses import cmrt_loss
from .utils import make_loader, move_batch, prepare_run, save_checkpoint, train_defaults, write_metrics


def build_model_cfg(cfg: dict) -> dict:
    return deep_update(cfg.get("model", {}), cfg.get("cmrt", {}))


def run_epoch(model, loader, optimizer, device: str, loss_cfg: dict, train: bool) -> dict[str, float]:
    total = 0.0
    count = 0
    model.train(train)
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.set_grad_enabled(train):
            out = model(batch)
            loss, metrics = cmrt_loss(out, batch, loss_cfg)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total += metrics["loss"]
        count += 1
    return {"loss": total / max(1, count)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    epochs, batch_size, lr, workers = train_defaults(cfg)
    train_loader = make_loader(args.data, "train", batch_size, workers, shuffle=True)
    val_loader = make_loader(args.data, "val", batch_size, workers, shuffle=False)
    model = CMRT(build_model_cfg(cfg)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    out_dir = prepare_run(args.output, cfg)
    best = float("inf")
    for epoch in range(epochs):
        train_metrics = run_epoch(model, train_loader, optimizer, device, cfg.get("loss", {}), train=True)
        val_metrics = run_epoch(model, val_loader, optimizer, device, cfg.get("loss", {}), train=False)
        write_metrics(out_dir, "train", epoch, train_metrics)
        write_metrics(out_dir, "val", epoch, val_metrics)
        if val_metrics["loss"] <= best:
            best = val_metrics["loss"]
            save_checkpoint(out_dir / "checkpoints" / "best.pt", model, build_model_cfg(cfg), {"val_loss": best})


if __name__ == "__main__":
    main()
