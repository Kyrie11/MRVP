from __future__ import annotations

import argparse

import torch

from mrvp.common.config import deep_update, load_config
from mrvp.common.seed import set_seed
from mrvp.models.cmrt import CMRT
from mrvp.models.losses import rpfn_loss
from mrvp.models.rpfn import RPFN
from .utils import load_state_if_exists, make_loader, move_batch, prepare_run, save_checkpoint, train_defaults, write_metrics


def run_epoch(cmrt, rpfn, loader, optimizer, device: str, loss_cfg: dict, train: bool) -> dict[str, float]:
    total = 0.0
    count = 0
    cmrt.eval()
    rpfn.train(train)
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.no_grad():
            reset = cmrt.sample(batch, num_samples=1)[0]
        with torch.set_grad_enabled(train):
            out = rpfn(reset)
            loss, metrics = rpfn_loss(out, batch, loss_cfg)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(rpfn.parameters(), 5.0)
                optimizer.step()
        total += metrics["loss"]
        count += 1
    return {"loss": total / max(1, count)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--cmrt", required=True)
    parser.add_argument("--rpfn", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    set_seed(int(cfg.get("seed", 42)))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cmrt_ckpt = torch.load(args.cmrt, map_location=device)
    rpfn_ckpt = torch.load(args.rpfn, map_location=device)
    cmrt = CMRT(cmrt_ckpt.get("cfg", {})).to(device)
    rpfn = RPFN(rpfn_ckpt.get("cfg", {})).to(device)
    load_state_if_exists(cmrt, args.cmrt, device)
    load_state_if_exists(rpfn, args.rpfn, device)
    epochs, batch_size, lr, workers = train_defaults(cfg)
    train_loader = make_loader(args.data, "train", batch_size, workers, shuffle=True)
    val_loader = make_loader(args.data, "val", batch_size, workers, shuffle=False)
    optimizer = torch.optim.AdamW(rpfn.parameters(), lr=lr)
    out_dir = prepare_run(args.output, cfg)
    best = float("inf")
    for epoch in range(epochs):
        train_metrics = run_epoch(cmrt, rpfn, train_loader, optimizer, device, cfg.get("loss", {}), train=True)
        val_metrics = run_epoch(cmrt, rpfn, val_loader, optimizer, device, cfg.get("loss", {}), train=False)
        write_metrics(out_dir, "train", epoch, train_metrics)
        write_metrics(out_dir, "val", epoch, val_metrics)
        if val_metrics["loss"] <= best:
            best = val_metrics["loss"]
            save_checkpoint(out_dir / "checkpoints" / "best.pt", rpfn, rpfn_ckpt.get("cfg", {}), {"val_loss": best})


if __name__ == "__main__":
    main()
