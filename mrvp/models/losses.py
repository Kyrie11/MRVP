from __future__ import annotations

import torch
import torch.nn.functional as F


def cmrt_loss(out: dict, batch: dict, cfg: dict | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = cfg or {}
    lambda_w = float(cfg.get("lambda_w", 1.0))
    lambda_cf = float(cfg.get("lambda_cf", 0.1))
    lambda_u = float(cfg.get("lambda_u", 0.01))
    loss_tau = F.cross_entropy(out["tau_logits"], batch["tau_index"].clamp(max=out["tau_logits"].shape[-1] - 1))
    loss_r = F.l1_loss(out["r_mu"], batch["r_reset"])
    loss_d = F.l1_loss(out["d_pred"], batch["deg"])
    loss_A = F.binary_cross_entropy_with_logits(out["world_A"], batch["world_A"])
    loss_O = F.binary_cross_entropy_with_logits(out["world_O"], batch["world_O"])
    loss_G = F.binary_cross_entropy_with_logits(out["world_G"], batch["world_G"])
    loss_Y = F.l1_loss(torch.tanh(out["world_Y"]), batch["world_Y"])
    loss_unc = torch.mean(out["r_logstd"].pow(2))
    cf = torch.tensor(0.0, device=loss_tau.device)
    if batch.get("root_id") is not None and len(batch["root_id"]) > 1:
        z = F.normalize(out["z_summary"], dim=-1)
        y = F.normalize(torch.cat([batch["r_reset"], batch["deg"], batch["score_star"].unsqueeze(-1)], dim=-1), dim=-1)
        proj = out.get("cf_target_proj")
        if proj is not None:
            y = proj(y)
        sim = z @ y.t() / 0.1
        labels = torch.arange(sim.shape[0], device=sim.device)
        cf = F.cross_entropy(sim, labels)
    total = loss_tau + loss_r + loss_d + lambda_w * (loss_A + loss_O + loss_G + loss_Y) + lambda_u * loss_unc + lambda_cf * cf
    metrics = {"loss": float(total.detach().cpu()), "loss_tau": float(loss_tau.detach().cpu()), "loss_r": float(loss_r.detach().cpu()), "loss_d": float(loss_d.detach().cpu())}
    return total, metrics


def rpfn_loss(out: dict, batch: dict, cfg: dict | None = None) -> tuple[torch.Tensor, dict[str, float]]:
    cfg = cfg or {}
    lambda_c = float(cfg.get("lambda_c", 1.0))
    lambda_f = float(cfg.get("lambda_f", 0.2))
    controls = out["controls"]
    target_u = batch["teacher_u"].unsqueeze(1).expand_as(controls)
    branch_err = torch.mean(torch.abs(controls - target_u), dim=(-1, -2))
    loss_exec = torch.min(branch_err, dim=1).values.mean()
    cert = out["cert_pred"]
    target_cert = batch["score_star"].unsqueeze(1).expand_as(cert)
    loss_cert = F.huber_loss(cert, target_cert)
    loss_funnel = F.relu(-out["funnel_values"]).mean() + F.relu(-out["margins"]["goal"]).mean()
    total = loss_exec + lambda_c * loss_cert + lambda_f * loss_funnel
    metrics = {"loss": float(total.detach().cpu()), "loss_exec": float(loss_exec.detach().cpu()), "loss_cert": float(loss_cert.detach().cpu())}
    return total, metrics
