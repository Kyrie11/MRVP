from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
import torch.nn.functional as F

from .encoders import MLP, TokenAttention


@dataclass
class CMRTConfig:
    dim: int = 64
    slot_count: int = 16
    slot_dim: int = 64
    slot_layers: int = 2
    tau_steps: int = 11
    reset_state_dim: int = 12
    deg_dim: int = 8
    world_size: int = 64
    channels_A: int = 6
    steps_O: int = 15
    channels_O: int = 1
    channels_G: int = 3
    channels_Y: int = 2
    use_action_intervention_memory: bool = True
    use_counterfactual_slot_update: bool = True
    use_degradation_decoder: bool = True
    use_recovery_world_decoder: bool = True
    use_same_root_contrastive_loss: bool = True


class CMRT(nn.Module):
    def __init__(self, cfg: CMRTConfig | dict | None = None):
        super().__init__()
        if isinstance(cfg, dict):
            cfg = CMRTConfig(**{k: v for k, v in cfg.items() if k in CMRTConfig.__annotations__})
        self.cfg = cfg or CMRTConfig()
        d = self.cfg.dim
        self.root_encoder = MLP([12 + 5 * 6 * 8, d, d])
        self.action_encoder = MLP([self.cfg.tau_steps * 12 + (self.cfg.tau_steps - 1) * 3, d, d])
        self.nominal_encoder = MLP([self.cfg.tau_steps * 12 + (self.cfg.tau_steps - 1) * 3, d, d])
        self.slot_init = nn.Parameter(torch.randn(self.cfg.slot_count, d) * 0.02)
        self.action_to_slot = nn.Linear(d, d)
        self.scene_attn = nn.ModuleList([TokenAttention(d) for _ in range(self.cfg.slot_layers)])
        self.action_attn = nn.ModuleList([TokenAttention(d) for _ in range(self.cfg.slot_layers)])
        self.nominal_attn = nn.ModuleList([TokenAttention(d) for _ in range(self.cfg.slot_layers)])
        self.tau_head = nn.Linear(d, self.cfg.tau_steps)
        self.r_head = nn.Linear(d, self.cfg.reset_state_dim * 2)
        self.d_head = nn.Linear(d, self.cfg.deg_dim)
        self.sigma_head = nn.Linear(d, self.cfg.reset_state_dim)
        self.world_channels = self.cfg.channels_A + self.cfg.steps_O * self.cfg.channels_O + self.cfg.channels_G + self.cfg.steps_O * self.cfg.channels_Y
        self.world_seed = nn.Linear(d, self.world_channels * 8 * 8)
        self.world_decoder = nn.Sequential(
            nn.Conv2d(self.world_channels, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
            nn.Conv2d(64, self.world_channels, 1),
        )
        self.cf_target_proj = nn.Linear(self.cfg.reset_state_dim + self.cfg.deg_dim + 1, d)

    def _flat_action(self, batch: dict) -> torch.Tensor:
        pr = batch["prefix_rollout"]
        pc = batch["prefix_controls"]
        return torch.cat([pr.flatten(1), pc.flatten(1)], dim=-1)

    def _flat_root(self, batch: dict) -> torch.Tensor:
        o = batch["o_hist"]
        x = batch["x_t"]
        flat = o.flatten(1)
        need = 5 * 6 * 8
        if flat.shape[1] < need:
            flat = F.pad(flat, (0, need - flat.shape[1]))
        flat = flat[:, :need]
        return torch.cat([x, flat], dim=-1)

    def encode_slots(self, batch: dict) -> torch.Tensor:
        B = batch["x_t"].shape[0]
        root_mem = self.root_encoder(self._flat_root(batch)).unsqueeze(1)
        action_mem = self.action_encoder(self._flat_action(batch)).unsqueeze(1)
        nominal_mem = self.nominal_encoder(self._flat_action(batch) * 0.0).unsqueeze(1)
        q = self.slot_init.unsqueeze(0).expand(B, -1, -1) + self.action_to_slot(action_mem).expand(-1, self.cfg.slot_count, -1)
        q0 = self.slot_init.unsqueeze(0).expand(B, -1, -1)
        for s_attn, a_attn, n_attn in zip(self.scene_attn, self.action_attn, self.nominal_attn):
            q_scene = s_attn(q, root_mem)
            if self.cfg.use_action_intervention_memory:
                qa = a_attn(q_scene, action_mem)
            else:
                qa = q_scene
            if self.cfg.use_counterfactual_slot_update:
                qn = n_attn(q0, nominal_mem)
                q = q_scene + (qa - qn)
            else:
                q = qa
        return q

    def forward(self, batch: dict) -> dict:
        z = self.encode_slots(batch)
        summary = z.mean(dim=1)
        tau_logits = self.tau_head(summary)
        r_params = self.r_head(summary)
        r_mu, r_logstd = torch.chunk(r_params, 2, dim=-1)
        d_pred = torch.sigmoid(self.d_head(summary))
        if not self.cfg.use_degradation_decoder:
            d_pred = torch.ones_like(d_pred)
            d_pred[:, 0] = 0.95
            d_pred[:, 4] = 0.0
            d_pred[:, 5] = 0.35
            d_pred[:, 6] = 1.2
            d_pred[:, 7] = 0.0
        sigma = F.softplus(self.sigma_head(summary)) + 1e-3
        B = summary.shape[0]
        world_low = self.world_seed(summary).view(B, self.world_channels, 8, 8)
        world = F.interpolate(world_low, size=(self.cfg.world_size, self.cfg.world_size), mode="bilinear", align_corners=False)
        world = self.world_decoder(world)
        if not self.cfg.use_recovery_world_decoder:
            world = torch.zeros_like(world)
        cA = self.cfg.channels_A
        cO = self.cfg.steps_O * self.cfg.channels_O
        cG = self.cfg.channels_G
        world_A = world[:, :cA]
        world_O = world[:, cA:cA + cO].view(B, self.cfg.steps_O, self.cfg.channels_O, self.cfg.world_size, self.cfg.world_size)
        world_G = world[:, cA + cO:cA + cO + cG]
        world_Y = world[:, cA + cO + cG:].view(B, self.cfg.steps_O, self.cfg.channels_Y, self.cfg.world_size, self.cfg.world_size)
        return {
            "tau_logits": tau_logits,
            "r_mu": r_mu,
            "r_logstd": torch.clamp(r_logstd, -5.0, 3.0),
            "d_pred": d_pred,
            "z_slots": z,
            "z_summary": summary,
            "world_A": world_A,
            "world_O": world_O,
            "world_G": world_G,
            "world_Y": world_Y,
            "sigma": sigma,
            "cf_target_proj": self.cf_target_proj,
        }

    def sample(self, batch: dict, num_samples: int = 1) -> list[dict]:
        out = self.forward(batch)
        samples = []
        for _ in range(int(num_samples)):
            eps = torch.randn_like(out["r_mu"])
            r = out["r_mu"] + eps * torch.exp(out["r_logstd"])
            samples.append({
                "r_reset": r,
                "z_slots": out["z_slots"],
                "world_A": torch.sigmoid(out["world_A"]),
                "world_O": torch.sigmoid(out["world_O"]),
                "world_G": torch.sigmoid(out["world_G"]),
                "world_Y": torch.tanh(out["world_Y"]),
                "deg": out["d_pred"],
                "sigma": out["sigma"],
            })
        return samples
