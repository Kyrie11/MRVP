from __future__ import annotations

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self, dims: list[int], activation: nn.Module | None = None, final_activation: nn.Module | None = None):
        super().__init__()
        activation = activation or nn.ReLU()
        layers = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(activation)
            elif final_activation is not None:
                layers.append(final_activation)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BEVEncoder(nn.Module):
    def __init__(self, in_channels: int, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, dim // 2, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(dim // 2, dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(dim, dim, 3, stride=2, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).flatten(1)


class TokenAttention(nn.Module):
    def __init__(self, dim: int, heads: int = 4):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4 * dim), nn.ReLU(), nn.Linear(4 * dim, dim))
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, q: torch.Tensor, mem: torch.Tensor) -> torch.Tensor:
        out, _ = self.attn(q, mem, mem, need_weights=False)
        q = self.norm(q + out)
        return self.norm2(q + self.ff(q))
