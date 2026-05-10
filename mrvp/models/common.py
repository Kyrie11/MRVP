from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


class _GradientReverseFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return -ctx.scale * grad_output, None


def grad_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Gradient reversal layer used by optional adversarial diagnostics.

    Forward is the identity. During backpropagation the gradient wrt ``x`` is
    multiplied by ``-scale`` so a downstream classifier is trained to predict a
    nuisance variable while the upstream representation is trained to hide it.
    """
    return _GradientReverseFn.apply(x, float(scale))


def make_mlp(in_dim: int, hidden: Sequence[int], out_dim: int, dropout: float = 0.0, layer_norm: bool = False) -> nn.Sequential:
    layers = []
    last = in_dim
    for h in hidden:
        layers.append(nn.Linear(last, h))
        if layer_norm:
            layers.append(nn.LayerNorm(h))
        layers.append(nn.SiLU())
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        last = h
    layers.append(nn.Linear(last, out_dim))
    return nn.Sequential(*layers)


class SmoothMin(nn.Module):
    def __init__(self, tau: float = 0.1) -> None:
        super().__init__()
        self.tau = tau

    def forward(self, x: torch.Tensor, dim: int = -1) -> torch.Tensor:
        tau = max(float(self.tau), 1e-6)
        return -tau * torch.logsumexp(-x / tau, dim=dim)


def diag_gaussian_nll(x: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """Element-wise diagonal Gaussian NLL without constants, returns [...]."""
    inv_var = torch.exp(-2.0 * log_scale.clamp(-7, 5))
    return 0.5 * ((x - mean) ** 2 * inv_var + 2.0 * log_scale.clamp(-7, 5)).sum(dim=-1)


def mixture_diag_gaussian_nll(x: torch.Tensor, logits: torch.Tensor, mean: torch.Tensor, log_scale: torch.Tensor) -> torch.Tensor:
    """Mixture NLL for x [B,D], logits [B,R], mean/log_scale [B,R,D]."""
    x_exp = x[:, None, :].expand_as(mean)
    comp_nll = diag_gaussian_nll(x_exp, mean, log_scale)  # [B,R]
    log_mix = F.log_softmax(logits, dim=-1)
    return -torch.logsumexp(log_mix - comp_nll, dim=-1)


def empirical_cvar(losses: torch.Tensor, beta: float = 0.9, dim: int = -1) -> torch.Tensor:
    """Average the largest (1-beta) tail of nonnegative losses."""
    n = losses.shape[dim]
    k = max(1, int(torch.ceil(torch.tensor((1.0 - beta) * n)).item()))
    topk = torch.topk(losses, k=k, dim=dim).values
    return topk.mean(dim=dim)
