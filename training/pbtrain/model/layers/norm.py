"""RMSNorm and per-group gated RMSNorm (spec/FORMAT.md §5 "RMS normalization", §9.1 step 7)."""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, d: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d))
        self.eps = eps

    def forward(self, x):
        dt = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dt)


class GatedRMSNorm(nn.Module):
    """Normalizes each of `n_groups` channel groups after the scan (the gate is applied by the block before it)."""

    def __init__(self, d: int, n_groups: int, eps: float = 1e-5):
        super().__init__()
        if d % n_groups:
            raise ValueError(f"d {d} not divisible by n_groups {n_groups}")
        self.weight = nn.Parameter(torch.ones(d))
        self.n_groups, self.eps = n_groups, eps

    def forward(self, x):
        dt = x.dtype
        *lead, d = x.shape
        g = x.float().reshape(*lead, self.n_groups, d // self.n_groups)
        g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + self.eps)
        return (g.reshape(*lead, d) * self.weight.float()).to(dt)
