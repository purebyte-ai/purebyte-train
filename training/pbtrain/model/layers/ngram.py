"""Hashed byte n-gram tables: the model's n-gram memory (spec/FORMAT.md §8.1).

For each position t, the n-grams that END at t are hashed (polynomial base 257, modulo 2^31-1, seed 1000*i + 7 +
7919*h for order index i and hash head h), looked up in one table per (order, head), concatenated and projected to
d_model. It is a RAM lookup: almost free on CPU, and the only architecture lever that has won so far (binaries: false
alarms / 13). The hash is specified exactly because the engine reproduces it bit for bit.

The gated variant scales the vector by a context gate before adding it:
    a_t = sigmoid( <rms(x_t), rms(e_t)> / sqrt(d) + b ),   x_t <- x_t + a_t * e_t
`b` (`ngram.gate_bias`, one float scalar) is born at +2 and draws no RNG: every other weight is born as without it.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

_MOD = (1 << 31) - 1
_BASE = 257


def ngram_hash(ids: torch.Tensor, n: int, buckets: int, seed: int) -> torch.Tensor:
    """ids (B,T) int64 bytes -> bucket (B,T) of the n-gram that ends at t (positions before 0 read as byte 0)."""
    h = torch.full_like(ids, seed % _MOD)
    for k in range(n - 1, -1, -1):
        prev = F.pad(ids, (k, 0))[:, : ids.shape[1]] if k else ids
        h = (h * _BASE + prev + 1) % _MOD
    return h % buckets


def hash_seed(order_index: int, head: int = 0) -> int:
    return 1000 * order_index + 7 + 7919 * head


class NGramEmbedding(nn.Module):
    """Tables per (order, hash head), concatenated order-major then head, projected to d_model."""

    def __init__(self, orders, buckets: int, dim: int, d_model: int, heads: int = 1):
        super().__init__()
        self.orders = tuple(orders)
        self.buckets, self.dim, self.heads = buckets, dim, heads
        if dim % heads:
            raise ValueError(f"ngram_dim {dim} is not divisible by ngram_heads {heads}")
        self.tables = nn.ModuleList([nn.Embedding(buckets, dim // heads) for _ in self.orders for _ in range(heads)])
        for t in self.tables:
            nn.init.normal_(t.weight, std=dim ** -0.5)
        self.proj = nn.Linear(dim * len(self.orders), d_model, bias=False)
        nn.init.normal_(self.proj.weight, std=(dim * len(self.orders)) ** -0.5)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        K = self.heads
        parts = [self.tables[i * K + h](ngram_hash(ids, n, self.buckets, hash_seed(i, h)))
                 for i, n in enumerate(self.orders) for h in range(K)]
        return self.proj(torch.cat(parts, dim=-1))

    def inject(self, x, e):
        """Adds the n-gram vector to the residual stream (ungated)."""
        return x + e

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())


def rms(x, eps: float = 1e-5):
    """RMSNorm WITHOUT weights, float32."""
    x = x.float()
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


class GatedNGramEmbedding(NGramEmbedding):
    def __init__(self, orders, buckets: int, dim: int, d_model: int, heads: int = 1):
        super().__init__(orders, buckets, dim, d_model, heads=heads)
        self.gate_bias = nn.Parameter(torch.tensor(2.0))   # no RNG drawn

    def gate(self, x, e):
        """(B,T,d), (B,T,d) -> a (B,T,1), float32."""
        s = (rms(x) * rms(e)).sum(-1, keepdim=True) / math.sqrt(x.shape[-1])
        return torch.sigmoid(s + self.gate_bias.float())

    def inject(self, x, e):
        return x + (self.gate(x, e) * e.float()).to(x.dtype)
