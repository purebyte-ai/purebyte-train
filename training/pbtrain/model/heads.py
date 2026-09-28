"""Task heads, always float (spec/FORMAT.md §11). Public registry names: `choice`, `tag`, `ordinal`. The runtime
also defines an `exit` head (early exit): the exporter writes one when it is given its weights
(`export(..., exit_head=...)`), and pbtrain does not train one. `ByteLMHead` (next-byte prediction) is not a registered
head and is never exported; nothing in the factory uses it.

Pooling (for window-level heads): `max` is the right inductive bias for detection ("is there a needle anywhere?"): the
`last_mean` pooling dilutes a 40-byte span 1:13 in a 512-byte window and left the secrets accuracy at 0.53 with chance
at 0.51. Every secrets specialist uses a `choice` head with `max` pooling as its document gate.
"""
import torch
import torch.nn as nn

POOLINGS = ("last", "mean", "max", "last_mean", "max_mean")


class ByteLMHead(nn.Module):
    """Next-byte / span-MLM pretraining only. Never exported."""

    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Linear(d_model, 256, bias=False)

    def forward(self, h):
        return self.proj(h)


def pool(h, mask=None, mode="last_mean"):
    """(B,T,d) -> one vector per window (spec/FORMAT.md §11.1)."""
    if mask is None:
        last, mean = h[:, -1], h.mean(1)
        mx = h.max(1).values
    else:
        m = mask.to(h.dtype).unsqueeze(-1)
        idx = mask.sum(1).clamp(min=1) - 1
        last = h[torch.arange(h.shape[0], device=h.device), idx]
        mean = (h * m).sum(1) / m.sum(1).clamp(min=1e-6)
        mx = h.masked_fill(~mask.unsqueeze(-1), float("-inf")).max(1).values
    if mode == "last":
        return last
    if mode == "mean":
        return mean
    if mode == "max":
        return mx
    if mode == "max_mean":
        return torch.cat([mx, mean], dim=-1)
    return torch.cat([last, mean], dim=-1)


class ChoiceHead(nn.Module):
    """`choice`: a distribution over declared classes (class 0 = negative). The policy decides, the model does not."""
    TYPE = "choice"

    def __init__(self, d_model: int, n_classes: int, pooling: str = "last_mean", hidden: int = 0):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"unknown pooling {pooling!r}; known: {POOLINGS}")
        d_in = d_model * (2 if pooling in ("last_mean", "max_mean") else 1)
        hidden = hidden or max(64, d_model // 2)
        self.pooling = pooling
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(), nn.Linear(hidden, n_classes))

    def forward(self, h, mask=None):
        return self.mlp(pool(h, mask, self.pooling))


class TagHead(nn.Module):
    """`tag`: BIOES emissions per byte for k entity types (1 + 4k labels). Decoding is `decode.viterbi`."""
    TYPE = "tag"

    def __init__(self, d_model: int, n_labels: int):
        super().__init__()
        self.proj = nn.Linear(d_model, n_labels)

    @property
    def n_entities(self):
        return (self.proj.out_features - 1) // 4

    def forward(self, h, mask=None):
        return self.proj(h)


class OrdinalHead(nn.Module):
    """`ordinal`: CORAL, K-1 cumulative thresholds sharing one weight vector (exported as `heads.<i>.w.weight` and
    `heads.<i>.b`; training/tests/test_export_reference.py checks it against the NumPy reference)."""
    TYPE = "ordinal"

    def __init__(self, d_model: int, n_levels: int, pooling: str = "last_mean"):
        super().__init__()
        if pooling not in POOLINGS:
            raise ValueError(f"unknown pooling {pooling!r}; known: {POOLINGS}")
        if n_levels < 2:
            raise ValueError(f"an ordinal head needs at least 2 levels, got {n_levels}")
        d_in = d_model * (2 if pooling in ("last_mean", "max_mean") else 1)
        self.pooling, self.K = pooling, n_levels
        self.w = nn.Linear(d_in, 1, bias=False)
        self.b = nn.Parameter(torch.zeros(n_levels - 1))

    def forward(self, h, mask=None):
        z = self.w(pool(h, mask, self.pooling))                 # (B,1)
        return z + torch.sort(self.b, descending=True).values   # sorted thresholds
