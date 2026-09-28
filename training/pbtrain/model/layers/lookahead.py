"""Bounded lookahead over the final hidden states (spec/FORMAT.md §10).

    y[t] = h[t] + sum_{j=1..k} w[:, j-1] * h[t+j],     h[t+j] = 0 past the last VALID byte of the window

The causal detector decides B/I of a span before seeing the value's end; this gives every head the k bytes to the RIGHT
and nothing further (the task is local on both sides). Depthwise, FLOAT, ZERO-init: the model at step 0 is exactly the
causal one. The loop of shifted multiply-adds (not F.conv1d) is deliberate: deterministic on every backend and literally
the engine's loop (acc = w0*h[t+1]; acc += w1*h[t+2]; ...; y = h[t] + acc).

The parameter is named `conv_w` so the optimizer's name rule keeps it out of weight decay. Never LM-pretrain a
lookahead preset: it would learn to copy future bytes.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Lookahead(nn.Module):
    def __init__(self, d: int, k: int):
        super().__init__()
        if k < 1:
            raise ValueError(f"lookahead width must be >= 1, got {k}")
        self.k = k
        self.conv_w = nn.Parameter(torch.zeros(d, k))

    def forward(self, h, mask=None):
        """h (B,T,d) -> (B,T,d). `mask` (B,T) bool with padding at the END: padded positions are never read."""
        T = h.shape[1]
        hv = h if mask is None else h * mask[..., None].to(h.dtype)
        hp = F.pad(hv, (0, 0, 0, self.k))                  # (B, T+k, d): zeros to the right of the window
        w = self.conv_w.to(h.dtype)
        acc = hp[:, 1:T + 1] * w[:, 0]
        for j in range(2, self.k + 1):
            acc = acc + hp[:, j:T + j] * w[:, j - 1]
        return h + acc
