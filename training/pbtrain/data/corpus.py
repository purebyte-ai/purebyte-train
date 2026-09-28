"""Byte corpus over memmap: the real context bytes a generator draws windows from.

The holdout is ONE contiguous range at an end, never interleaved; a vertical cuts its train / select / test from
disjoint byte RANGES of the training part (`split_policy.unit: corpus_range`), so two splits never share a byte of real
context. `train_batch` and `val_batches` (next-byte batches, for language-model pretraining; the factory does not use
them) read the corpus directly; validation over a corpus uses SEQUENTIAL windows (random windows underestimate).
"""
import numpy as np
import torch


class ByteCorpus:
    def __init__(self, path: str, val_bytes: int = 4 << 20, hold_out_head: bool = False):
        self.data = np.memmap(path, dtype=np.uint8, mode="r")
        n = len(self.data)
        if val_bytes * 4 > n:
            val_bytes = n // 4
        if hold_out_head:
            self.val_lo, self.val_hi, self.tr_lo, self.tr_hi = 0, val_bytes, val_bytes, n
        else:
            self.val_lo, self.val_hi, self.tr_lo, self.tr_hi = n - val_bytes, n, 0, n - val_bytes
        self.path = path

    def __repr__(self):
        return f"ByteCorpus({self.path}, train={self.tr_hi - self.tr_lo}B, val={self.val_hi - self.val_lo}B)"

    def train_batch(self, bs: int, T: int, gen: np.random.Generator, device="cpu"):
        hi = self.tr_hi - T - 1
        off = gen.integers(self.tr_lo, hi, size=bs)
        win = np.stack([np.asarray(self.data[o:o + T + 1]) for o in off])
        x = torch.from_numpy(win[:, :-1].astype(np.int64))
        y = torch.from_numpy(win[:, 1:].astype(np.int64))
        return x.to(device), y.to(device)

    def val_batches(self, bs: int, T: int, max_bytes: int, device="cpu"):
        """Sequential, deterministic, non-overlapping windows."""
        lo, hi = self.val_lo, min(self.val_hi, self.val_lo + max_bytes)
        offs = list(range(lo, hi - T - 1, T))
        for i in range(0, len(offs) - bs + 1, bs):
            win = np.stack([np.asarray(self.data[o:o + T + 1]) for o in offs[i:i + bs]])
            x = torch.from_numpy(win[:, :-1].astype(np.int64))
            y = torch.from_numpy(win[:, 1:].astype(np.int64))
            yield x.to(device), y.to(device)
