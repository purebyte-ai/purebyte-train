"""Content fingerprints of byte data, for the exam-leakage check. ONE algorithm, used by the data build scripts
(`data/`: exam fingerprints, corpus filtering) and by the factory's `data` stage (every training split, stream and
guard set is checked against every registered exam).

Content-anchored blocks, independent of offsets: a position p is an ANCHOR when the 4 bytes at p hash into 1/64 of the
32-bit space; the FINGERPRINT of an anchor is a 64-bit hash of the BLOCK of 64 bytes starting at p. Two byte strings that
share a region of L bytes share, on average, (L - 67) / 64 anchors: a shared region of 512 bytes is missed with
probability ~ (63/64)^445 < 0.1 %; of 1 KB, < 1e-6. Blocks with fewer than 4 distinct byte values (padding, zero runs)
identify no file and are skipped.

    h4(p)     = little-endian uint32 of data[p:p+4]
    anchor(p) = ((h4(p) * 0x9E3779B1) mod 2^32) >> 26 == 0
    block(p)  = sum_{i<64} data[p+i] * K^(63-i)  mod 2^64      (K = 0x100000001B3, uint64 wrap-around)

Everything is vectorised with NumPy; long inputs are processed in 1 MiB pieces that overlap by 67 bytes.
"""
import json

import numpy as np

BLOCK = 64
ANCHOR_BITS = 6                      # 1 anchor per 64 positions on random data
MIN_DISTINCT = 4
_MIX = np.uint32(0x9E3779B1)
_K = 0x100000001B3
_POW = np.array([pow(_K, BLOCK - 1 - i, 1 << 64) for i in range(BLOCK)], dtype=np.uint64)
_PIECE = 1 << 20


def _anchor_mask(a):
    """a uint8 [..., n] -> bool [..., n - BLOCK + 1]: anchors whose block fits."""
    n = a.shape[-1] - BLOCK + 1
    if n <= 0:
        return np.zeros(a.shape[:-1] + (0,), dtype=bool)
    b = a[..., : n + 3].astype(np.uint32)
    h4 = b[..., :n] | (b[..., 1:n + 1] << 8) | (b[..., 2:n + 2] << 16) | (b[..., 3:n + 3] << 24)
    return ((h4 * _MIX) >> np.uint32(32 - ANCHOR_BITS)) == 0


def _hash_rows(rows):
    """rows uint8 [k, BLOCK] -> uint64 [k] fingerprints; rows with < MIN_DISTINCT byte values -> dropped (mask)."""
    if len(rows) == 0:
        return np.zeros(0, np.uint64), np.zeros(0, bool)
    s = np.sort(rows, axis=1)
    distinct = (s[:, 1:] != s[:, :-1]).sum(1) + 1
    with np.errstate(over="ignore"):
        h = (rows.astype(np.uint64) * _POW).sum(axis=1, dtype=np.uint64)
    return h, distinct >= MIN_DISTINCT


def fingerprints(data):
    """Sorted unique uint64 fingerprints of `data` (bytes-like or a uint8 array)."""
    a = data if isinstance(data, np.ndarray) else np.frombuffer(bytes(data), dtype=np.uint8)
    a = a.astype(np.uint8, copy=False).reshape(-1)
    out = []
    for lo in range(0, max(1, len(a)), _PIECE):
        piece = a[lo: lo + _PIECE + BLOCK + 3]
        pos = np.flatnonzero(_anchor_mask(piece))
        if lo + _PIECE < len(a):
            pos = pos[pos < _PIECE]              # the overlap belongs to the next piece
        if len(pos):
            h, ok = _hash_rows(np.lib.stride_tricks.sliding_window_view(piece, BLOCK)[pos])
            out.append(h[ok])
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.uint64)


def window_fingerprints(X):
    """Sorted unique uint64 fingerprints of a set of windows X uint8 [n, T]: blocks inside each window only (a block
    never spans two windows)."""
    X = np.asarray(X, dtype=np.uint8)
    if X.ndim != 2 or X.shape[1] < BLOCK or len(X) == 0:
        return np.zeros(0, np.uint64)
    out = []
    for lo in range(0, len(X), 4096):
        part = X[lo: lo + 4096]
        r, c = np.nonzero(_anchor_mask(part))
        if len(r):
            h, ok = _hash_rows(np.lib.stride_tricks.sliding_window_view(part, BLOCK, axis=1)[r, c])
            out.append(h[ok])
    return np.unique(np.concatenate(out)) if out else np.zeros(0, np.uint64)


def window_hits(X, exam_fp):
    """X uint8 [n, T] windows -> bool [n]: True where a window shares at least one block with `exam_fp`."""
    X = np.asarray(X, dtype=np.uint8)
    hits = np.zeros(len(X), dtype=bool)
    exam_fp = np.asarray(exam_fp, dtype=np.uint64)
    if len(exam_fp) == 0 or X.shape[1] < BLOCK:
        return hits
    for lo in range(0, len(X), 4096):
        part = X[lo: lo + 4096]
        r, c = np.nonzero(_anchor_mask(part))
        if len(r) == 0:
            continue
        rows = np.lib.stride_tricks.sliding_window_view(part, BLOCK, axis=1)[r, c]
        h, ok = _hash_rows(rows)
        found = np.isin(h, exam_fp) & ok
        hits[lo + r[found]] = True
    return hits


def save(path, fp, meta):
    """An exam fingerprint file: `fp` sorted uint64 + a JSON `meta` (name, manifest sha256, files, bytes, algorithm)."""
    meta = dict(meta, algorithm=f"anchored-{BLOCK}B-1/{1 << ANCHOR_BITS}-min{MIN_DISTINCT}")
    np.savez_compressed(path, fp=np.asarray(fp, np.uint64), meta=json.dumps(meta, sort_keys=True))


def load(path):
    with np.load(path) as z:
        return z["fp"], json.loads(str(z["meta"]))
