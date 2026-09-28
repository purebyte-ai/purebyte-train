"""The CONTENT half of the exclusion guarantee.

Every evaluation set declared in data/exams/*.yaml has a fingerprint file `$PUREBYTE_DATA/exams/fingerprints/<set>.npz`
written by `fingerprint_exams.py` with `pbtrain.data.fingerprint` (the one algorithm shared with the training factory:
64-byte blocks at content-defined anchors). A build:

1. loads every fingerprint file, and FAILS when one declared in data/exams/*.yaml is missing or was made from another
   version of its manifest (`--allow-missing-exams` turns that into a warning written into the build manifest);
2. checks every candidate against every set, and removes what shares a block:
     text     the lines around each shared block are cut out; a file is dropped when little is left
     binary   one contiguous span free of shared blocks is taken (never spliced); the file is dropped when there is none
3. verifies the finished corpus as a whole (junctions between files included) and repairs it until no block is shared.

Blocks are located with `window_hits` on overlapping windows (window T, stride <= T - 63, so every 64-byte block lies
inside at least one window), fed only the blocks this file shares, which makes the search cheap.
"""
import os

import numpy as np

from . import paths
from .manifest import exam_sets, load_exams
from .util import DataError, sha256_bytes, warn

TEXT_WINDOW, TEXT_STRIDE, TEXT_MARGIN = 256, 128, 128
BIN_WINDOW, BIN_STRIDE = 512, 256


class ExamFingerprints:
    def __init__(self, name, role, fp, meta, path):
        self.name, self.role, self.meta, self.path = name, role, meta, path
        self.fp = np.asarray(fp, dtype=np.uint64)

    def shared(self, fps):
        """The subset of the sorted unique array `fps` that is in this set."""
        if len(fps) == 0 or len(self.fp) == 0:
            return fps[:0]
        i = np.searchsorted(self.fp, fps)
        i[i >= len(self.fp)] = len(self.fp) - 1
        return fps[self.fp[i] == fps]

    def __len__(self):
        return len(self.fp)


def declared_sets(exams=None):
    """[(set name, role, manifest sha256)] from data/exams/*.yaml (or from the given exam manifests)."""
    return [(s["name"], s["role"], m.sha256) for s, m in exam_sets(load_exams() if exams is None else exams)]


def load_all(allow_missing=False, roles=("test", "dev"), fingerprint_dir=None, exams=None):
    """(exam fingerprint sets, report). Raises when a declared set is missing or stale, unless `allow_missing`."""
    F = paths.fingerprint_module()
    fdir = fingerprint_dir or paths.fingerprints_dir()
    report = dict(fingerprint_dir=paths.rel(fdir), required=[], missing=[], stale=[], undeclared=[], warnings=[],
                  allow_missing_exams=bool(allow_missing))
    sets, seen = [], set()
    for name, role, msha in declared_sets(exams):
        if role not in roles:
            continue
        seen.add(name)
        report["required"].append(name)
        path = os.path.join(fdir, f"{name}.npz")
        if not os.path.exists(path):
            report["missing"].append(name)
            continue
        fp, meta = F.load(path)
        if meta.get("manifest_sha256") != msha:
            report["stale"].append(name)
            continue
        sets.append(ExamFingerprints(name, meta.get("role", role), fp, meta, path))
    if os.path.isdir(fdir):
        for f in sorted(os.listdir(fdir)):
            if f.endswith(".npz") and f[:-4] not in seen:
                fp, meta = F.load(os.path.join(fdir, f))       # every fingerprint file present is honoured
                sets.append(ExamFingerprints(f[:-4], meta.get("role", "test"), fp, meta, os.path.join(fdir, f)))
                report["undeclared"].append(f[:-4])
    problems = [f"`{n}` has no fingerprint file" for n in report["missing"]] + \
               [f"`{n}` was fingerprinted from another version of its manifest" for n in report["stale"]]
    if problems:
        msg = ("exam fingerprints are incomplete: " + "; ".join(problems) +
               ". Run `python data/fetch_exams.py` and `python data/fingerprint_exams.py` first")
        if not allow_missing:
            raise DataError(msg + " (or pass --allow-missing-exams to build anyway; the build manifest will say so).")
        warn(msg)
        report["warnings"].append(msg + " -- built with --allow-missing-exams: those sets were NOT excluded.")
    for s in sets:
        if s.meta.get("complete") is False:
            msg = (f"`{s.name}` was fingerprinted from an incomplete copy ({s.meta.get('missing_count')} file(s) "
                   f"missing): content of those files is not excluded")
            warn(msg)
            report["warnings"].append(msg)
        report.setdefault("sets", {})[s.name] = dict(
            role=s.role, fingerprints=int(len(s)), fp_sha256=fp_digest(s.fp),
            manifest_sha256=s.meta.get("manifest_sha256"), algorithm=s.meta.get("algorithm"),
            files=s.meta.get("files"), bytes=s.meta.get("bytes"), complete=s.meta.get("complete", True))
    return sets, report


def fp_digest(fp):
    """sha256 of a fingerprint array's content (independent of how the .npz file was compressed)."""
    return sha256_bytes(np.ascontiguousarray(np.asarray(fp, dtype="<u8")).tobytes())


class ContentFilter:
    """Checks and cleans candidate bytes against every exam set; keeps per-set counts for the build manifest."""

    def __init__(self, sets):
        self.F = paths.fingerprint_module()
        self.sets = list(sets)
        self.stats = {s.name: dict(files_dropped=0, bytes_dropped=0, files_trimmed=0, bytes_removed=0) for s in sets}

    # ------------------------------------------------------------------ detection
    def shared(self, data):
        """{set name: sorted unique shared fingerprints} for the sets that share at least one block with `data`."""
        if not self.sets or len(data) < self.F.BLOCK:
            return {}
        fps = self.F.fingerprints(data)
        out = {}
        for s in self.sets:
            hit = s.shared(fps)
            if len(hit):
                out[s.name] = hit
        return out

    def covered(self, data, shared, T, S, chunk=1 << 16):
        """bool[len(data)]: bytes inside a window that holds a shared block (windows are checked `chunk` at a time,
        so memory stays small on a whole corpus)."""
        a = np.frombuffer(bytes(data), dtype=np.uint8)
        n = len(a)
        if n < self.F.BLOCK:
            return np.zeros(n, bool)
        allfp = np.unique(np.concatenate(list(shared.values())))
        if n <= T:
            starts, width = np.array([0]), n
        else:
            starts, width = np.arange(0, n - T + 1, S), T
            if starts[-1] != n - T:
                starts = np.append(starts, n - T)
        view = np.lib.stride_tricks.sliding_window_view(a, width)
        delta = np.zeros(n + 1, dtype=np.int64)
        for lo in range(0, len(starts), chunk):
            part = starts[lo:lo + chunk]
            hits = self.F.window_hits(view[part], allfp)
            for s in part[hits]:
                delta[s] += 1
                delta[min(n, s + width)] -= 1
        return np.cumsum(delta[:-1]) > 0

    # ------------------------------------------------------------------ text
    @staticmethod
    def _whole_lines(data, cov, margin=TEXT_MARGIN):
        """bool mask: every line within `margin` bytes of a covered byte (the cut is always made of whole lines; the
        margin takes the unanchored ends of a shared region with it)."""
        cut = np.zeros(len(data), dtype=bool)
        idx = np.flatnonzero(cov)
        if len(idx) == 0:
            return cut
        breaks = np.flatnonzero(np.diff(idx) > 1)
        firsts = np.concatenate([[idx[0]], idx[breaks + 1]])
        lasts = np.concatenate([idx[breaks], [idx[-1]]])
        for a, z in zip(firsts, lasts):
            a, z = max(0, int(a) - margin), min(len(data) - 1, int(z) + margin)
            lo = data.rfind(b"\n", 0, a) + 1
            hi = data.find(b"\n", z)
            cut[lo:len(data) if hi < 0 else hi + 1] = True
        return cut

    def clean_text(self, data, min_bytes=64, max_cut=0.5, rounds=3):
        """(clean bytes, or None when the file is dropped; {set: bytes cut}). Whole lines around shared blocks are cut
        out, then the result is checked again (a cut can join two lines into a new shared block). A file that loses
        more than `max_cut` of its bytes, or keeps fewer than `min_bytes`, is dropped whole."""
        original, cut_by = len(data), {}
        for _ in range(rounds):
            sh = self.shared(data)
            if not sh:
                for name, n in cut_by.items():
                    self.stats[name]["files_trimmed"] += 1
                    self.stats[name]["bytes_removed"] += n
                return data, cut_by
            total = np.zeros(len(data), dtype=bool)
            for name, fps in sh.items():
                cut = self._whole_lines(data, self.covered(data, {name: fps}, TEXT_WINDOW, TEXT_STRIDE))
                cut_by[name] = cut_by.get(name, 0) + int(cut.sum())
                total |= cut
            data = np.frombuffer(data, dtype=np.uint8)[~total].tobytes()
            if len(data) < min_bytes or len(data) < (1 - max_cut) * original:
                break
        for name in cut_by:
            self.stats[name]["files_dropped"] += 1
            self.stats[name]["bytes_dropped"] += original
        return None, cut_by

    # ------------------------------------------------------------------ binary
    def clean_span(self, data, span_max, min_span, pick, anchor=None):
        """(start, end) of ONE contiguous span free of shared blocks, or None (the whole file is dropped).

        A binary is never spliced. `pick` (int) chooses deterministically among the three longest clean intervals of at
        least `min_span` bytes and the start inside the chosen one; with `anchor` (bytes) the span starts at an
        occurrence of it (the symbol-table group)."""
        sh = self.shared(data)
        n = len(data)
        if not sh:
            intervals = [(0, n)]
        else:
            cov = np.zeros(n, dtype=bool)
            for name, fps in sh.items():
                c = self.covered(data, {name: fps}, BIN_WINDOW, BIN_STRIDE)
                self.stats[name]["bytes_removed"] += int(c.sum())
                cov |= c
            edges = np.flatnonzero(np.diff(np.concatenate([[1], cov.astype(np.int8), [1]])))
            intervals = [(int(edges[k]), int(edges[k + 1])) for k in range(0, len(edges), 2)]
        good = sorted((iv for iv in intervals if iv[1] - iv[0] >= min_span), key=lambda iv: (iv[0] - iv[1], iv[0]))
        span = None
        if anchor is not None:
            for a, z in good:
                hits, p = [], data.find(anchor, a, z)
                while p >= 0 and z - p >= min_span:
                    hits.append(p)
                    p = data.find(anchor, p + 1, z)
                if hits:
                    s = hits[pick % max(1, len(hits) - 200)]
                    span = (s, min(z, s + span_max))
                    break
        elif good:
            a, z = good[pick % min(3, len(good))]
            length = min(span_max, z - a)
            start = a + (pick % (z - a - length + 1))
            span = (start, start + length)
        for name in sh:
            if span is None:
                self.stats[name]["files_dropped"] += 1
                self.stats[name]["bytes_dropped"] += n
            else:
                self.stats[name]["files_trimmed"] += 1
        return span

    # ------------------------------------------------------------------ whole corpus
    def verify_regions(self, data, T=BIN_WINDOW, S=BIN_STRIDE):
        """[(start, end)] regions of `data` holding a shared block (empty list: clean)."""
        sh = self.shared(data)
        if not sh:
            return []
        cov = self.covered(data, sh, T, S)
        edges = np.flatnonzero(np.diff(np.concatenate([[0], cov.astype(np.int8), [0]])))
        return [(int(edges[k]), int(edges[k + 1])) for k in range(0, len(edges), 2)]

    def report(self):
        return {name: dict(v) for name, v in sorted(self.stats.items())}
