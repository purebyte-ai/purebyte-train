"""Windows of REAL context bytes with SYNTHETIC credential lines injected: the `secrets_synthetic` data.

For every window: a real stretch of the context corpus (source code, configuration or binaries), cut on a line boundary;
one generated line (positive or negative at 50 %) inserted there; optionally a block of extra lines around it (half of
the windows when `extra_lines` > 0; in a negative window every extra line is a trap, in a positive one 50 % are
positives, so the window prior stays 0.50); half of the negative neighbours replaced by REAL lines of the corpus
(`real_neighbors`); for binaries, blocks embedded between NULs like a C string table (`nul_separator`) and, for a
single line, sometimes the value ALONE when it is decidable by itself (`bare_value`: provider prefix, connection string,
PEM). The span mask marks the value bytes of every positive line. Train, select and test are cut from DISJOINT byte
ranges of the corpus, with a guard equal to the bytes a window actually reads.

Behaviour locked by training/tests/test_secrets_data_golden.py (same corpus + catalogues -> same arrays, bit for bit).
"""
import re

import numpy as np

from . import credentials as _cred

# what is decided BY THE VALUE, with no name next to it: provider prefix, URI with a password, PEM
_DECIDABLE = re.compile(r"^(ghp_|gho_|ghs_|github_pat_|glpat-|sk_live_|rk_live_|whsec_|AKIA|ASIA|AIza|"
                        r"xox[bp]-|SG\.|sk-proj-|sk-ant-|npm_|dop_v1_|shpat_|hvs\.|dckr_pat_|pypi-|"
                        r"-----BEGIN|[a-z][a-z0-9+]{2,12}://[^\s/@:]+:[^\s/@]+@)")
GENERATOR_VERSIONS = ("v3", "v4")         # the same generator; the label records which data a model saw


def _generator(version):
    if version not in GENERATOR_VERSIONS:
        raise ValueError(f"unknown generator `{version}`; available: {GENERATOR_VERSIONS}")
    return _cred.line


def build(corpus, n_examples, seq_len, rng, frac=(0.0, 1.0), version="v4", extra_lines=0, nul_separator=False,
          real_neighbors=False, bare_value=False):
    """-> (X uint8 [N, T], Y int64 [N], S uint8 [N, T]). `frac` bounds the span of the corpus the windows come from."""
    data = corpus.data
    span = corpus.tr_hi - corpus.tr_lo
    lo = corpus.tr_lo + int(span * frac[0])
    hi = corpus.tr_lo + int(span * frac[1]) - seq_len * 2 - 1        # the guard is what a window READS
    X = np.zeros((n_examples, seq_len), dtype=np.uint8)
    Y = np.zeros(n_examples, dtype=np.int64)
    S = np.zeros((n_examples, seq_len), dtype=np.uint8)
    gen0 = _generator(version)
    if _cred.HOLDOUT:
        def gen(rng_, pos_):
            for _ in range(200):
                r = gen0(rng_, pos_)
                if not _cred.contains_held_out(r[0]):
                    return r
            raise RuntimeError("200 lines in a row with a held-out provider: impossible")
    else:
        gen = gen0
    for i in range(n_examples):
        pos = bool(rng.integers(0, 2))
        win = np.asarray(data[int(rng.integers(lo, hi)):][:seq_len * 2])
        nl = np.flatnonzero(win == 10)
        cut = int(nl[len(nl) // 3]) + 1 if len(nl) > 3 else seq_len // 2
        line, (a, b) = gen(rng, pos)
        lb = (line + "\n").encode()
        spans = [(a, b)] if pos else []
        has_real = False
        if extra_lines and rng.random() < 0.5:
            for _ in range(int(rng.integers(1, extra_lines + 1))):
                pos2 = bool(rng.integers(0, 2)) and pos
                l2, (a2, b2) = gen(rng, pos2)
                lb2 = (l2 + "\n").encode()
                # a PEM block is itself a run of base64-looking lines: with `no_pem`, a block that carries one gets
                # no real neighbours (real neighbours next to a PEM taught that its lines are harmless)
                with_pem = b"-----BEGIN" in lb or b"-----BEGIN" in lb2
                if real_neighbors == "no_pem" and with_pem and has_real:
                    continue
                if real_neighbors and not pos2 and rng.random() < 0.5 \
                        and not (real_neighbors == "no_pem" and with_pem):
                    cand = [x for x in bytes(win[seq_len:]).split(b"\n") if 8 < len(x) < 100]
                    if cand:
                        lb2 = cand[int(rng.integers(0, len(cand)))] + b"\n"
                        has_real = True
                if len(lb) + len(lb2) > seq_len - 64:          # the whole block must fit
                    break
                if rng.integers(0, 2):                         # behind
                    if pos2:
                        spans.append((len(lb) + a2, len(lb) + b2))
                    lb = lb + lb2
                else:                                          # in front
                    spans = [(x + len(lb2), y + len(lb2)) for x, y in spans]
                    if pos2:
                        spans.append((a2, b2))
                    lb = lb2 + lb
        if bare_value and len(spans) <= 1 and b"\n" not in lb[:-1] and rng.random() < 0.25:
            txt = lb[:-1].decode("utf8", "replace")
            va, vb = (spans[0] if spans else (a, b))
            value = txt[va:vb]
            if (not spans) or _DECIDABLE.search(value):
                lb = (value + "\n").encode()
                spans = [(0, len(value.encode()))] if spans else []
        if nul_separator and rng.random() < 0.7:
            lb = b"\x00" + lb[:-1].replace(b"\n", b"\x00") + b"\x00"
            spans = [(x + 1, y + 1) for x, y in spans]
        merged = np.concatenate([win[:cut], np.frombuffer(lb, dtype=np.uint8), win[cut:]])
        # the line must fall ENTIRELY inside the window, or the positive is not learnable
        margin = min(cut, max(0, seq_len - len(lb) - 8))
        start = max(0, cut - int(rng.integers(margin // 4, margin + 1))) if margin else 0
        seg = merged[start:start + seq_len]
        if len(seg) < seq_len:
            seg = np.pad(seg, (0, seq_len - len(seg)), constant_values=32)
        X[i] = seg
        Y[i] = int(pos)
        for a_, b_ in spans:
            vs, ve = cut - start + a_, cut - start + b_
            if 0 <= vs < seq_len:
                S[i, vs:min(ve, seq_len)] = 1
    return X, Y, S


class SecretsTask:
    """Disjoint train / select / test (seeds seed, seed + 1000, seed + 2000 over disjoint corpus ranges)."""

    def __init__(self, corpus, n_train, n_eval, seq_len, seed, version="v4", fracs=(0.80, 0.10, 0.10), extra_lines=0,
                 nul_separator=False, real_neighbors=False, bare_value=False, mined_negatives=None):
        tr, se, te = (float(x) for x in fracs)
        if abs(tr + se + te - 1.0) > 1e-9 or min(tr, se, te) <= 0:
            raise ValueError(f"split fractions {fracs} must be positive and add up to 1")
        kw = dict(version=version, extra_lines=int(extra_lines), nul_separator=bool(nul_separator),
                  real_neighbors=real_neighbors, bare_value=bool(bare_value))
        self.train = build(corpus, n_train, seq_len, np.random.default_rng(seed), (0.0, tr), **kw)
        self.select = build(corpus, n_eval, seq_len, np.random.default_rng(seed + 1000), (tr, tr + se), **kw)
        self.test = build(corpus, n_eval, seq_len, np.random.default_rng(seed + 2000), (tr + se, 1.0), **kw)
        # MINED NEGATIVES: real windows a released model flagged falsely replace a share of the NEGATIVE windows of
        # train (select and test untouched); without the key, not one random number is drawn.
        self.mined_used = 0
        if mined_negatives:
            path, prop = mined_negatives
            with np.load(path) as z:
                M = z["X"]
            if M.shape[1] != seq_len:
                raise ValueError(f"the mined windows are {M.shape[1]} bytes and the task {seq_len}")
            r = np.random.default_rng(seed + 77)
            neg = np.where(self.train[1] == 0)[0]
            k = min(int(len(neg) * float(prop)), len(neg))
            where = r.choice(neg, size=k, replace=False)
            which = r.integers(0, len(M), size=k)
            self.train[0][where] = M[which]
            self.train[2][where] = 0
            self.mined_used = int(k)
