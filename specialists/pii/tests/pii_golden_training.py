"""The golden training cases of the pii specialist: tiny CPU runs of the span trainer with ONE entity type on windows
that carry IGNORE bytes (label -100: the look-alikes of the structured-forms variant), and the release size preset
`nano_n15`. The core's own cases (training/tests/golden_cases.py) cover every trainer path on 0/1 masks; these cover
what the pii recipes add. Fingerprint = sha256 of every weight of the three modules after training.
`golden/pii_training_fingerprints.json` holds the fingerprints produced by the code the released model was trained
with; `test_pii_training_golden.py` requires pbtrain to reproduce them bit for bit on the platform and torch build
they were recorded with. Changing a case invalidates its fingerprint.
"""
import numpy as np

STEPS = 20
THREADS = 1
T = 64

COMMON = dict(n_entities=1, steps=STEPS, batch=8, lr=1e-3, warmup=5)
CASES = {
    "one_type_with_ignore": dict(size="mini64"),
    "one_type_with_ignore_control": dict(size="mini64", shuffle_spans=True),
    "one_type_ngram_tables": dict(size="mini64_n"),
    "release_preset_nano_n15": dict(size="nano_n15", steps=6, lr=3e-4),
}


class MaskTask:
    """48 windows of 64 bytes: half carry one span (mask value 1) of 4-8 bytes, a third carry an IGNORE run (255) of
    3-10 bytes that never touches the span. Fixed by its own seed."""

    def __init__(self, n=48, seed=4321):
        g = np.random.default_rng(seed)
        X = g.integers(0, 256, (n, T), dtype=np.uint8)
        S = np.zeros((n, T), dtype=np.uint8)
        Y = np.zeros(n, dtype=np.int64)
        for i in range(n):
            if i % 2 == 0:
                L = int(g.integers(4, 9))
                a = int(g.integers(0, T - L - 12))
                S[i, a:a + L] = 1
                Y[i] = 1
            if i % 3 == 0:
                L = int(g.integers(3, 11))
                a = int(g.integers(T - 12, T - L + 1)) if i % 2 == 0 else int(g.integers(0, T - L))
                S[i, a:a + L] = 255
        self.train = (X, Y, S)
        self.select = (X[:16], Y[:16], S[:16])
        self.test = (X[16:32], Y[16:32], S[16:32])
        self.n_classes = 2


def resolve(case):
    """Case -> (seed, recipe kwargs)."""
    kw = dict(COMMON)
    kw.update(CASES[case])
    return kw.pop("_seed", 0), kw


def fingerprint(*mods):
    """sha256 over every tensor of the modules' state_dicts, in order, as float32 bytes; plus the key list."""
    import hashlib
    h = hashlib.sha256()
    keys = []
    for m in mods:
        for k, t in m.state_dict().items():
            keys.append(k)
            h.update(t.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest(), keys
