"""The golden training cases: tiny CPU runs that cover every trainer path, and their deterministic inputs.

Each case is a SpanRecipe on minimal shapes (d_model 64, 2 layers, 20 steps). The fingerprint of a case is the sha256 of
every weight of the three modules after training. `golden/training_fingerprints.json` holds the fingerprints produced by
the code base the released specialists were trained with; `test_golden_training.py` requires pbtrain to reproduce them
bit for bit (on the platform and torch build they were recorded with). Changing a case invalidates its fingerprint.
"""
import hashlib
import json
import os

import numpy as np

STEPS = 20
THREADS = 1
T = 64

# test-only presets, registered on top of the product presets (never used outside tests)
TEST_PRESETS = {
    "mini64_la4": ("mini64", dict(lookahead=4)),
    "mini64_g1": ("mini64_n", dict(ngram_gate=True, ngram_layer=1)),
    "mini64_a": ("mini64", dict(attn_layer=1)),
    "mini64_h2": ("mini64_n", dict(ngram_heads=2)),
}


def register_test_presets(presets):
    for k, v in TEST_PRESETS.items():
        presets.setdefault(k, v)


def preset_values(presets, name):
    """The flat ModelConfig overrides of a preset (to register the same shape in another code base)."""
    chain = []
    while name is not None:
        chain.append(presets[name][1])
        name = presets[name][0]
    out = {}
    for part in reversed(chain):
        out.update(part)
    return out


class Task:
    """48 windows of 64 bytes; half carry one span of 4-8 bytes. Fixed by its own seed."""

    def __init__(self, n=48, seed=1234):
        g = np.random.default_rng(seed)
        X = g.integers(0, 256, (n, T), dtype=np.uint8)
        S = np.zeros((n, T), dtype=np.uint8)
        Y = np.zeros(n, dtype=np.int64)
        for i in range(0, n, 2):
            L = int(g.integers(4, 9))
            a = int(g.integers(0, T - L))
            S[i, a:a + L] = 1
            Y[i] = 1
        self.train = (X, Y, S)
        self.select = (X[:16], Y[:16], S[:16])
        self.test = (X[16:32], Y[16:32], S[16:32])
        self.n_classes = 2


def make_files(d):
    """The .npz inputs of the stream / guard cases, deterministic. Returns {name: path}."""
    os.makedirs(d, exist_ok=True)
    g = np.random.default_rng(99)
    paths = {}
    X = g.integers(0, 256, (24, T), dtype=np.uint8)
    paths["neg"] = os.path.join(d, "real_neg.npz")
    np.savez(paths["neg"], X=X)
    S = np.zeros((24, T), dtype=np.uint8)
    for i in range(0, 24, 3):
        a = int(g.integers(0, T - 6))
        S[i, a:a + 6] = 1
    paths["spans"] = os.path.join(d, "real_spans.npz")
    np.savez(paths["spans"], X=X, S=S, Y=S.any(1).astype(np.uint8))
    pt = g.random((24, T, 5)).astype(np.float16)
    pd = g.random((24, 2)).astype(np.float16)
    paths["distill"] = os.path.join(d, "real_distill.npz")
    np.savez(paths["distill"], X=X, P_tag=pt, P_doc=pd)
    G = g.integers(0, 256, (10, T), dtype=np.uint8)
    paths["guard"] = os.path.join(d, "guard.npz")
    np.savez(paths["guard"], X=G, src=np.array([0] * 6 + [1] * 4, np.uint8), src_names=json.dumps(["a", "b"]))
    return paths


# name -> recipe fields (on top of the common ones). `base` / `ngram_init` are filled in with checkpoints built by
# `make_checkpoints`; "{neg}" etc. are replaced by the paths of `make_files`.
COMMON = dict(size="mini64", steps=STEPS, batch=8, lr=1e-3, warmup=5)
CASES = {
    "default": dict(),
    "seed_1": dict(_seed=1),
    "control_shuffled": dict(shuffle_spans=True),
    "ngram_tables": dict(size="mini64_n"),
    "ngram_heads_2": dict(size="mini64_h2"),
    "ngram_gate_layer1": dict(size="mini64_g1"),
    "lookahead_4": dict(size="mini64_la4"),
    "attention_layer1": dict(size="mini64_a"),
    "hydra": dict(direction="hydra"),
    "bimamba": dict(direction="bimamba"),
    "float_control": dict(ternary=False),
    "lr_3e-4_wd_clip": dict(lr=3e-4, wd=0.1, clip=0.5, lam=0.25),
    "out_lr_mult": dict(out_lr_mult=0.25),
    "beta2": dict(beta2=0.997),
    "init_seed": dict(init_seed=7),
    "span_weight": dict(span_weight=20.0),
    "real_neg": dict(real_windows="{neg}", real_frac=0.25, real_mode="neg"),
    "real_spans": dict(real_windows="{spans}", real_frac=0.25, real_mode="spans"),
    "real_spans_control": dict(real_windows="{spans}", real_frac=0.25, real_mode="spans", shuffle_spans=True),
    "real_distill": dict(real_windows="{distill}", real_frac=0.25, real_mode="distill", distill_alpha=0.5),
    "guard_select": dict(guard_windows="{guard}", guard_every=5, guard_mode="select"),
    "guard_rollback": dict(guard_windows="{guard}", guard_every=5, guard_mode="rollback", guard_k=1.0),
    "base_body": dict(base="{base}"),
    "adapter_scales": dict(base="{base}", adapter="scales"),
    "adapter_lora": dict(base="{base}", adapter="lora", lora_rank=4),
    "adapter_heads": dict(base="{base}", adapter="heads"),
    "ngram_init_frozen": dict(size="mini64_n", ngram_init="{ngram}", freeze_ngram=True),
}


def make_checkpoints(d, files):
    """A raw body checkpoint (`base`) and an n-gram checkpoint with its config sidecar (`ngram`), from fresh inits
    of pbtrain at fixed seeds. Adds their paths to `files`."""
    import torch
    from dataclasses import asdict
    from pbtrain.model import ByteModel, ModelConfig
    torch.manual_seed(5)
    body = ByteModel(ModelConfig.preset("mini64"))
    files["base"] = os.path.join(d, "base_body.pt")
    torch.save(body.state_dict(), files["base"])
    torch.manual_seed(6)
    cfg = ModelConfig.preset("mini64_n")
    mem = ByteModel(cfg)
    files["ngram"] = os.path.join(d, "ngram_memory.pt")
    torch.save(mem.state_dict(), files["ngram"])
    with open(files["ngram"] + ".json", "w", encoding="utf-8") as f:
        json.dump({"cfg": asdict(cfg)}, f)
    return files


def resolve(case, files):
    """Case -> (seed, recipe kwargs) with the file placeholders filled in."""
    kw = dict(COMMON)
    kw.update(CASES[case])
    seed = kw.pop("_seed", 0)
    for k, v in list(kw.items()):
        if isinstance(v, str) and v.startswith("{") and v.endswith("}"):
            kw[k] = files[v[1:-1]]
    return seed, kw


def fingerprint(*mods):
    """sha256 over every tensor of the modules' state_dicts, in order, as float32 bytes; plus the key list."""
    h = hashlib.sha256()
    keys = []
    for m in mods:
        for k, t in m.state_dict().items():
            keys.append(k)
            h.update(t.detach().cpu().float().contiguous().numpy().tobytes())
    return h.hexdigest(), keys
