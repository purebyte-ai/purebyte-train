"""Python vs the runtime on the same windows: the spans the product would report must be IDENTICAL.

Comparing probabilities is not enough: a different tie-break in Viterbi changes the list of spans while barely moving the
numbers. Runtime = the engine's `model_dump` tool when PUREBYTE_MODEL_DUMP points to it (with `--threads=N` when the
task sets `engine.threads`), else the NumPy reference in reference/ (slow: keep the window count small; it runs in
this process, with pbtrain's thread cap).

Which windows (`pick_windows`): the factory compares windows where Python finds a span AND windows where it finds none,
so that a model that emits nothing cannot pass the comparison vacuously.

A file with 4-bit n-gram tables (`purebyte.ngram.bits = 4`) is compared with the checkpoint whose tables went through
the same quantization and back (what the file computes); the windows where the quantization alone changes the spans
are counted apart (`int4_changed_windows`), as information.
"""
import json
import os
import struct
import subprocess
import sys
import tempfile

import numpy as np
import torch
import torch.nn.functional as F

from .. import paths
from ..model.decode import label_bias, spans, viterbi
from ..train import checkpoint
from .chain import StageError


def _decoder(ck):
    """row -> the spans Python reports for it, with the checkpoint's own operating point (per-type vector when it has
    one). The model is built once."""
    m, th, ch = checkpoint.rebuild(ck)
    m.eval(); th.eval(); ch.eval()
    k = (th.proj.out_features - 1) // 4
    vec = ck.get("bias_vec")
    lb = label_bias(k, 0.0, vec) if vec is not None else label_bias(k, float(ck["bias"]))

    @torch.no_grad()
    def decode(row):
        h, _ = m(torch.from_numpy(row.astype(np.int64))[None])
        if int(F.softmax(ch(h).float(), -1)[0].argmax()) == 0:
            return []
        lp = F.log_softmax(th(h).float(), -1)[0].numpy()
        return [tuple(int(v) for v in s) for s in spans(viterbi(lp, k, lb), k)]
    return decode


def python_spans(ck, X):
    """What Python reports with the checkpoint's own operating point (per-type vector when it has one)."""
    decode = _decoder(ck)
    return [decode(row) for row in X]


def pick_windows(ck, X, n):
    """Indices of up to `n` windows of X, in order: ceil(n/2) where Python finds a span, then windows where it finds
    none, topped up from either kind when one runs short. Python's spans are computed only until both quotas are met."""
    decode = _decoder(ck)
    want_pos = (n + 1) // 2
    pos, neg = [], []
    for i in range(len(X)):
        if len(pos) >= want_pos and len(neg) >= n - want_pos:
            break
        (pos if decode(X[i]) else neg).append(i)
    take = pos[:want_pos] + neg[:n - want_pos]
    rest = [i for i in pos[want_pos:] + neg[n - want_pos:] if i not in take]
    return sorted(take + rest[:max(0, n - len(take))])


def _engine_spans(tool, gguf, X, threads=None):
    with tempfile.TemporaryDirectory() as td:
        inp = os.path.join(td, "inputs.bin")
        with open(inp, "wb") as f:
            f.write(struct.pack("<I", len(X)))
            for row in X:
                f.write(struct.pack("<I", len(row)) + bytes(row))
        cmd = [tool, gguf, inp, "--whole"] + ([f"--threads={int(threads)}"] if threads else [])
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise StageError(f"the engine's model_dump ({tool}) could not run: {e}")
        if r.returncode != 0:
            raise StageError(f"the engine's model_dump failed: {r.stderr[-400:]}")
        res = json.loads(r.stdout)
    out = []
    for item in res["inputs"]:
        w = item["windows"][0] if item["windows"] else {"heads": [None]}
        head = w["heads"][0]
        out.append([] if head is None else [tuple(int(v) for v in s[:3]) for s in head["spans"]])
    return out


def _reference_spans(gguf, X):
    ref = paths.require_source_tree(paths.REFERENCE_SRC)
    if ref not in sys.path:
        sys.path.insert(0, ref)
    from purebyte_ref import Model, scan
    model = Model.open(gguf)
    out = []
    for row in X:
        ws = list(scan(model, bytes(row), whole=True))
        head = ws[0]["heads"][0] if ws else None
        out.append([] if head is None else [tuple(int(v) for v in s[:3]) for s in head["spans"]])
    return out


def compare_runtime(gguf, ck, X, n=None, threads=None):
    """The runtime runs the file with ITS OWN operating point (no override): this also checks that the operating point
    was exported right. `n`: compare `pick_windows(n)` of X instead of every window of X. `threads`: the engine's
    thread count (model_dump only)."""
    tool = os.environ.get("PUREBYTE_MODEL_DUMP")
    runtime = f"engine model_dump ({os.path.basename(tool)})" if tool else "NumPy reference"
    ck0, (ck, bits) = ck, as_exported(ck, gguf)
    idx = pick_windows(ck, X, n) if n is not None else list(range(len(X)))
    Xc = X[idx]
    theirs = _engine_spans(tool, gguf, Xc, threads) if tool else _reference_spans(gguf, Xc)
    ours = python_spans(ck, Xc)
    equal = sum(int(a == b) for a, b in zip(ours, theirs))
    out = dict(n=len(Xc), equal=equal, identical_spans=equal == len(Xc), windows=[int(i) for i in idx],
               operating_bias=float(ck["bias"]), operating_bias_vec=ck.get("bias_vec"), runtime=runtime,
               engine_threads=(int(threads) if threads else 1) if tool else None,
               windows_with_spans=int(sum(1 for s in ours if s)))
    if bits == 4:
        full = python_spans(ck0, Xc)
        out.update(ngram_bits=4, int4_changed_windows=int(sum(int(a != b) for a, b in zip(full, ours))))
    return out


def as_exported(ck, gguf):
    """(the checkpoint as the file computes it, the file's n-gram bits): 4-bit n-gram tables are quantized and
    dequantized exactly as the exporter writes them and the engine reads them back."""
    from ..export.gguf import read_metadata
    from ..export.pbformat import dequant_ngram_int4, quant_ngram_int4
    bits = read_metadata(gguf).get("purebyte.ngram.bits")
    if bits != 4:
        return ck, bits
    model = dict(ck["model"])
    for k, v in ck["model"].items():
        if k.startswith("ngram.tables.") and k.endswith(".weight"):
            packed, scale = quant_ngram_int4(v.detach().cpu().float().numpy())
            model[k] = torch.from_numpy(dequant_ngram_int4(packed, scale))
    return dict(ck, model=model), bits
