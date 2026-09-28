"""Span metrics on labeled windows, decoded exactly as the engine decodes them.

Three readings of the same forward pass (spans compared as SETS with EXACT boundaries; a model that gets single bytes
right but cuts the entities is of no use for redaction):
    tag_only  the tag head alone
    gated     the product rule: spans only in windows where the choice head says positive (the file's `gated_by`)
    oracle    spans only in windows that truly hold one (an upper bound of what the gate can give)
`doc_acc` is the choice head's window accuracy.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..model.decode import label_bias, mask_spans, spans, viterbi, viterbi_batch

READINGS = ("tag_only", "gated", "oracle")


def _prf(tp, fp, fn):
    pr = tp / (tp + fp) if tp + fp else 0.0
    rc = tp / (tp + fn) if tp + fn else 0.0
    return dict(span_f1=round(2 * pr * rc / (pr + rc) if pr + rc else 0.0, 4), precision=round(pr, 4),
                recall=round(rc, 4))


@torch.no_grad()
def forward_windows(m, th, ch, X, *, bs=16, device="cpu"):
    """(log-probs of the tag head [n, T, L] float32, choice-head argmax [n]) in eval mode; modes restored."""
    mods = (m, th, ch)
    was = [x.training for x in mods]
    for x in mods:
        x.eval()
    lps, docs = [], []
    try:
        for i in range(0, len(X), bs):
            x = torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device)
            h, _ = m(x)
            lps.append(F.log_softmax(th(h).float(), -1).cpu().numpy())
            docs.append(ch(h).float().argmax(-1).cpu().numpy())
    finally:
        for x, w in zip(mods, was):
            x.train(w)
    return np.concatenate(lps), np.concatenate(docs)


def score(lp, doc, S, Y, bias, n_entities=1, batched=True):
    """Metrics from cached forward outputs at one operating bias (scalar, or one value per entity type)."""
    per_type = None if np.isscalar(bias) else list(bias)
    lb = label_bias(n_entities, 0.0 if per_type is not None else float(bias), per_type)
    paths = viterbi_batch(lp, n_entities, lb) if batched else np.stack([viterbi(r, n_entities, lb) for r in lp])
    res = {k: [0, 0, 0] for k in READINGS}
    doc_ok = 0
    for j in range(len(lp)):
        gs = set(mask_spans(S[j], n_entities))
        ps = set(spans(paths[j], n_entities))
        pos = bool(doc[j])
        doc_ok += int(pos == bool(Y[j]))
        for name, alive in (("tag_only", True), ("gated", pos), ("oracle", bool(Y[j]))):
            p = ps if alive else set()
            res[name][0] += len(p & gs)
            res[name][1] += len(p - gs)
            res[name][2] += len(gs - p)
    out = {k: _prf(*v) for k, v in res.items()}
    out["doc_acc"] = round(doc_ok / max(1, len(lp)), 4)
    return out


def eval_spans(m, th, ch, X, S, Y, bias, *, n_entities=1, bs=16, device="cpu"):
    """Span metrics at one operating bias."""
    lp, doc = forward_windows(m, th, ch, X, bs=bs, device=device)
    return score(lp, doc, S, Y, bias, n_entities)


def span_f1(pred_spans, gold_spans):
    """(F1, precision, recall) of two span collections, exact boundaries."""
    p, g = set(map(tuple, pred_spans)), set(map(tuple, gold_spans))
    tp = len(p & g)
    prec = tp / len(p) if p else 0.0
    rec = tp / len(g) if g else 0.0
    return (2 * prec * rec / (prec + rec) if prec + rec else 0.0), prec, rec
