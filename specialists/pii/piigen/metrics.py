"""Scores of a span predictor against gold spans, per type and for the redaction product.

Three readings, reported side by side (none replaces the others):
  exact      a predicted span counts only with the SAME start, end and type: a model that cuts entities is of no use
             for redaction. The primary reading where the labelling policy is this specialist's own.
  overlap    a prediction of type t is right if it overlaps a gold span of type t; a prediction that only touches an
             IGNORED region (a label outside the taxonomy) or a gold span of another type counts neither way. The
             primary reading on public datasets, whose boundary conventions differ.
  redaction  the product view, TYPE-AGNOSTIC: a gold entity LEAKS if less than 80 % of its letters and digits end up
             inside some predicted span of any type; `over_bytes_per_kb` = predicted bytes outside every gold or
             ignored span, per KB of text.
Spans are [(start, end, type index)] in bytes; `ignore` is [(start, end)].
"""
from collections import defaultdict
import numpy as np
from .types import TYPES

def _ov(a, b, x, y):
    return a < y and x < b

def score_doc(pred, gold, ignore=(), text=None, acc=None, k=len(TYPES)):
    acc = acc if acc is not None else new_acc()
    (P, G) = (set(map(tuple, pred)), set(map(tuple, gold)))
    for t in range(k):
        pt = {p for p in P if p[2] == t}
        gt = {g for g in G if g[2] == t}
        tp = len(pt & gt)
        c = acc['exact'][t]
        c[0] += tp
        c[1] += len(pt) - tp
        c[2] += len(gt) - tp
        o = acc['overlap'][t]
        for p in pt:
            if any((_ov(p[0], p[1], g[0], g[1]) for g in gt)):
                o[0] += 1
            elif any((_ov(p[0], p[1], a, b) for (a, b) in ignore)) or any((_ov(p[0], p[1], g[0], g[1]) for g in G)):
                o[3] += 1
            else:
                o[1] += 1
        o[2] += sum((1 for g in gt if not any((_ov(p[0], p[1], g[0], g[1]) for p in pt))))
        o[4] += len(gt)
    if text is not None:
        n = len(text)
        cov = np.zeros(n + 1, dtype=bool)
        for (a, b, _) in P:
            cov[max(0, a):min(n, b)] = True
        alnum = np.frombuffer(bytes(text), dtype=np.uint8)
        alnum = (alnum >= 48) & (alnum <= 57) | (alnum >= 65) & (alnum <= 90) | (alnum >= 97) & (alnum <= 122) | (alnum >= 128)
        for (a, b, t) in G:
            m = alnum[a:b]
            need = int(m.sum()) or b - a
            got = int((cov[a:b] & m).sum()) if m.any() else int(cov[a:b].sum())
            r = acc['redaction'][t]
            r[0] += 1
            r[1] += int(got < 0.8 * need)
            r[2] += int(got == 0)
        gold_mask = np.zeros(n, dtype=bool)
        for (a, b, _) in G:
            gold_mask[a:b] = True
        for (a, b) in ignore:
            gold_mask[a:b] = True
        acc['over_bytes'] += int((cov[:n] & ~gold_mask).sum())
        acc['bytes'] += n
        acc['docs'] += 1
    return acc

def new_acc():
    return {'exact': defaultdict(lambda : [0, 0, 0]), 'overlap': defaultdict(lambda : [0, 0, 0, 0, 0]), 'redaction': defaultdict(lambda : [0, 0, 0]), 'over_bytes': 0, 'bytes': 0, 'docs': 0}

def _prf(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (p, r, 2 * p * r / (p + r) if p + r else 0.0)

def summary(acc, min_gold=0, types=TYPES):
    (out, E, O) = ({}, [0, 0, 0], [0, 0, 0])
    for (t, name) in enumerate(types):
        (tp, fp, fn) = acc['exact'][t]
        (otp, ofp, ofn, oign, ng) = acc['overlap'][t]
        (p, r, f) = _prf(tp, fp, fn)
        (op, orr, of) = _prf(otp, ofp, ofn)
        orr = (ng - ofn) / ng if ng else 0.0
        of = 2 * op * orr / (op + orr) if op + orr else 0.0
        rd = acc['redaction'][t]
        out[name] = dict(gold=tp + fn, pred=tp + fp, exact_p=round(p, 4), exact_r=round(r, 4), exact_f1=round(f, 4), overlap_p=round(op, 4), overlap_r=round(orr, 4), overlap_f1=round(of, 4), pred_on_ignored=oign, leak_rate=round(rd[1] / rd[0], 4) if rd[0] else None, evaluable=tp + fn >= min_gold)
        E[0] += tp
        E[1] += fp
        E[2] += fn
        O[0] += otp
        O[1] += ofp
        O[2] += ofn
    (p, r, f) = _prf(*E)
    out['_micro'] = dict(exact_p=round(p, 4), exact_r=round(r, 4), exact_f1=round(f, 4))
    tot = sum((v[0] for v in acc['redaction'].values()))
    leak = sum((v[1] for v in acc['redaction'].values()))
    full = sum((v[2] for v in acc['redaction'].values()))
    if acc['bytes']:
        out['_redaction'] = dict(entities=tot, leak_rate=round(leak / tot, 4) if tot else None, fully_missed=round(full / tot, 4) if tot else None, over_bytes_per_kb=round(1024 * acc['over_bytes'] / acc['bytes'], 3), docs=acc['docs'], kb=round(acc['bytes'] / 1024, 1))
    return out
