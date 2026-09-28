"""Mandatory rivals and the delivery gate.

Hard rule: a specialist that does not beat the classic rival of its task is not delivered. For document labels the rival
is a logistic regression / complement naive Bayes over hashed byte n-grams (the better one rules). For spans the rival
is the rule a competent human would write, which is task-specific: each vertical provides it
(`Vertical.span_rival`), tuned on `select` and reported on `test`, scored exactly like the model.
"""
import time

import numpy as np


def _as_text(X):
    return [bytes(r).decode("latin-1") for r in X]


def classic_baseline(Xtr, Ytr, Xte, Yte, ngram_range=(1, 4), n_features=2 ** 18):
    """Logistic regression and complement naive Bayes over hashed byte n-grams; the better one is the rival."""
    from sklearn.feature_extraction.text import HashingVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.naive_bayes import ComplementNB

    vec = HashingVectorizer(analyzer="char", ngram_range=ngram_range, n_features=n_features, alternate_sign=False,
                            lowercase=False)
    t0 = time.perf_counter()
    A, B = vec.transform(_as_text(Xtr)), vec.transform(_as_text(Xte))
    out = {}
    for name, clf in (("logistic", LogisticRegression(max_iter=500)), ("complement_nb", ComplementNB())):
        t1 = time.perf_counter()
        clf.fit(A, Ytr)
        p = clf.predict_proba(B)
        oh = np.eye(p.shape[1])[Yte]
        out[name] = dict(acc=float((p.argmax(1) == Yte).mean()), brier=float(((p - oh) ** 2).sum(1).mean()),
                         nll=float(-np.log(np.clip(p[np.arange(len(Yte)), Yte], 1e-9, 1)).mean()),
                         fit_s=round(time.perf_counter() - t1, 2))
    best = max(out, key=lambda k: out[k]["acc"])
    return dict(family="classic", ngram_range=list(ngram_range), best=best, acc=out[best]["acc"], detail=out,
                vectorize_s=round(time.perf_counter() - t0, 2))


def chance(Y):
    """Always guessing the majority class."""
    _, c = np.unique(Y, return_counts=True)
    return float(c.max() / c.sum())


def unigram_probe(Xtr, Ytr, Xte, Yte):
    """A trap detector, not a target: if byte unigrams clearly beat chance, the window label is on the surface and the
    model will learn the shortcut. Blocking for document tasks; recorded for span tasks (the span rival decides).
    A train split with one class (every window holds a span, as with documents that are all labeled) has no shortcut
    to probe: that is recorded instead."""
    if len(np.unique(Ytr)) < 2:
        return dict(unigram_acc=None, chance=chance(Yte), suspicious=False,
                    note=f"one class in train ({int(np.unique(Ytr)[0])}): nothing to probe")
    r = classic_baseline(Xtr, Ytr, Xte, Yte, ngram_range=(1, 1), n_features=2 ** 12)
    return dict(unigram_acc=r["acc"], chance=chance(Yte), suspicious=bool(r["acc"] - chance(Yte) > 0.25))


def gate(model_score, rival_score, spread):
    """(passes, reason). The advantage over the rival must be STRICTLY larger than the spread between seeds."""
    d = model_score - rival_score
    if d <= 0:
        return False, f"does not beat the classic rival ({model_score:.4f} <= {rival_score:.4f})"
    if d <= max(spread, 1e-9):
        return False, (f"the advantage ({d:+.4f}) is not larger than the spread between seeds ({spread:.4f}): a "
                       f"tie")
    return True, f"beats the classic rival by {d:+.4f} ({d / max(spread, 1e-9):.1f} spreads)"
