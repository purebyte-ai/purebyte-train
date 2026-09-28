"""A stream of REAL windows appended to every batch.

    neg      hard negatives: the real windows join the batch and its flat CE with an all-O target and class 0.
             Stable: on binaries, real Windows negatives cut false alarms on PE files ~50x. Mine them with a filter for
             anything that could be a real credential (training/docs/pitfalls.md).
    spans    windows labeled by hand per span (`S` 0/1, optional `Y` = S.any(1)): they join the batch with their own
             BIOES targets and class, like one more synthetic example. Stable: secrets-code 1.0.0 trains with it.
    distill  the mean softmax of an ensemble of teacher models on real windows (`P_tag` [N,T,L], `P_doc` [N,2],
             float16), taught by KL(teacher || student) at T = 1 with weight `distill_alpha`. Experimental.

round(batch * real_frac) real windows are APPENDED to the batch (0.25 x 16 -> 4). The real stream draws from its OWN
generators, so the synthetic windows of every step are exactly those of the same run without it. The shuffled-spans
control shuffles the real targets between the real windows (same inputs, no label), like the synthetic spans.
"""
import numpy as np
import torch
import torch.nn.functional as F

from ..model.decode import mask_to_bioes
from .recipe import REAL_MODES

REAL_SEED = 0x5EA1


def load_stream(path, frac, mode, alpha, bs, T, n_lab, seed, shuffle, n_entities=1):
    """Loads and checks the real stream. None when off (`path is None`); raises on a half-declared one."""
    if path is None:
        if frac or mode is not None or alpha != 1.0:
            raise ValueError("`real_frac` / `real_mode` / `distill_alpha` without `real_windows`: they would be "
                             "declared with no effect")
        return None
    if mode not in REAL_MODES:
        raise ValueError(f"`real_mode: {mode!r}`; the valid ones are {REAL_MODES}")
    if mode in ("neg", "spans") and alpha != 1.0:
        raise ValueError("`distill_alpha` only acts with `real_mode: distill`")
    n = int(round(bs * float(frac)))
    if n < 1:
        raise ValueError(f"`real_frac: {frac}` with batch {bs} gives {n} real windows per step")
    with np.load(path) as z:
        X = z["X"]
        Pt = z["P_tag"] if (mode == "distill" and "P_tag" in z) else None
        Pd = z["P_doc"] if (mode == "distill" and "P_doc" in z) else None
        Sr = z["S"] if (mode == "spans" and "S" in z) else None
        Yr = z["Y"] if (mode == "spans" and "Y" in z) else None
    if X.dtype != np.uint8 or X.ndim != 2 or X.shape[1] != T:
        raise ValueError(f"{path}: `X` is {X.dtype} {X.shape}; the task needs uint8 [N, {T}]")
    if len(X) < n:
        raise ValueError(f"{path} holds {len(X)} windows and each step takes {n}")
    r = dict(mode=mode, n=n, X=X, P_tag=None, P_doc=None, perm=None)
    if mode == "spans":
        if Sr is None:
            raise ValueError(f"{path} has no `S`: it is not a set of labeled real windows")
        if Sr.shape != X.shape or not (np.asarray(Sr) <= 1).all() or (np.asarray(Sr) < 0).any():
            raise ValueError(f"{path}: `S` {Sr.dtype} {Sr.shape} must be a 0/1 span mask shaped like `X` {X.shape}")
        has = np.asarray(Sr).any(axis=1)
        if Yr is not None and (np.shape(Yr) != (len(X),) or not np.array_equal(np.asarray(Yr).astype(bool), has)):
            raise ValueError(f"{path}: `Y` must be exactly «the window holds a span» (S.any(1)), one per window")
        r.update(tag=np.stack([mask_to_bioes(s, n_entities) for s in np.asarray(Sr)]), doc=has.astype(np.int64),
                 perm=(np.random.default_rng([seed, REAL_SEED + 2]).permutation(len(X)) if shuffle
                       else np.arange(len(X))))
        if int(r["tag"].max()) >= n_lab:
            raise ValueError(f"{path}: span labels up to {int(r['tag'].max())} with {n_lab} labels")
    if mode == "distill":
        if Pt is None or Pd is None:
            raise ValueError(f"{path} has no `P_tag`/`P_doc`: it is not a distillation set")
        if Pt.shape != (len(X), T, n_lab) or Pd.shape != (len(X), 2):
            raise ValueError(f"{path}: P_tag {Pt.shape} / P_doc {Pd.shape}; expected ({len(X)}, {T}, {n_lab}) / "
                             f"({len(X)}, 2)")
        if not (np.isfinite(Pt).all() and np.isfinite(Pd).all() and (Pt >= 0).all() and (Pd >= 0).all()):
            raise ValueError(f"{path}: the teacher's probabilities are not finite and non-negative")
        r.update(P_tag=Pt, P_doc=Pd,
                 perm=(np.random.default_rng([seed, REAL_SEED + 1]).permutation(len(X)) if shuffle
                       else np.arange(len(X))))
    return r


def distill_loss(th, ch, hr, real, j, lam, device):
    """KL(teacher || student) at T = 1: averaged per byte (span head) and per window (document head)."""
    k = real["perm"][j]
    L = real["P_tag"].shape[-1]
    pt = torch.from_numpy(real["P_tag"][k].astype(np.float32)).to(device)
    pd = torch.from_numpy(real["P_doc"][k].astype(np.float32)).to(device)
    pt = pt / pt.sum(-1, keepdim=True)            # stored in float16: rows add up to 1 +- 1e-3
    pd = pd / pd.sum(-1, keepdim=True)
    kl_tag = F.kl_div(F.log_softmax(th(hr).float(), -1).reshape(-1, L), pt.reshape(-1, L), reduction="batchmean")
    kl_doc = F.kl_div(F.log_softmax(ch(hr).float(), -1), pd, reduction="batchmean")
    return kl_tag + lam * kl_doc
