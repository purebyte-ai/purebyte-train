"""Tail guard (experimental).

The problem: about one training seed in six of secrets-bin ends with many times the false alarms of the others on real
binaries; warmup, clipping and quantization tweaks only moved the problem to another seed. Every `guard_every` steps, in eval mode and without
gradients, the model reads N fixed REAL negative windows (`guard_windows`) and the proxy of its false alarms is how many
it FLAGS, decoded like the factory (document gate + Viterbi at bias GUARD_BIAS; a window is flagged when it yields a span).
    select    the run is NOT touched (same batches, same updates, no RNG); at the end the weights of the guard point with
              the lowest proxy in the second half (ties -> the latest) come back.
    rollback  when a point's proxy exceeds guard_k x max(lowest proxy seen, 1), weights AND optimizer state go back to
              the last good point and the run continues with the NEXT batch (nothing replayed; the schedule keeps its step).
Snapshots live on the CPU (VRAM rule). The curve is left in `m.guard_log`; the checkpoint keeps it whole, and the
`finetune` stage's done.json records its mode, chosen step and rollbacks per seed.

Evidence so far: the proxy flagged none of its windows even on a bad seed (whose false alarms came from one unusual
file): it is expected to be blind. Keep it OFF unless an evaluation shows otherwise.
"""
import json
import math
import time

import numpy as np
import torch
import torch.nn.functional as F

from ..model.decode import label_bias, spans, viterbi
from ..runtime import synchronize
from .optim import cpu_clone
from .recipe import GUARD_MODES

GUARD_BIAS = 0.0        # the default operating point: the engine's without --bias
GUARD_BS = 32           # eval batch (tiny_n18 at 64 on CPU peaked at 7.7 GB)
GUARD_K = 3.0


def setup(path, every, mode, k, steps, T, real):
    """Loads and checks the guard. None when off; raises on a half-declared one BEFORE anything is born. No RNG."""
    if mode is None:
        if path is not None or every or k != GUARD_K:
            raise ValueError("`guard_windows` / `guard_every` / `guard_k` without `guard_mode`: they would be declared "
                             "with no effect")
        return None
    if mode not in GUARD_MODES:
        raise ValueError(f"`guard_mode: {mode!r}`; the valid ones are {GUARD_MODES}")
    if path is None:
        raise ValueError(f"`guard_mode: {mode}` needs `guard_windows` (real negative windows)")
    if isinstance(every, bool) or not isinstance(every, (int, float)) or every != int(every) or int(every) < 1:
        raise ValueError(f"`guard_every: {every!r}` must be a whole number of steps >= 1")
    every = int(every)
    if mode != "rollback" and k != GUARD_K:
        raise ValueError("`guard_k` only acts with `guard_mode: rollback`")
    if isinstance(k, bool) or not isinstance(k, (int, float)) or not math.isfinite(k) or k < 1.0:
        raise ValueError(f"`guard_k: {k!r}` must be a finite number >= 1")
    points = list(range(every, steps + 1, every))
    if not points or points[-1] != steps:
        points.append(steps)                  # the end is always judged: it is the answer without the guard
    late = [t for t in points if 2 * t >= steps]
    if mode == "select" and len(late) < 2:
        raise ValueError(f"`guard_every: {every}` with {steps} steps leaves {len(late)} point(s) in the second half: "
                         f"`select` would have nothing to choose from")
    if mode == "rollback" and len(points) < 2:
        raise ValueError(f"`guard_every: {every}` with {steps} steps judges the run once: `rollback` needs 2 points")
    with np.load(path) as z:
        X = z["X"]
        src = np.asarray(z["src"]) if "src" in z.files else None
        names = json.loads(str(z["src_names"])) if "src_names" in z.files else None
    if X.dtype != np.uint8 or X.ndim != 2 or X.shape[1] != T or len(X) < 1:
        raise ValueError(f"{path}: `X` is {X.dtype} {X.shape}; the guard needs uint8 [N >= 1, {T}]")
    if src is None or names is None:
        src, names = np.zeros(len(X), np.uint8), ["all"]
    elif src.shape != (len(X),) or int(src.max()) >= len(names):
        raise ValueError(f"{path}: `src` {src.shape} / `src_names` {names} do not describe its {len(X)} windows")
    if real is not None:
        mine = {r.tobytes() for r in X}
        both = sum(r.tobytes() in mine for r in real["X"])
        if both:
            raise ValueError(f"{path}: {both} guard window(s) are also in `real_windows`: the model TRAINS on them, the "
                             f"proxy would measure the training set")
    return dict(X=X, src=src, names=list(names), mode=mode, every=every, k=float(k), steps=int(steps),
                points=set(points), path=str(path), log=[], best=None, good=None, low=None, rollbacks=[], secs=0.0)


@torch.no_grad()
def proxy(m, th, ch, G, device, n_entities=1):
    """What the model flags among the guard windows. Eval mode, no gradient, no RNG; modes restored afterwards."""
    mods = (m, th, ch)
    was = [x.training for x in mods]
    for x in mods:
        x.eval()
    lb = label_bias(n_entities, GUARD_BIAS)
    X = G["X"]
    doc, tag, fire, pdoc = (np.zeros(len(X), bool), np.zeros(len(X), bool), np.zeros(len(X), bool),
                            np.zeros(len(X), np.float64))
    try:
        for i in range(0, len(X), GUARD_BS):
            x = torch.from_numpy(X[i:i + GUARD_BS].astype(np.int64)).to(device)
            h, _ = m(x)
            lp = F.log_softmax(th(h).float(), -1).cpu().numpy()
            dl = ch(h).float()
            d = dl.argmax(-1).cpu().numpy() != 0
            pdoc[i:i + len(d)] = torch.softmax(dl, -1)[:, 1].cpu().double().numpy()   # float64 on the CPU (MPS has none)
            em = lp.astype(np.float64) + lb
            # Viterbi returns all-O where O scores >= every other label at every byte: only other windows can hold a span
            need = (em[:, :, 1:].max(-1) > em[:, :, 0]).any(-1)
            doc[i:i + len(d)], tag[i:i + len(d)] = d, need
            for k in np.flatnonzero(need & d):
                fire[i + k] = bool(spans(viterbi(lp[k], n_entities, lb), n_entities))
    finally:
        for x, w in zip(mods, was):
            x.train(w)
    src = G["src"]
    return dict(fire=int(fire.sum()), doc=int(doc.sum()), tag=int(tag.sum()), doc_mass=round(float(pdoc.sum()), 3),
                fire_by_src={n: int(fire[src == j].sum()) for j, n in enumerate(G["names"]) if fire[src == j].any()})


def restore(snap, trainable, opt):
    """Weights and optimizer state of `snap` back into the run. The lr stays the one the schedule gives NOW."""
    w, o = snap
    with torch.no_grad():
        for p, s in zip(trainable, w):
            p.copy_(s)
    lrs = [g["lr"] for g in opt.param_groups]
    opt.load_state_dict(cpu_clone(o))         # a fresh copy: on the CPU load_state_dict ALIASES what it is given
    for g, lr in zip(opt.param_groups, lrs):
        g["lr"] = lr


def point(G, t, m, th, ch, trainable, opt, device, n_entities=1):
    """The guard after step `t` (1-based, complete): measures, and keeps (select) or restores (rollback) a state."""
    t0 = time.perf_counter()
    synchronize(device)                       # the pending training kernels are not the guard's time
    r = proxy(m, th, ch, G, device, n_entities)
    p, last = r["fire"], t == G["steps"]
    e = dict(step=t, **r, lr=float(opt.param_groups[0]["lr"]))
    if G["mode"] == "select":
        if 2 * t >= G["steps"] and (G["best"] is None or p <= G["best"][1]):     # ties -> the latest
            G["best"] = (t, p, None if last else [q.detach().to("cpu", copy=True) for q in trainable])
    else:
        lim = None if G["low"] is None else G["k"] * max(G["low"], 1)
        e["limit"] = lim
        if lim is None or p <= lim:
            G["low"] = p if G["low"] is None else min(G["low"], p)
            if not last:                      # at the end the run's own state is the good one
                G["good"] = (t, ([q.detach().to("cpu", copy=True) for q in trainable], cpu_clone(opt.state_dict())))
        else:
            restore(G["good"][1], trainable, opt)
            e["back_to"] = G["good"][0]
            G["rollbacks"].append(dict(step=t, fire=p, limit=lim, back_to=G["good"][0]))
    e["secs"] = round(time.perf_counter() - t0, 3)
    G["secs"] += e["secs"]
    G["log"].append(e)


def finish(G, m, trainable, run_secs):
    """`select`: the chosen weights back into the model. Both: the curve into `m.guard_log`."""
    out = dict(mode=G["mode"], every=G["every"], bias=GUARD_BIAS, path=G["path"], windows=int(len(G["X"])),
               windows_by_src={n: int((G["src"] == j).sum()) for j, n in enumerate(G["names"])}, points=G["log"])
    if G["mode"] == "select":
        t, p, snap = G["best"]
        if snap is not None:
            with torch.no_grad():
                for q, s in zip(trainable, snap):
                    q.copy_(s)
        out.update(chosen_step=t, chosen_fire=p)
    else:
        last = G["log"][-1]
        out.update(k=G["k"], rollbacks=G["rollbacks"],
                   returned=("end" if "back_to" not in last else f"state of step {last['back_to']}"))
    out.update(guard_seconds=round(G["secs"], 1), run_seconds=round(run_secs, 1),
               guard_frac=round(G["secs"] / max(1e-9, run_secs - G["secs"]), 4))
    m.guard_log = out


def brief(g):
    """A short form of the guard's curve (the checkpoint keeps the whole log; the factory does not call this)."""
    if not g:
        return None
    out = dict(mode=g["mode"], curve=[[p["step"], p["fire"], p["doc"]] for p in g["points"]],
               guard_frac=g.get("guard_frac"))
    if g["mode"] == "select":
        out.update(chosen_step=g["chosen_step"], chosen_fire=g["chosen_fire"])
    else:
        out.update(rollbacks=[[r["step"], r["back_to"]] for r in g["rollbacks"]], returned=g["returned"])
    return out
