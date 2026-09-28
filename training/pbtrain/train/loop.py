"""THE span trainer: one byte body, two heads.

    tag head     per byte, BIOES over k entity types: WHERE the spans are
    choice head  per window, 2 classes, `max` pooling: WHETHER the window holds one (it gates the tag head)

Loss = CE(tag, flat over every byte) + lam * CE(choice). Flat CE always (no per-byte weighting) unless `span_weight`
> 1 (extraction tasks, where answer bytes are ~1 %; values below 1 are refused by the recipe). AdamW with decay on
matrices only, WSD schedule, clip 1.0, batches without replacement. `train_spans` returns the model at the LAST step (or
the guard's choice). On a GPU it first asks PyTorch for deterministic algorithms (`runtime.gpu_determinism`); a CPU run
never does.

Bit-identity contract: the same (task, recipe, seed) gives the SAME weights on the same platform and torch build, and
the released specialists were trained by this exact sequence of operations (training/tests/test_golden_training.py
locks it with fingerprints). The order of RNG draws is: global seed -> [control permutation (own generator)] -> [real stream / guard / memory checks:
no RNG] -> [init_seed] -> body -> heads -> [adapter: own generator] -> batches (own generator seeded by `seed`); the real
stream draws from its own generators. Keep it.
"""
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..model.body import ByteModel
from ..model.config import ModelConfig
from ..model.decode import IGNORE_LABEL, mask_to_bioes, n_labels
from ..model.heads import ChoiceHead, TagHead
from ..runtime import gpu_determinism
from . import guard as guard_mod
from .init import attach_adapter, check_adapter_args, load_base, load_ngram, ngram_state
from .optim import batches, make_optimizer
from .recipe import SpanRecipe
from .streams import REAL_SEED, distill_loss, load_stream


class SpanTask:
    """The three disjoint splits a vertical builds: (X uint8 [n, T], Y uint8 [n], S uint8 [n, T]) each."""

    def __init__(self, train, select, test):
        self.train, self.select, self.test = train, select, test

    @classmethod
    def from_npz(cls, z):
        """The factory's partitions.npz; only train is needed to train (select / test may be absent)."""
        part = lambda s: (z["X" + s], z["Y" + s], z["S" + s]) if "X" + s in z.files else None
        return cls(part("tr"), part("se"), part("te"))


def _check_memory_flags(recipe, cfg):
    if recipe.freeze_ngram not in (True, False):
        raise ValueError(f"`freeze_ngram: {recipe.freeze_ngram!r}` must be true or false")
    if (recipe.ngram_init is not None or recipe.freeze_ngram) and not cfg.ngram_orders:
        raise ValueError(f"`ngram_init` / `freeze_ngram` with size `{recipe.size}`, which has no n-gram tables: they "
                         f"would be declared with no effect")


def train_spans(task, recipe: SpanRecipe = None, *, seed=0, device="cpu", log=None):
    """Trains one run. Returns (body, tag head, choice head); the tail guard's curve, if any, is `body.guard_log`."""
    r = recipe or SpanRecipe()
    if torch.device(device).type != "cpu":
        gpu_determinism()                       # no RNG, and never on the CPU (the golden reference)
    torch.manual_seed(seed)
    np.random.seed(seed)
    cfg = ModelConfig.preset(r.size, ternary=r.ternary, direction=r.direction)
    X, Y, S = task.train
    if r.shuffle_spans:
        # the shuffled-spans control: spans move BETWEEN windows, so they land where there is nothing: span-F1 must be ~0
        p = np.random.default_rng(seed).permutation(len(S))
        S, Y = S[p], Y[p]
    n_lab = n_labels(r.n_entities)
    Ylab = np.stack([mask_to_bioes(row, r.n_entities) for row in S])
    has_ignored = bool((Ylab == IGNORE_LABEL).any())
    # everything declared is checked BEFORE anything is born (a bad file fails in a second, not after the run); no RNG
    real = load_stream(r.real_windows, r.real_frac, r.real_mode, r.distill_alpha, r.batch, X.shape[1], n_lab, seed,
                       r.shuffle_spans, r.n_entities)
    G = guard_mod.setup(r.guard_windows, r.guard_every, r.guard_mode, r.guard_k, r.steps, X.shape[1], real)
    _check_memory_flags(r, cfg)
    ng_state = ngram_state(r.ngram_init) if r.ngram_init is not None else None
    check_adapter_args(r.adapter, r.lora_rank, r.base, r.ternary)
    if r.init_seed is not None:
        torch.manual_seed(r.init_seed)          # everything born from here on follows init_seed
    m = ByteModel(cfg).to(device)
    if r.base:
        load_base(m, r.base, cfg, device)
    if ng_state is not None:                    # ONLY tables + projection; the body keeps its init (or `base`'s)
        load_ngram(m, ng_state, r.size, r.ngram_init)
        del ng_state
    if r.freeze_ngram:
        for p in m.ngram.parameters():
            p.requires_grad_(False)
    th = TagHead(cfg.d_model, n_lab).to(device)
    ch = ChoiceHead(cfg.d_model, 2, pooling="max").to(device)
    if r.adapter != "none":                     # AFTER the heads (born as without it)
        attach_adapter(m, r.adapter, r.lora_rank, seed if r.init_seed is None else r.init_seed)
    mods = nn.ModuleList([m, th, ch])
    opt, sch = make_optimizer(mods, lr=r.lr, wd=r.wd, beta2=r.beta2, out_lr_mult=r.out_lr_mult, steps=r.steps,
                              warmup=r.warmup)
    rng = np.random.default_rng(seed)
    trainable = [p for p in mods.parameters() if p.requires_grad]
    ridx = batches(np.random.default_rng([seed, REAL_SEED]), len(real["X"]), real["n"], r.steps) if real else None
    t_run = time.perf_counter() if G is not None else None
    for step, i in enumerate(batches(rng, len(X), r.batch, r.steps)):
        x = torch.from_numpy(X[i].astype(np.int64)).to(device)
        if real is None:
            h, _ = m(x)
        else:                                   # ONE forward over [synthetic ; real], then split
            j = next(ridx)
            hh, _ = m(torch.cat([x, torch.from_numpy(real["X"][j].astype(np.int64)).to(device)]))
            h, hr = hh[:x.shape[0]], hh[x.shape[0]:]
        yl = torch.from_numpy(Ylab[i]).to(device).reshape(-1)
        yd = torch.from_numpy(Y[i]).to(device)
        if real is not None and real["mode"] == "neg":        # hard negatives join the batch: all-O, class 0
            h = hh
            yl = torch.cat([yl, yl.new_zeros(hr.shape[0] * hr.shape[1])])
            yd = torch.cat([yd, yd.new_zeros(hr.shape[0])])
        elif real is not None and real["mode"] == "spans":    # labeled real windows with their own targets
            h = hh
            k = real["perm"][j]
            yl = torch.cat([yl, torch.from_numpy(real["tag"][k]).to(device).reshape(-1)])
            yd = torch.cat([yd, torch.from_numpy(real["doc"][k]).to(device)])
        if r.span_weight > 1.0:                # extraction: the answer bytes are ~1 %, flat CE rewards silence
            ce = F.cross_entropy(th(h).float().reshape(-1, n_lab), yl, reduction="none")
            w = 1.0 + (r.span_weight - 1.0) * (yl > 0).float()
            if has_ignored:
                w = w * (yl != IGNORE_LABEL).float()
            l_tag = (ce * w).sum() / w.sum()
        else:
            l_tag = F.cross_entropy(th(h).float().reshape(-1, n_lab), yl)
        l_doc = F.cross_entropy(ch(h).float(), yd)
        loss = l_tag + r.lam * l_doc
        if real is not None and real["mode"] == "distill":
            loss = loss + r.distill_alpha * distill_loss(th, ch, hr, real, j, r.lam, device)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mods.parameters(), r.clip)
        opt.step()
        sch.step()
        opt.zero_grad(set_to_none=True)
        if G is not None and step + 1 in G["points"]:        # after step `step + 1` is complete
            guard_mod.point(G, step + 1, m, th, ch, trainable, opt, device, r.n_entities)
        if log is not None and (step + 1) % max(1, r.steps // 20) == 0:
            log(dict(step=step + 1, loss=float(loss.detach()), l_tag=float(l_tag.detach()),
                     l_doc=float(l_doc.detach()), lr=float(opt.param_groups[0]["lr"])))
    if G is not None:
        guard_mod.finish(G, m, trainable, time.perf_counter() - t_run)
    return m, th, ch
