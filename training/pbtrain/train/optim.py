"""Optimizer pieces shared by every trainer: parameter groups, the WSD schedule, epoch-shuffled batches.

Rules (all measured):
- weight decay only on matrices. Every parameter with fewer than two dimensions (A_log, D, dt_bias, norms, biases,
  the n-gram gate bias) is never decayed, whatever its name. Among MATRICES, a parameter whose name holds one of
  NO_DECAY_SUBSTRINGS (dt_proj, the conv1d weight, embeddings, n-gram tables and projection, the `scales` adapter's
  gains, LoRA, the lookahead's `conv_w`) is not decayed either: a new matrix that must not be decayed must carry one
  of these substrings in its name.
- WSD: linear warmup -> plateau -> linear decay over the last 20 %.
- batches are drawn WITHOUT replacement, reshuffled per epoch (drawing with replacement saw only 63 % of the data).
  A batch larger than the training split is refused by the task's validation (`train.batch <= data.n_train`).
"""
import numpy as np
import torch

NO_DECAY_SUBSTRINGS = ("A_log", "dt_bias", "dt_proj", "norm", "conv", "embed", "ngram", "bias", "gain", "lora")


def param_groups(model, wd):
    """[decayed matrices, everything else at 0] over the parameters that train, in module order."""
    no_decay, decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        flat = any(k in n for k in NO_DECAY_SUBSTRINGS) or n.endswith(".D") or n == "D"
        (no_decay if (flat or p.ndim < 2) else decay).append(p)
    return [{"params": decay, "weight_decay": wd},
            {"params": no_decay, "weight_decay": 0.0}]


def split_out_proj(groups, mods, lr, out_lr_mult):
    """Every `out_proj` matrix in its own group with lr x `out_lr_mult` (same weight decay). 1.0 = no split."""
    if out_lr_mult == 1.0:
        return groups
    ids = {id(p) for n, p in mods.named_parameters() if ".out_proj." in n}
    groups = [dict(g, params=[p for p in g["params"] if (id(p) in ids) == own],
                   **({"lr": lr * out_lr_mult} if own else {}))
              for g in groups for own in (False, True)]
    return [g for g in groups if g["params"]]


def wsd_lr(step, total, warmup, decay_frac=0.2):
    """Multiplier of the base lr: linear warmup -> constant plateau -> linear decay to zero."""
    if step < warmup:
        return (step + 1) / max(1, warmup)
    d0 = int(total * (1 - decay_frac))
    if step < d0:
        return 1.0
    return max(0.0, (total - step) / max(1, total - d0))


def make_optimizer(mods, *, lr, wd, beta2, out_lr_mult, steps, warmup):
    groups = split_out_proj(param_groups(mods, wd), mods, lr, out_lr_mult)
    opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, beta2))
    sch = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: wsd_lr(s, steps, warmup))
    return opt, sch


def batches(rng, n, bs, steps):
    """Batch indices shuffled per epoch, without replacement."""
    order, cursor = rng.permutation(n), 0
    for _ in range(steps):
        if cursor + bs > len(order):
            order, cursor = rng.permutation(n), 0
        yield order[cursor:cursor + bs]
        cursor += bs


def cpu_clone(o):
    """A deep copy of an optimizer state_dict with every tensor cloned to the CPU (snapshots never live in VRAM)."""
    if torch.is_tensor(o):
        return o.detach().to("cpu", copy=True)
    if isinstance(o, dict):
        return {k: cpu_clone(v) for k, v in o.items()}
    if isinstance(o, list):
        return [cpu_clone(v) for v in o]
    return o


def as_long(a, device):
    return torch.from_numpy(np.asarray(a).astype(np.int64)).to(device)
