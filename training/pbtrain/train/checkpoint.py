"""The checkpoint of a span run: body + tag head + choice head + the config that rebuilds them.

    {model, tag, head, cfg, pooling, n_classes, n_entities, seed, init_seed, bias, complete, provenance?, guard?, ...}

`bias=None, complete=False` marks a checkpoint saved BEFORE its operating point was chosen: the weights are good for
sweeping again, not for delivering (`load(..., require_complete=True)` refuses it). `provenance` records the device,
backend and software that trained it (pbtrain/provenance.py).

Checkpoints are loaded with `torch.load(weights_only=True)`: they hold only tensors, numbers, strings, lists, tuples and
dicts, and a file that holds anything else (arbitrary pickled objects) is refused instead of executed.
"""
import os
from dataclasses import asdict

import torch

from ..model.body import ByteModel
from ..model.config import ModelConfig
from ..model.decode import n_labels
from ..model.heads import ChoiceHead, TagHead


def state(m, th, ch, **meta):
    cpu = lambda mod: {k: v.detach().cpu() for k, v in mod.state_dict().items()}
    return dict(model=cpu(m), tag=cpu(th), head=cpu(ch), cfg=asdict(m.cfg), pooling="max", n_classes=2,
                n_entities=(th.proj.out_features - 1) // 4, **meta)


def save(path, m, th, ch, **meta):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    torch.save(state(m, th, ch, **meta), path + ".tmp")
    os.replace(path + ".tmp", path)


def rebuild(ck, device="cpu"):
    """(body, tag head, choice head) from a checkpoint dict written by `save`."""
    cfg = ModelConfig.from_dict(ck["cfg"])
    m = ByteModel(cfg)
    m.load_state_dict(ck["model"])
    k = int(ck.get("n_entities", 1))
    th = TagHead(cfg.d_model, n_labels(k))
    th.load_state_dict(ck["tag"])
    ch = ChoiceHead(cfg.d_model, int(ck.get("n_classes", 2)), pooling=ck.get("pooling", "max"))
    ch.load_state_dict(ck["head"])
    if ck.get("guard"):
        m.guard_log = ck["guard"]
    return m.to(device), th.to(device), ch.to(device)


def safe_load(path, what="checkpoint"):
    """`torch.load(path, weights_only=True)` on the CPU, with a clear error for a file that holds pickled objects."""
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as e:            # an UnpicklingError, or whatever torch raises for a file it cannot read safely
        raise ValueError(f"{what} {path} cannot be loaded as plain tensors and values (torch.load weights_only=True: "
                         f"{type(e).__name__}: {str(e)[:200]}). Save it as a plain state_dict") from e


def load(path, require_complete=False):
    ck = safe_load(path)
    if require_complete and not ck.get("complete", True):
        raise ValueError(f"{path} is a HALF-DONE checkpoint (no operating point was chosen): sweep it again with the "
                         f"finetune stage; it cannot be delivered")
    if require_complete and ck.get("bias") is None:
        raise ValueError(f"{path} carries no operating bias")
    return ck
