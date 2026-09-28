"""Where the weights of a run come from, besides a fresh init: a pretrained body, an n-gram memory, a frozen body with
an adapter. All OFF by default; none of them draws from the global RNG, so the rest of the model is born exactly as
without them (bit-identity of the untouched paths).

base          a body checkpoint: the raw state_dict of a pretraining, or a factory checkpoint (`finetune/seed_<s>/
              final.pt`; its `cfg` must equal this run's). A body pretrained WITHOUT n-gram tables may seed a model
              that has them; nothing else may be missing. It helped on binaries (fewer false alarms) and did not help
              on source code.
ngram_init    copies EXACTLY a checkpoint's `ngram.*` tensors (tables + projection); keys, shapes AND the hashing
              config must match (orders (2, 3) and (2, 4) have the same shapes and different hashes).
freeze_ngram  `m.ngram` gets no gradient (out of the optimizer, no decay).
adapter       frozen body + `heads` (linear probe) | `scales` (a gain per ternary group scale + the float vectors of every
              block; exported fused) | `lora` (the control, Python-only). Switched on AFTER the heads are born.

Both files are read with `torch.load(weights_only=True)` (checkpoint.safe_load): a file holding pickled objects other
than tensors and plain values is refused, not executed.
"""
import json
import os
from dataclasses import asdict

import torch

from ..model.layers.ternary import TernaryLinear
from .checkpoint import safe_load
from .recipe import ADAPTERS

NGRAM_HASH_CFG = ("ngram_orders", "ngram_buckets", "ngram_dim", "ngram_heads")
ADAPTER_VECTORS = (".norm.weight", ".gated_norm.weight", ".A_log", ".D", ".dt_bias", ".dt_proj.weight",
                   ".conv.conv.weight", ".conv.conv.bias")
ADAPTER_SEED = 0xADA


def _norm(v):
    return tuple(v) if isinstance(v, (list, tuple)) else v


def base_state(path, cfg, device):
    """`base` as a state_dict (on the CPU; `load_base` copies it into the model on `device`). A factory checkpoint
    gives its body, and its cfg must be THIS run's."""
    ck = safe_load(path, "`base`")
    if not (isinstance(ck, dict) and isinstance(ck.get("model"), dict)):
        return ck
    if ck.get("cfg") is not None:
        mine = asdict(cfg)
        bad = {k: (ck["cfg"][k], mine[k]) for k in sorted(set(ck["cfg"]) & set(mine))
               if _norm(ck["cfg"][k]) != _norm(mine[k])}
        if bad:
            raise ValueError(f"`base: {path}` was trained with another model config than this run's `size` "
                             f"(checkpoint, run): {bad}")
    return ck["model"]


def load_base(m, path, cfg, device):
    r = m.load_state_dict(base_state(path, cfg, device), strict=False)
    if r.unexpected_keys or not all(k.startswith("ngram.") for k in r.missing_keys):
        raise ValueError(f"`base: {path}` does not fit: unexpected {r.unexpected_keys}, missing {r.missing_keys}")


def ngram_state(path):
    """(the `ngram.*` tensors of a checkpoint, on CPU; the config dict it declares, or None)."""
    if not os.path.isfile(str(path)):
        raise ValueError(f"`ngram_init: {path}` does not exist")
    ck = safe_load(path, "`ngram_init`")
    cfg = None
    if isinstance(ck, dict) and isinstance(ck.get("model"), dict):          # a factory checkpoint
        cfg, ck = ck.get("cfg"), ck["model"]
    elif os.path.isfile(str(path) + ".json"):                                # a pretraining's sidecar
        with open(str(path) + ".json", encoding="utf-8") as f:
            cfg = json.load(f).get("cfg")
    sd = {k: v for k, v in ck.items() if isinstance(k, str) and k.startswith("ngram.") and torch.is_tensor(v)}
    if not sd:
        raise ValueError(f"`ngram_init: {path}` holds no `ngram.` tensor: it is not a model with n-gram tables")
    return sd, cfg


def load_ngram(m, state, size, path):
    """Copies EXACTLY the `ngram.*` tensors into `m`, in place (no RNG, nothing else touched)."""
    sd, ck_cfg = state
    own = {k: v for k, v in m.state_dict().items() if k.startswith("ngram.")}
    miss, extra = sorted(set(own) - set(sd)), sorted(set(sd) - set(own))
    shape = [f"{k} {tuple(sd[k].shape)} vs {tuple(own[k].shape)}" for k in sorted(set(own) & set(sd))
             if sd[k].shape != own[k].shape]
    if miss or extra or shape:
        raise ValueError(f"`ngram_init: {path}` does not fit `{size}`: missing {miss}, unexpected {extra}, shape {shape}")
    if ck_cfg is not None:
        mine = asdict(m.cfg)
        bad = {k: (ck_cfg[k], mine[k]) for k in NGRAM_HASH_CFG
               if k in ck_cfg and k in mine and _norm(ck_cfg[k]) != _norm(mine[k])}
        if bad:
            raise ValueError(f"`ngram_init: {path}` was trained with another n-gram hashing than `{size}` "
                             f"(checkpoint, model): {bad}")
    r = m.load_state_dict(sd, strict=False)
    assert not r.unexpected_keys, r


def attach_adapter(m, adapter, lora_rank, seed):
    """Freezes the WHOLE body and switches `adapter` on; returns the names of the body parameters that train. Born =
    the model before (gain 1, LoRA B 0); LoRA's A comes from its own generator (the global RNG is not touched)."""
    if adapter not in ADAPTERS:
        raise ValueError(f"`adapter: {adapter!r}`; the valid ones are {ADAPTERS}")
    for p in m.parameters():
        p.requires_grad_(False)
    if adapter == "heads":
        return []
    gen = torch.Generator().manual_seed(int(seed) + ADAPTER_SEED)
    for mod in m.modules():
        if isinstance(mod, TernaryLinear):
            if adapter == "scales":
                mod.enable_gain()
            else:
                mod.enable_lora(lora_rank, generator=gen)
    names = []
    for n, p in m.named_parameters():
        if n.endswith(ADAPTER_VECTORS) or n == "out_norm.weight" or n.endswith((".gain", ".lora_A", ".lora_B")):
            p.requires_grad_(True)
            names.append(n)
    return names


def check_adapter_args(adapter, lora_rank, base, ternary):
    if adapter not in ADAPTERS:
        raise ValueError(f"`adapter: {adapter!r}`; the valid ones are {ADAPTERS}")
    if lora_rank is not None and adapter != "lora":
        raise ValueError("`lora_rank` without `adapter: lora`: it would be declared with no effect")
    if adapter == "lora" and (lora_rank is None or int(lora_rank) < 1):
        raise ValueError("`adapter: lora` needs `lora_rank` >= 1 (8 is the control)")
    if adapter != "none" and not base:
        raise ValueError(f"`adapter: {adapter}` freezes the body and there is no `base` (`base_checkpoint`): it would "
                         f"freeze a RANDOM body")
    if adapter == "scales" and not ternary:
        raise ValueError("`adapter: scales` corrects the ternary group scales: a float model has none")
