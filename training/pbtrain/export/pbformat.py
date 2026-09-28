"""A trained model -> one GGUF file in the engine's format (spec/FORMAT.md). The spec is the authority; this module is a
table from registered PyTorch modules to spec tensors and keys.

Guarantees:
- COVERAGE: every tensor of the body's and the heads' state_dicts is written (or fused: a `scales` adapter gain goes
  into the ternary group scales, the same float32 product the model computes) or the export RAISES. A module the spec
  cannot express (a LoRA adapter) raises. Nothing is silently dropped.
- LOWEST VERSION: format 2 when the file only uses what version 2 defines, else 3 with `required_features` listing
  every optional capability the file needs (spec §3).
- The exporter never invents behavior: window policy, profile, labels and operating point come from the caller
  (the specialist's `specialist.yaml` and the factory's calibration).

    export(path, model, heads=[tag, choice], gating={0: 1}, labels=["secret"], operating_bias=0.5,
           profile="secrets-code", window=dict(size=512, stride=384), ngram_bits=4)
"""
import numpy as np

from ..model.layers.bitpack import pack_ternary_linear
from ..model.layers.ternary import TernaryLinear
from ..model.registry import head_type
from . import gguf

NGRAM_BITS = (4, 32)
TERN_META = {"purebyte.tern.type": "pb_tern2", "purebyte.tern.encoding": "00=0,01=+1,10=-1"}
EXIT_FEATURES = "rms_max_mean"
# Informative `general.*` keys of the GGUF convention (spec/FORMAT.md §4): written as strings, never read by the engine.
GENERAL_INFO = ("version", "license", "license.name", "license.link", "description", "author", "organization", "url")


class ExportError(ValueError):
    pass


def quant_ngram_int4(W):
    """Per-row symmetric 4-bit quantisation of a table [rows, part] -> (packed uint8 [rows, part/2], scale f32 [rows]).
    scale = max|row| / 7, q = rint(W / scale) in [-7, 7]; column 2j in the low nibble, 2j+1 in the high one (two's
    complement); a zero row gets scale 1. The engine reads it back as float32(scale) * float32(q)."""
    W = np.ascontiguousarray(W, dtype=np.float32)
    if W.ndim != 2 or W.shape[1] % 2:
        raise ExportError(f"a 4-bit n-gram table needs [rows, even part], got {W.shape}")
    scale = (np.abs(W).max(axis=1) / 7.0).astype(np.float32)
    scale = np.where(scale > 0, scale, np.float32(1.0)).astype(np.float32)
    q = np.clip(np.rint(W / scale[:, None]), -7, 7).astype(np.int8)
    nib = q.astype(np.uint8) & 0xF
    return (nib[:, 0::2] | (nib[:, 1::2] << 4)).astype(np.uint8), scale


def dequant_ngram_int4(packed, scale):
    """The way back, exactly as the engine does it (for the Python reference of a 4-bit model)."""
    packed = np.asarray(packed, dtype=np.uint8)
    q = np.empty((packed.shape[0], packed.shape[1] * 2), dtype=np.int8)
    q[:, 0::2] = (packed & 0xF).astype(np.int8)
    q[:, 1::2] = (packed >> 4).astype(np.int8)
    q = np.where(q >= 8, q - 16, q).astype(np.int8)
    return (q.astype(np.float32) * np.asarray(scale, np.float32)[:, None]).astype(np.float32)


class _Writer:
    def __init__(self):
        self.meta, self.tensors, self.covered, self.features = {}, {}, set(), set()

    def f32(self, name, t, key=None):
        a = t.detach().cpu().float().numpy().astype(np.float32) if hasattr(t, "detach") else np.asarray(t, np.float32)
        self.tensors[name] = a.reshape(1) if a.ndim == 0 else a
        if key is not None:
            self.covered.add(key)

    def projection(self, name, lin, key_prefix):
        """A linear map: ternary (packed + scales, gain fused) or float (spec §7)."""
        if not isinstance(lin, TernaryLinear):
            self.f32(name, lin.weight, key_prefix + "weight")
            return
        if lin.lora_A is not None:
            raise ExportError(f"{key_prefix[:-1]} carries a LoRA adapter: it has no layout in the format (the engine "
                              f"would silently run without it). Merge it or use the `scales` adapter.")
        if not lin.quantize:
            self.f32(name, lin.weight, key_prefix + "weight")
            return
        packed, scales = pack_ternary_linear(lin.weight, lin.group, lin.thr_mult)
        if lin.gain is not None:                  # `scales` adapter: fused exactly into the group scales
            scales = (scales * lin.gain.detach().cpu().float().numpy()).astype(np.float32)
            self.covered.add(key_prefix + "gain")
        self.tensors[name + ".packed"] = packed
        self.tensors[name + ".scales"] = scales
        self.covered.add(key_prefix + "weight")
        self.meta.update(TERN_META)
        self.meta["purebyte.tern.group"] = int(lin.group)


def _export_ngram(w, model, sd, bits):
    cfg = model.cfg
    ng = model.ngram
    K = ng.heads
    if bits not in (None,) + NGRAM_BITS:
        raise ExportError(f"ngram_bits={bits!r}; known: {NGRAM_BITS}")
    w.meta["purebyte.ngram.orders"] = [int(n) for n in cfg.ngram_orders]
    w.meta["purebyte.ngram.buckets"] = int(cfg.ngram_buckets)
    w.meta["purebyte.ngram.dim"] = int(cfg.ngram_dim)
    if K > 1:
        w.meta["purebyte.ngram.heads"] = int(K)
        w.meta["purebyte.ngram.hash"] = "poly base=257 mod=2^31-1 seed=1000*i+7+7919*h"
        w.features.add("ngram_heads")
    else:
        w.meta["purebyte.ngram.hash"] = "poly base=257 mod=2^31-1 seed=1000*i+7"
    if bits is not None:
        w.meta["purebyte.ngram.bits"] = int(bits)
    for i, n in enumerate(cfg.ngram_orders):
        for h in range(K):
            key = f"ngram.tables.{i * K + h}.weight"
            name = f"ngram.{n}.table" if K == 1 else f"ngram.{n}.{h}.table"
            if bits == 4:
                packed, scale = quant_ngram_int4(sd[key].detach().cpu().float().numpy())
                w.tensors[name] = packed
                w.tensors[name[:-len("table")] + "scale"] = scale
                w.covered.add(key)
            else:
                w.f32(name, sd[key], key)
    w.f32("ngram.proj.weight", sd["ngram.proj.weight"], "ngram.proj.weight")
    if cfg.ngram_layer:
        w.meta["purebyte.ngram.layer"] = int(cfg.ngram_layer)
        w.features.add("ngram_layer")
    if cfg.ngram_gate:
        w.meta["purebyte.ngram.gate"] = True
        w.f32("ngram.gate_bias", sd["ngram.gate_bias"], "ngram.gate_bias")
        w.features.add("ngram_gate")


def _export_ssm_v2(w, i, blk, sd):
    p = f"blocks.{i}."
    for nm in ("norm.weight", "gated_norm.weight", "A_log", "D", "dt_bias", "dt_proj.weight"):
        w.f32(p + nm, sd[p + nm], p + nm)
    if blk.conv is not None:
        w.f32(p + "conv1d.weight", sd[p + "conv.conv.weight"], p + "conv.conv.weight")
        w.f32(p + "conv1d.bias", sd[p + "conv.conv.bias"], p + "conv.conv.bias")
    w.projection(p + "in_proj.weight", blk.in_proj, p + "in_proj.")
    w.projection(p + "out_proj.weight", blk.out_proj, p + "out_proj.")


def _export_attention(w, i, blk, sd):
    p = f"blocks.{i}."
    w.meta[f"purebyte.block.{i}.head_dim"] = int(blk.HEAD_DIM)
    w.f32(p + "norm.weight", sd[p + "norm.weight"], p + "norm.weight")
    w.projection(p + "qkv.weight", blk.qkv, p + "qkv.")
    w.projection(p + "out.weight", blk.out, p + "out.")
    w.features.add("attention")


EXPORT_BLOCKS = {"ssm_v2": _export_ssm_v2, "attention": _export_attention}


def _export_head(w, i, hh, labels, operating_bias, gated_by):
    kind = head_type(hh)
    for k, v in hh.state_dict().items():
        w.f32(f"heads.{i}.{k}", v)
    w.meta[f"purebyte.head.{i}.type"] = kind
    if kind in ("choice", "ordinal"):
        w.meta[f"purebyte.head.{i}.pooling"] = hh.pooling
    if kind == "ordinal":
        w.features.add("ordinal")
    if kind == "tag":
        k = hh.n_entities
        w.meta[f"purebyte.head.{i}.n_entities"] = int(k)
        w.meta[f"purebyte.head.{i}.n_labels"] = int(1 + 4 * k)
        w.meta[f"purebyte.head.{i}.scheme"] = "BIOES:O,B-e,I-e,E-e,S-e"
        if labels:
            if len(labels) != k:
                raise ExportError(f"head {i}: {len(labels)} labels for {k} entity type(s)")
            w.meta[f"purebyte.head.{i}.labels"] = [str(x) for x in labels]
        if operating_bias is not None:
            if np.isscalar(operating_bias):
                w.meta[f"purebyte.head.{i}.operating_bias"] = float(operating_bias)
            else:
                if len(operating_bias) != k:
                    raise ExportError(f"head {i}: operating_bias_vec has {len(operating_bias)} values for {k} types")
                w.meta[f"purebyte.head.{i}.operating_bias_vec"] = [float(x) for x in operating_bias]
    if gated_by is not None:
        w.meta[f"purebyte.head.{i}.gated_by"] = int(gated_by)


def export(path, model, heads, *, gating=None, labels=None, operating_bias=None, profile=None, window=None,
           ngram_bits=None, name="purebyte", exit_head=None, info=None):
    """Writes the GGUF. `info`: informative strings written as `general.<key>` (keys in GENERAL_INFO, for example
    {"version": "1.0.0", "license": "other", "license.name": "..."}); they do not change what the file computes.
    `heads` are modules of the registered types `tag`, `choice` and `ordinal`; `exit_head` ({weight [2*d_model],
    bias, layer, threshold}, spec §11) appends the runtime's early-exit head (pbtrain does not train one; the export is
    checked against the NumPy reference in training/tests/test_export_reference.py).
    Returns dict(bytes, format_version, required_features, tensors)."""
    cfg = model.cfg
    sd = model.state_dict()
    w = _Writer()
    w.meta.update({"general.architecture": "purebyte", "general.name": str(name), "purebyte.format_version": 2,
                   "purebyte.d_model": int(cfg.d_model), "purebyte.n_layers": int(cfg.n_layers)})
    for k, v in (info or {}).items():
        if k not in GENERAL_INFO:
            raise ExportError(f"info key {k!r}: known informative keys are {GENERAL_INFO}")
        if v is not None:
            w.meta[f"general.{k}"] = str(v)
    if any(t == "ssm_v2" for t in cfg.block_types()):
        w.meta.update({"purebyte.n_heads": int(cfg.n_heads), "purebyte.d_head": int(cfg.d_head),
                       "purebyte.d_state": int(cfg.d_state), "purebyte.n_groups": int(cfg.n_groups),
                       "purebyte.conv_width": int(cfg.conv_width), "purebyte.direction": cfg.direction})
    w.f32("embed.weight", sd["embed.weight"], "embed.weight")
    w.f32("out_norm.weight", sd["out_norm.weight"], "out_norm.weight")
    if model.ngram is not None:
        _export_ngram(w, model, sd, ngram_bits)
    elif ngram_bits is not None:
        raise ExportError("ngram_bits given for a model without n-gram tables")
    for i, (kind, blk) in enumerate(zip(cfg.block_types(), model.blocks)):
        w.meta[f"purebyte.block.{i}.type"] = kind
        EXPORT_BLOCKS[kind](w, i, blk, sd)
    if model.look is not None:
        w.f32("look.conv_w", sd["look.conv_w"], "look.conv_w")
        w.meta["purebyte.lookahead"] = int(model.look.k)
        w.features.add("lookahead")
    heads = list(heads or [])
    w.meta["purebyte.head.count"] = len(heads) + (1 if exit_head is not None else 0)
    head_tensors = 0
    for i, hh in enumerate(heads):
        _export_head(w, i, hh, labels if head_type(hh) == "tag" else None,
                     operating_bias if head_type(hh) == "tag" else None, (gating or {}).get(i))
        head_tensors += len(hh.state_dict())
    for i, j in (gating or {}).items():
        if not (0 <= int(i) < len(heads) and 0 <= int(j) < len(heads)) or int(i) == int(j):
            raise ExportError(f"gating {i} <- {j} is not valid with {len(heads)} heads")
        if head_type(heads[int(j)]) not in ("choice", "ordinal"):
            raise ExportError(f"head {i} is gated by head {j}, which is not a choice / ordinal head")
    if exit_head is not None:
        i = len(heads)
        wt = np.asarray(exit_head["weight"], dtype=np.float32).reshape(1, -1)
        if wt.shape[1] != 2 * cfg.d_model:
            raise ExportError(f"exit head weight has {wt.shape[1]} columns, the features have {2 * cfg.d_model}")
        w.meta.update({f"purebyte.head.{i}.type": "exit", f"purebyte.head.{i}.layer": int(exit_head["layer"]),
                       f"purebyte.head.{i}.threshold": float(exit_head["threshold"]),
                       f"purebyte.head.{i}.features": EXIT_FEATURES})
        w.tensors[f"heads.{i}.proj.weight"] = wt
        w.tensors[f"heads.{i}.proj.bias"] = np.asarray([exit_head["bias"]], dtype=np.float32)
    if profile:
        w.meta["purebyte.profile"] = str(profile)
    for k in ("size", "stride", "min"):
        if window and window.get(k) is not None:
            w.meta[f"purebyte.window.{k}"] = int(window[k])
    missing = sorted(set(sd) - w.covered)
    if missing:
        raise ExportError(f"incomplete coverage, tensors of the body not exported: {missing}")
    v3 = (bool(w.features) or bool(profile) or bool(window) or bool(gating)
          or any(k.endswith(".pooling") for k in w.meta))
    if v3:
        w.meta["purebyte.format_version"] = 3
        if w.features:
            w.meta["purebyte.required_features"] = sorted(w.features)
    n = gguf.write(path, w.meta, w.tensors)
    return dict(bytes=n, format_version=w.meta["purebyte.format_version"], required_features=sorted(w.features),
                tensors=len(w.tensors), head_tensors=head_tensors)
