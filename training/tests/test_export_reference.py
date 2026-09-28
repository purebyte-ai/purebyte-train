"""Export -> the runtime's own readers. A trained model exported by pbtrain must load in the NumPy reference
(reference/purebyte_ref, the executable companion of spec/FORMAT.md) and compute what PyTorch computes:
- final hidden states and the tag head's log-probabilities within float tolerance (the reference follows the spec's
  order of operations, PyTorch its own GEMM order: they agree to ~1e-5, not bit for bit);
- the SAME spans when both decode the same log-probabilities (the decoding contract is exact);
- the choice head's probabilities within tolerance;
- the heads pbtrain exports but the factory does not train (an `ordinal` head, an `exit` head): loaded and computed by
  the reference like PyTorch.
With PUREBYTE_MODEL_DUMP pointing to the engine's `model_dump` tool (the runtime repository's
tests/tools/model_dump.cpp), the same files are also loaded by the C++ engine, whose validation is strict (spec §13),
and its hidden states are compared too.
"""
import json
import os
import struct
import subprocess
import sys

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import golden_cases as gc
from pbtrain.export.pbformat import dequant_ngram_int4, export, quant_ngram_int4
from pbtrain.export.gguf import read_metadata
from pbtrain.model.config import PRESETS
from pbtrain.model.decode import label_bias, spans, viterbi
from pbtrain.train.loop import train_spans
from pbtrain.train.recipe import SpanRecipe

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "reference"))
gc.register_test_presets(PRESETS)

CASES = {
    "ternary_causal": dict(),
    "float": dict(ternary=False),
    "hydra": dict(direction="hydra"),
    "bimamba": dict(direction="bimamba"),
    "ngram_32bit": dict(size="mini64_n"),
    "ngram_4bit": dict(size="mini64_n", _bits=4),
    "ngram_heads_2": dict(size="mini64_h2"),
    "ngram_gate_layer1": dict(size="mini64_g1"),
    "lookahead": dict(size="mini64_la4"),
    "attention": dict(size="mini64_a"),
}
DATA = bytes(np.random.default_rng(7).integers(32, 127, 96, dtype=np.uint8)) + b' api_key = "sk_live_' + b"x" * 24 + b'"\n'


def _train(case):
    kw = dict(gc.COMMON)
    kw.update({k: v for k, v in CASES[case].items() if not k.startswith("_")})
    kw["steps"] = 12
    return train_spans(gc.Task(), SpanRecipe(**kw), seed=0)


def _to_4bit(m):
    """The Python reference of a 4-bit model: every table replaced by its dequantised self."""
    with torch.no_grad():
        for t in m.ngram.tables:
            packed, scale = quant_ngram_int4(t.weight.detach().numpy())
            t.weight.copy_(torch.from_numpy(dequant_ngram_int4(packed, scale)))


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("pbtrain_export"))
    out = {}
    for case, cfg in CASES.items():
        m, th, ch = _train(case)
        bits = cfg.get("_bits")
        if bits == 4:
            _to_4bit(m)
        path = os.path.join(d, case + ".gguf")
        info = export(path, m, [th, ch], gating={0: 1}, labels=["secret"], operating_bias=0.5, profile="secrets-code",
                      window=dict(size=512, stride=384), ngram_bits=bits)
        out[case] = (path, m.eval(), th.eval(), ch.eval(), info)
    return out


@pytest.mark.parametrize("case", list(CASES))
def test_reference_loads_and_matches(case, exported):
    from purebyte_ref import Model
    path, m, th, ch, info = exported[case]
    ref = Model.open(path)
    with torch.no_grad():
        h, _ = m(torch.tensor(list(DATA), dtype=torch.int64)[None])
        lp_t = F.log_softmax(th(h).float(), -1)[0].numpy()
        p_t = F.softmax(ch(h).float(), -1)[0].numpy()
    H = ref.forward(DATA)
    ht = h[0].numpy()
    assert H.shape == ht.shape
    assert np.abs(H - ht).max() <= 1e-4 * max(1.0, np.abs(ht).max()), f"{case}: hidden states differ"
    tag, choice = ref.heads[0], ref.heads[1]
    lp_r = tag.logp(H)
    assert np.abs(lp_r - lp_t).max() < 1e-3
    for b in (0.0, 0.5, -1.0):
        ours = [tuple(s) for s in spans(viterbi(lp_r, 1, label_bias(1, b)), 1)]
        theirs = [tuple(s[:3]) for s in tag.run(H, 0, {"scalar": b})["spans"]]
        assert ours == theirs, f"{case} bias {b}: decoding differs"
    assert np.abs(choice.run(H, 0)["values"] - p_t).max() < 1e-4


def test_declared_metadata(exported):
    meta = read_metadata(exported["ternary_causal"][0])
    assert meta["purebyte.format_version"] == 3                   # profile + window + gating are version-3 keys
    assert meta["purebyte.profile"] == "secrets-code"
    assert meta["purebyte.head.0.type"] == "tag" and meta["purebyte.head.1.type"] == "choice"
    assert meta["purebyte.head.0.gated_by"] == 1 and meta["purebyte.head.0.labels"] == ["secret"]
    assert abs(meta["purebyte.head.0.operating_bias"] - 0.5) < 1e-7
    assert "purebyte.required_features" not in meta
    feats = read_metadata(exported["attention"][0])["purebyte.required_features"]
    assert feats == ["attention"]
    assert sorted(read_metadata(exported["ngram_gate_layer1"][0])["purebyte.required_features"]) == \
        ["ngram_gate", "ngram_layer"]


def test_an_ordinal_head_exports_and_matches_the_reference(tmp_path):
    from purebyte_ref import Model
    from pbtrain.model.heads import OrdinalHead
    m, th, _ = _train("ternary_causal")
    torch.manual_seed(3)
    oh = OrdinalHead(m.cfg.d_model, 4, pooling="max_mean")
    with torch.no_grad():
        oh.b.copy_(torch.tensor([0.3, -0.2, 0.1]))
    path = str(tmp_path / "ordinal.gguf")
    export(path, m.eval(), [th.eval(), oh.eval()], gating={0: 1}, labels=["secret"], operating_bias=0.5)
    assert read_metadata(path)["purebyte.head.1.type"] == "ordinal"
    ref = Model.open(path)
    with torch.no_grad():
        h, _ = m(torch.tensor(list(DATA), dtype=torch.int64)[None])
        p_t = torch.sigmoid(oh(h))[0].numpy()
    got = ref.heads[1].run(ref.forward(DATA), 0)["values"]
    assert np.abs(got - p_t).max() < 1e-4


def test_an_exit_head_exports_and_the_reference_exits_with_it(tmp_path):
    from purebyte_ref import Model
    m, th, ch = _train("ternary_causal")
    w = np.random.default_rng(4).normal(size=2 * m.cfg.d_model).astype(np.float32) * 0.1
    outs = {}
    for name, threshold in (("never", -1e9), ("always", 1e9)):
        path = str(tmp_path / f"exit_{name}.gguf")
        export(path, m.eval(), [th.eval(), ch.eval()], gating={0: 1}, labels=["secret"], operating_bias=0.5,
               exit_head=dict(weight=w, bias=0.25, layer=1, threshold=threshold))
        meta = read_metadata(path)
        assert meta["purebyte.head.count"] == 3 and meta["purebyte.head.2.type"] == "exit"
        ref = Model.open(path)
        assert ref.exit is not None and ref.exit.layer == 1
        outs[name] = ref.forward(DATA, early_exit=True)
    with torch.no_grad():
        h, _ = m(torch.tensor(list(DATA), dtype=torch.int64)[None])
    assert outs["always"] is None                                    # the exit head stopped the run at layer 1
    assert np.abs(outs["never"] - h[0].numpy()).max() <= 1e-4 * max(1.0, float(h.abs().max()))


def test_lora_refuses_to_export(tmp_path):
    from pbtrain.model.layers.ternary import TernaryLinear
    m, th, ch = _train("ternary_causal")
    for mod in m.modules():
        if isinstance(mod, TernaryLinear):
            mod.enable_lora(2)
    with pytest.raises(ValueError, match="LoRA"):
        export(str(tmp_path / "x.gguf"), m, [th, ch], gating={0: 1})


def _inputs_file(path, items):
    with open(path, "wb") as f:
        f.write(struct.pack("<I", len(items)))
        for b in items:
            f.write(struct.pack("<I", len(b)) + b)


@pytest.mark.skipif(not os.environ.get("PUREBYTE_MODEL_DUMP"), reason="set PUREBYTE_MODEL_DUMP to the engine's model_dump")
@pytest.mark.parametrize("case", list(CASES))
def test_engine_loads_and_matches(case, exported, tmp_path):
    path, m, th, ch, _ = exported[case]
    inp = str(tmp_path / "inputs.bin")
    _inputs_file(inp, [DATA])
    out = subprocess.run([os.environ["PUREBYTE_MODEL_DUMP"], path, inp, "--whole", "--hidden"], capture_output=True,
                         text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-600:]
    res = json.loads(out.stdout)
    H = np.asarray(res["inputs"][0]["hidden"], dtype=np.float32)
    with torch.no_grad():
        h, _ = m(torch.tensor(list(DATA), dtype=torch.int64)[None])
    ht = h[0].numpy()
    assert np.abs(H - ht).max() <= 1e-4 * max(1.0, np.abs(ht).max()), f"{case}: engine hidden states differ"
