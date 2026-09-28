"""`verify`: the runtime runs the exported file with ITS OWN operating point (scalar, or one value per entity type) and
must report the same spans as Python. Uses the NumPy reference; with PUREBYTE_MODEL_DUMP also the engine."""
import os
import types

import numpy as np
import pytest

import golden_cases as gc
from pbtrain.export.pbformat import export
from pbtrain.factory.verify import compare_runtime
from pbtrain.model.decode import window_class
from pbtrain.train import checkpoint
from pbtrain.train.loop import SpanTask, train_spans
from pbtrain.train.recipe import SpanRecipe


def _typed_model():
    X, Y, S = gc.Task().train
    S3 = S.copy()
    S3[::4][S3[::4] == 1] = 3
    task = SpanTask((X, window_class(S3).astype(np.int64), S3), None, None)
    m, th, ch = train_spans(task, SpanRecipe(size="mini64", steps=30, batch=8, lr=3e-3, warmup=2, n_entities=3))
    return X, m.eval(), th.eval(), ch.eval()


@pytest.mark.parametrize("runtime", ["reference", "engine"])
def test_runtime_uses_the_files_operating_point(runtime, tmp_path, monkeypatch):
    if runtime == "engine":
        if not os.environ.get("PUREBYTE_MODEL_DUMP"):
            pytest.skip("set PUREBYTE_MODEL_DUMP to the engine's model_dump")
    else:
        monkeypatch.delenv("PUREBYTE_MODEL_DUMP", raising=False)
    X, m, th, ch = _typed_model()
    vec = [2.0, 4.0, 6.0]                                   # generous: the tiny model must emit some spans
    ck = checkpoint.state(m, th, ch, bias=1.0, bias_vec=vec, complete=True)
    path = str(tmp_path / "typed.gguf")
    export(path, m, [th, ch], gating={0: 1}, labels=["a", "b", "c"], operating_bias=vec, window=dict(size=64, stride=48))
    res = compare_runtime(path, ck, X[:6 if runtime == "reference" else 48])
    assert res["identical_spans"], res
    assert res["windows_with_spans"] > 0, "vacuous: no window produced a span"


def test_verify_compares_windows_with_and_without_spans(tmp_path, monkeypatch):
    """`verify` picks half of its windows where Python finds a span and half where it finds none; a model that emits
    nothing leaves no window with a span (the stage then refuses: the comparison would prove nothing)."""
    from pbtrain.factory.verify import pick_windows, python_spans
    monkeypatch.delenv("PUREBYTE_MODEL_DUMP", raising=False)
    X, m, th, ch = _typed_model()
    vec = [2.0, 4.0, 6.0]
    ck = checkpoint.state(m, th, ch, bias=1.0, bias_vec=vec, complete=True)
    idx = pick_windows(ck, X, 6)
    spans_of = python_spans(ck, X[idx])
    assert len(idx) == 6 and sum(1 for s in spans_of if s) == 3 and sum(1 for s in spans_of if not s) == 3
    path = str(tmp_path / "typed.gguf")
    export(path, m, [th, ch], gating={0: 1}, labels=["a", "b", "c"], operating_bias=vec, window=dict(size=64, stride=48))
    res = compare_runtime(path, ck, X, n=4)
    assert res["n"] == 4 and res["identical_spans"] and res["windows_with_spans"] == 2 and res["engine_threads"] is None
    silent = checkpoint.state(m, th, ch, bias=-30.0, complete=True)
    assert all(not s for s in python_spans(silent, X[pick_windows(silent, X, 6)]))


def test_the_engine_gets_the_tasks_thread_count(monkeypatch, tmp_path):
    import json as _json
    import pbtrain.factory.verify as verify
    seen = {}

    def fake_run(cmd, **kw):
        seen["cmd"] = cmd
        return types.SimpleNamespace(returncode=0, stdout=_json.dumps({"inputs": [{"windows": []}]}), stderr="")
    monkeypatch.setattr(verify.subprocess, "run", fake_run)
    verify._engine_spans("model_dump", "m.gguf", np.zeros((1, 8), np.uint8), threads=3)
    assert seen["cmd"][-1] == "--threads=3"


@pytest.mark.parametrize("runtime", ["reference", "engine"])
def test_runtime_of_a_4bit_ngram_file_is_compared_with_its_own_tables(runtime, tmp_path, monkeypatch):
    """A file with 4-bit n-gram tables computes the checkpoint with quantized tables: verify compares with THAT (and
    counts apart the windows where the quantization alone changes the spans)."""
    if runtime == "engine":
        if not os.environ.get("PUREBYTE_MODEL_DUMP"):
            pytest.skip("set PUREBYTE_MODEL_DUMP to the engine's model_dump")
    else:
        monkeypatch.delenv("PUREBYTE_MODEL_DUMP", raising=False)
    from pbtrain.model.config import PRESETS
    gc.register_test_presets(PRESETS)
    kw = dict(gc.COMMON, size="mini64_n", steps=60, lr=3e-3, warmup=2)
    m, th, ch = train_spans(gc.Task(), SpanRecipe(**kw), seed=0)
    m, th, ch = m.eval(), th.eval(), ch.eval()
    ck = checkpoint.state(m, th, ch, bias=6.0, complete=True)   # generous: the tiny model must emit some spans
    path = str(tmp_path / "ngram4.gguf")
    export(path, m, [th, ch], gating={0: 1}, labels=["secret"], operating_bias=6.0, ngram_bits=4)
    X = gc.Task().train[0]
    res = compare_runtime(path, ck, X[:6 if runtime == "reference" else 48])
    assert res["identical_spans"], res
    assert res["ngram_bits"] == 4 and "int4_changed_windows" in res
    assert res["windows_with_spans"] > 0, "vacuous: no window produced a span"
