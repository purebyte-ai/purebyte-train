"""The training contract: pbtrain reproduces, BIT FOR BIT, the weights of the code base the released specialists were
trained with, on every trainer path (golden_cases.CASES: n-gram tables and their gate, lookahead, attention, the three
directions, float control, optimizer options, real-window streams, the tail guard, pretrained bodies, adapters, the
n-gram memory).

The fingerprints in golden/training_fingerprints.json were recorded on one platform (see its `meta`): CPU kernels
differ between PyTorch builds, between Python and NumPy versions, and between processors of one platform (the AVX2 and
AVX-512 code paths of `torch.backends.cpu.get_cpu_capability()`). The comparison therefore runs only where all of
them match the recording: PyTorch, NumPy, operating system, machine, Python major.minor and CPU capability, one
thread; anywhere else it is skipped and says why (`-rs`). CI does not compare them (its runners set
PBTRAIN_GOLDEN_SKIP=1: a hosted runner's processor is not guaranteed to match); they are compared on the recording
platform before every release (AGENTS.md). The determinism test runs everywhere. Never regenerate the fingerprints to
make this test pass: a difference means the trainer changed.
"""
import json
import os
import platform

import numpy as np
import pytest
import torch

import golden_cases as gc
from pbtrain.model.config import PRESETS
from pbtrain.provenance import cpu_capability
from pbtrain.train.loop import train_spans
from pbtrain.train.recipe import SpanRecipe

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "training_fingerprints.json")
gc.register_test_presets(PRESETS)


def _golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


def here():
    """What decides the bits of a CPU run, on this machine."""
    return dict(torch=torch.__version__, numpy=np.__version__, system=platform.system(), machine=platform.machine(),
                python=".".join(platform.python_version_tuple()[:2]), cpu_capability=cpu_capability())


def same_platform(meta):
    """(True when this machine matches the recording in every field the meta records, what this machine is)."""
    h = here()
    rec = dict(meta, python=".".join(str(meta.get("python", "")).split(".")[:2]))
    return all(rec.get(k) == v for k, v in h.items() if k in meta), h


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("pbtrain_golden"))
    return gc.make_checkpoints(d, gc.make_files(d))


def _run(case, files):
    torch.set_num_threads(gc.THREADS)            # CPU reductions depend on the thread count
    seed, kw = gc.resolve(case, files)
    m, th, ch = train_spans(gc.Task(), SpanRecipe(**kw), seed=seed)
    return gc.fingerprint(m, th, ch)


@pytest.mark.parametrize("case", list(gc.CASES))
def test_weights_match_the_golden_fingerprints(case, files):
    if os.environ.get("PBTRAIN_GOLDEN_SKIP") == "1":
        pytest.skip("PBTRAIN_GOLDEN_SKIP=1: CPU kernels can differ between processors (compared on the recording "
                    "platform before every release)")
    g = _golden()
    ok, h = same_platform(g["meta"])
    if not ok:
        pytest.skip(f"fingerprints recorded on {g['meta']}, this is {h}")
    sha, keys = _run(case, files)
    assert len(keys) == g["cases"][case]["n_tensors"]
    assert sha == g["cases"][case]["sha"], f"{case}: the trainer no longer reproduces the reference weights"


def test_the_platform_gate_compares_every_recorded_field():
    """A recording that differs from this machine in ANY field it records (the CPU capability or the Python minor
    version included) is not compared here."""
    h = here()
    meta = dict(h, python=h["python"] + ".7", threads=1)
    assert same_platform(meta)[0]
    for field, other in (("cpu_capability", "AVX512" if h["cpu_capability"] != "AVX512" else "AVX2"),
                         ("python", "2.7.18"), ("torch", "0.0.0"), ("numpy", "0.0.0"), ("machine", "other")):
        assert not same_platform(dict(meta, **{field: other}))[0], field
    assert "cpu_capability" in _golden()["meta"] and "python" in _golden()["meta"]


def test_training_is_deterministic(files):
    """Twice the same run -> the same bits (holds on every platform)."""
    assert _run("real_spans", files) == _run("real_spans", files)


def test_every_case_is_in_the_golden_file():
    assert set(_golden()["cases"]) == set(gc.CASES)
