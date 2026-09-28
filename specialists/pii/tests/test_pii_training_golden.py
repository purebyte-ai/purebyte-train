"""The training contract of the pii recipes: pbtrain reproduces, BIT FOR BIT, the weights of the code the released pii
model was trained with, on what the pii recipes add to the core's golden cases (one entity type with IGNORE bytes, its
control, the n-gram tables on such windows, the release size preset `nano_n15`): pii_golden_training.CASES.

The fingerprints in golden/pii_training_fingerprints.json were recorded on one platform and torch build (see its
`meta`): CPU kernels differ between builds, so on any other combination the comparison is skipped (the determinism test
still runs everywhere). Never regenerate the fingerprints to make this test pass: a difference means the trainer
changed.
"""
import json
import os
import platform

import numpy as np
import pytest
import torch
from pbtrain.model.decode import IGNORE_LABEL, mask_to_bioes
from pbtrain.train.loop import train_spans
from pbtrain.train.recipe import SpanRecipe

import pii_golden_training as pg

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "pii_training_fingerprints.json")


def _golden():
    with open(GOLDEN, encoding="utf-8") as f:
        return json.load(f)


def _same_platform(meta):
    """Like training/tests/test_golden_training.py: every field the meta records must match this machine, the Python
    major.minor and the CPU kernel path (`torch.backends.cpu.get_cpu_capability()`) included."""
    from pbtrain.provenance import cpu_capability
    here = dict(torch=torch.__version__, numpy=np.__version__, system=platform.system(), machine=platform.machine(),
                python=".".join(platform.python_version_tuple()[:2]), cpu_capability=cpu_capability())
    rec = dict(meta, python=".".join(str(meta.get("python", "")).split(".")[:2]))
    return all(rec.get(k) == v for k, v in here.items() if k in meta), here


def _run(case):
    torch.set_num_threads(pg.THREADS)              # CPU reductions depend on the thread count
    seed, kw = pg.resolve(case)
    m, th, ch = train_spans(pg.MaskTask(), SpanRecipe(**kw), seed=seed)
    return pg.fingerprint(m, th, ch)


def test_the_task_carries_ignore_bytes_the_trainer_skips():
    X, Y, S = pg.MaskTask().train
    labels = np.stack([mask_to_bioes(row, 1) for row in S])
    assert int((S == 255).sum()) == _golden()["meta"]["ignored_bytes"] > 0
    assert (labels == IGNORE_LABEL).sum() == (S == 255).sum()
    assert set(np.unique(labels).tolist()) == {IGNORE_LABEL, 0, 1, 2, 3}, "B, I, E of one type (spans of 4-8 bytes)"
    assert np.array_equal(Y.astype(bool), ((S == 1).any(1)))


@pytest.mark.parametrize("case", list(pg.CASES))
def test_weights_match_the_golden_fingerprints(case):
    if os.environ.get("PBTRAIN_GOLDEN_SKIP") == "1":
        pytest.skip("PBTRAIN_GOLDEN_SKIP=1: CPU kernels can differ between processors (compared on the recording "
                    "platform before every release)")
    g = _golden()
    ok, here = _same_platform(g["meta"])
    if not ok:
        pytest.skip(f"fingerprints recorded on {g['meta']}, this is {here}")
    sha, keys = _run(case)
    assert len(keys) == g["cases"][case]["n_tensors"]
    assert sha == g["cases"][case]["sha"], f"{case}: the trainer no longer reproduces the reference weights"


def test_training_is_deterministic():
    """Twice the same run -> the same bits (holds on every platform)."""
    assert _run("one_type_with_ignore") == _run("one_type_with_ignore")


def test_every_case_is_in_the_golden_file():
    assert set(_golden()["cases"]) == set(pg.CASES)
