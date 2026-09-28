"""The verticals against the core: the contract (shapes, typed masks with IGNORE, Y), determinism, the refusal of keys
that would do nothing, and the factory's data and rival stages on generated look-alike resources. The typed vertical
(`pii_spans`, experimental) here; the label-agnostic one (`pii_mask`, the release policy) in test_pii_mask.py."""
import os

import numpy as np
import pytest
import yaml
from pbtrain.data.leakage import split_overlap
from pbtrain.model.decode import IGNORE, window_class
from pbtrain.verticals import get

from pii_testkit import LABELS, MASK_LABELS, RECIPES, build, data_block, digest


def test_registered_with_both_policies(pii_env):
    v = get("pii_spans")
    assert list(v.entities) == LABELS and v.n_entities == 12 and v.needs_corpus
    m = get("pii_mask")
    assert list(m.entities) == MASK_LABELS and m.n_entities == 1 and m.needs_corpus
    ex = pii_env.specialist.export
    assert ex["profile"] == "redact" and ex["window"] == {"size": 512, "stride": 384, "min": 1}
    assert "labels" not in ex, "the type names are each recipe's `output_schema.labels`"
    assert get("secrets_synthetic") is not None                   # `uses: [secrets-code]` loaded its generator


def test_windows_meet_the_core_contract(pii_env):
    v, t, w = build(pii_env, data_block())
    t.check(128, 12)
    for X, Y, S in (t.train, t.select, t.test):
        assert X.dtype == np.uint8 and S.dtype == np.uint8 and X.shape == S.shape and X.shape[1] == 128
        assert set(np.unique(S).tolist()) <= set(range(13)) | {IGNORE}
        assert np.array_equal(np.asarray(Y).astype(bool), window_class(S).astype(bool))
    S = t.train[2]
    assert (S == IGNORE).any(), "the public documents carry labels outside the taxonomy: IGNORE bytes"
    assert len({int(x) for x in np.unique(S) if 0 < x < IGNORE}) >= 6
    assert 0.02 <= float((t.train[1] == 0).mean()) <= 0.6
    assert sorted(w.unread()) == ["control", "format", "path", "split_policy.unit"]     # the ones the core reads
    v.guards(w, t)
    man = v.manifest(w, t)
    assert man["k"] == 12 and man["n_labels"] == 49 and set(man["bytes_per_type_train"]) == set(LABELS)
    assert man["policy"] == "typed" and man["lookalikes"] == "text"
    assert man["resources"]["catalogs"]["missing"] == [] and len(man["resources"]["lists"]) == 12
    assert split_overlap(t.train[0], t.test[0], v.overlap_guard) == 0


def test_same_seed_same_windows_other_seed_other_windows(pii_env):
    _, a, _ = build(pii_env, data_block())
    _, b, _ = build(pii_env, data_block())
    for pa, pb in zip((a.train, a.select, a.test), (b.train, b.select, b.test)):
        assert all(np.array_equal(x, y) for x, y in zip(pa, pb))
    _, c, _ = build(pii_env, data_block(split_policy=dict(unit="corpus_range", seed=1)))
    assert not np.array_equal(a.train[0], c.train[0])


# The generator is LOCKED: this digest of a small build on the generated resources (pii_lookalike.py) changes only when
# the data of every recipe would change. Update it only together with a new recipe version, and say why.
LOCKED_DIGEST = "8e5541dd6e7a09035fa584a84173ac276767df0570e6b0e1d6411ea6396649fe"


def test_the_generator_is_locked(pii_env):
    _, t, _ = build(pii_env, data_block())
    assert digest(t) == LOCKED_DIGEST


def test_the_splits_draw_on_disjoint_corpus_ranges_and_documents(pii_env):
    v, t, _ = build(pii_env, data_block(mix=dict(code=1.0, data=0, prose=0, email=0, log=0), public_train=None,
                                        n_train=48, n_eval=16, empty_frac=0.0))
    data = open(pii_env.corpus, "rb").read()
    chunks = lambda X: {bytes(r[i:i + 32]) for r in X for i in range(0, 96, 32)}
    shared = chunks(t.train[0]) & chunks(t.test[0])
    assert not {c for c in shared if data.count(c) == 1}, "a real 32-byte stretch of the corpus in train AND test"
    docs = pii_env.plugin._load_public(os.path.join(pii_env.data, "pii", "public_train"), None)[0]
    parts = {p: {i for i in docs.ids if pii_env.plugin._part_of(i) == p} for p in ("train", "select", "test")}
    assert parts["train"] and parts["select"] and parts["test"]
    assert not (parts["train"] & parts["test"]) and not (parts["train"] & parts["select"])


@pytest.mark.parametrize("over,message", [
    (dict(mix=dict(code=0.5, emial=0.1)), "unknown scene"),
    (dict(lists="test"), "held-out half"),
    (dict(collapse=True), "format: pii_mask"),
    (dict(collapse=False), "format: pii_mask"),
    (dict(lookalikes="ignore"), "belongs to `format: pii_mask`"),
    (dict(mix=dict(public=0.0)), "would do nothing"),
    (dict(public_train=None), "needs the labelled public documents"),
    (dict(empty_frac=1.5), "must be in"),
    (dict(mix=dict(code=0, data=0, prose=0, email=0, log=0, public=0), public_train=None), "every scene off"),
])
def test_keys_that_would_do_nothing_or_harm_are_refused(pii_env, over, message):
    block = data_block(**over)
    if block.get("public_train") is None:
        block.pop("public_train")
    with pytest.raises(ValueError, match=message):
        build(pii_env, block)


def recipe_file(tmp_path, name, train=None, **data):
    """A recipe of recipes/ on the test resources: its `data:` overridden, its exams left out (no fingerprints here)."""
    t = yaml.safe_load(open(os.path.join(RECIPES, name), encoding="utf-8"))
    t["data"].update(data)
    t["train"].update(train or {})
    t.pop("exams", None)
    p = tmp_path / name
    p.write_text(yaml.safe_dump(t), encoding="utf-8")
    return str(p)


TYPED_SMALL = dict(path="${PUREBYTE_DATA}/corpora/smoke-context.bin", seq_len=128,
                   public_train=dict(dir="${PUREBYTE_DATA}/pii/public_train"))


def test_the_factory_builds_the_typed_data_and_scores_the_rival(pii_env, tmp_path):
    from pbtrain.factory import spec as S
    from pbtrain.factory.stages import Context, run
    p = recipe_file(tmp_path, "experimental-typed.yaml", train=dict(seeds=[0], size="mini64", steps=10, decide=False),
                    n_train=96, n_eval=48, **TYPED_SMALL)
    out = run(S.load(p), ["data", "baselines"], str(tmp_path / "runs"), Context(log=lambda *a: None))
    d, b = out["data"], out["baselines"]
    assert d["vertical"] == "pii_spans" and d["manifest"]["k"] == 12 and d["n_train"] == 96
    assert 0.0 <= b["rival_f1"] <= 1.0 and b["detail"]["min_score"] in pii_env.plugin.RULE_SCORE_FLOORS
    assert set(b["detail"]["per_type_test"]) == set(LABELS)


def test_a_declared_key_that_nobody_reads_fails_the_data_stage(pii_env, tmp_path):
    from pbtrain.factory import spec as S
    from pbtrain.factory.chain import StageError
    from pbtrain.factory.stages import Context, run
    with pytest.raises(StageError, match="never read"):
        run(S.load(recipe_file(tmp_path, "smoke-cpu.yaml", n_train=32, n_eval=16, empty_fraction=0.2)), ["data"],
            str(tmp_path / "runs"), Context(log=lambda *a: None))
