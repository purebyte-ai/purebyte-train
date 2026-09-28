"""The label-agnostic vertical `pii_mask` (the release policy: one type, PII): the contract, determinism, the locked
data of the release recipe and of its structured-forms variant, the look-alikes as IGNORE, the refusal of keys that
would do nothing, the recipes and their variants, and the factory's data and rival stages."""
import copy
import os

import numpy as np
import pytest
import yaml
from pbtrain.data.leakage import split_overlap
from pbtrain.model.decode import IGNORE, window_class

from pii_testkit import MASK_LABELS, NO_SCENE, RECIPES, build, digest, forms_block, mask_block

# The data are LOCKED, as the typed generator's (test_pii_plugin.py): digests of small builds on the generated
# resources (pii_lookalike.py), of the release recipe's data block and of its structured-forms variant. They change only
# when the data of every recipe would change (the full-scale arrays of the release recipe were checked bit for bit
# against the code the released model's data were built with). Update them only together with a new recipe version,
# and say why.
LOCKED_MASK = "d1cb69dcf5528173b8a7ac040bfa3afd57d153add076ebe10d8589172ddace2f"
LOCKED_FORMS_IGNORE = "1f690f7728b6c27325c6356b7b4b31f3780c95c691aa54af1054586443993bc0"


def test_windows_meet_the_core_contract(pii_env):
    v, t, w = build(pii_env, mask_block())
    t.check(128, 1)
    for X, Y, S in (t.train, t.select, t.test):
        assert X.dtype == np.uint8 and S.shape == X.shape and X.shape[1] == 128
        assert set(np.unique(S).tolist()) <= {0, 1}, "public documents only: every source label is PII, none ignored"
        assert np.array_equal(np.asarray(Y).astype(bool), window_class(S).astype(bool))
    assert (t.train[2] == 1).any()
    assert sorted(w.unread()) == ["control", "format", "path", "split_policy.unit"]     # the ones the core reads
    v.guards(w, t)
    man = v.manifest(w, t)
    assert man["k"] == 1 and man["n_labels"] == 5 and man["types"] == MASK_LABELS and man["policy"] == "label-agnostic"
    assert set(man["bytes_per_type_train"]) == {"PII"} and man["ignored_bytes_train"] == 0
    assert man["stats"]["train"]["scenes"] == {"public": 64}
    assert split_overlap(t.train[0], t.test[0], v.overlap_guard) == 0


def test_every_source_label_is_personal_data(pii_env):
    """The public documents of the test resources carry names, e-mail, phone, SSN and city labels: the typed policy
    ignores the last two, the label-agnostic one masks all five."""
    plugin = pii_env.plugin
    d = os.path.join(pii_env.data, "pii", "public_train")
    typed = plugin._load_public(d, None, collapse=False)[0]
    agnostic = plugin._load_public(d, None, collapse=True)[0]
    assert typed.ids == agnostic.ids
    t_ign = sum(1 for _, sp in typed.docs for *_, t in sp if t == 254)
    a_ign = sum(1 for _, sp in agnostic.docs for *_, t in sp if t == 254)
    assert t_ign > 0 and a_ign == 0
    assert all(t == 0 for _, sp in agnostic.docs for *_, t in sp)
    covered = lambda docs: sum(b - a for _, sp in docs for a, b, _ in sp)
    assert covered(agnostic.docs) == covered(typed.docs), "the same bytes: some typed, some ignored -> all PII"


def test_same_seed_same_windows(pii_env):
    _, a, _ = build(pii_env, mask_block())
    _, b, _ = build(pii_env, mask_block())
    assert digest(a) == digest(b)
    _, c, _ = build(pii_env, mask_block(split_policy=dict(unit="corpus_range", seed=1)))
    assert digest(a) != digest(c)


def test_the_data_are_locked(pii_env):
    _, t, _ = build(pii_env, mask_block())
    _, f, _ = build(pii_env, forms_block())
    assert (digest(t), digest(f)) == (LOCKED_MASK, LOCKED_FORMS_IGNORE)


def test_lookalikes_ignore_only_turns_outside_bytes_of_generated_windows_into_ignore(pii_env):
    """Same seed, `text` against `ignore`: the same bytes, the same PII bytes; only bytes outside every span change,
    and only into IGNORE; the public windows are untouched."""
    v, plain, _ = build(pii_env, forms_block(lookalikes="text"))
    w, ign, _ = build(pii_env, forms_block(lookalikes="ignore"))
    changed = 0
    for (Xa, Ya, Sa), (Xb, Yb, Sb) in zip((plain.train, plain.select, plain.test), (ign.train, ign.select, ign.test)):
        assert np.array_equal(Xa, Xb) and np.array_equal(Ya, Yb)
        assert np.array_equal(Sa == 1, Sb == 1)
        diff = Sa != Sb
        assert (Sa[diff] == 0).all() and (Sb[diff] == IGNORE).all()
        changed += int(diff.sum())
    st = w._built["stats"]["train"]
    assert changed > 0 and st["lookalike_ignored_bytes"] > 0
    assert set(np.unique(ign.train[2]).tolist()) <= {0, 1, IGNORE}
    assert v._built["stats"]["train"]["scenes"] == st["scenes"]


def test_crop_at_is_the_generator_crop():
    from piigen import generator as GN
    from piigen.lookalikes import crop_at
    g = np.random.default_rng(3)
    for i in range(400):
        n = int(g.integers(20, 900))
        data = bytes(g.choice([10, 32, 97, 98, 99], n))
        spans = sorted({(int(a), int(a) + int(g.integers(1, 30)), int(g.integers(0, 3)))
                        for a in g.integers(-5, n, int(g.integers(0, 6)))})
        anchor = None if i % 3 else (int(g.integers(0, n)), int(g.integers(0, n)))
        if anchor:
            anchor = (min(anchor), max(anchor))
        a = GN._crop(data, spans, 128, np.random.default_rng(i), anchor)
        b = crop_at(data, spans, 128, np.random.default_rng(i), anchor)
        assert a == b[:2] and 0 <= b[2]


@pytest.mark.parametrize("over,message", [
    (dict(collapse=True), "its own format"),
    (dict(lookalikes="ignore"), "would do nothing"),
    (dict(lookalikes="yes"), "text | ignore"),
    (dict(lists="test"), "held-out half"),
    (dict(mix=dict(public=0.0, code=0.0, data=0.0, prose=0.0, email=0.0, log=0.0)), "every scene off"),
])
def test_keys_that_would_do_nothing_or_harm_are_refused(pii_env, over, message):
    with pytest.raises(ValueError, match=message):
        build(pii_env, mask_block(**over))


def test_the_rival_scores_one_type(pii_env):
    v, t, _ = build(pii_env, mask_block(n_eval=48))
    r = v.span_rival(t.select, t.test)
    assert set(r["per_type_test"]) == {"PII"} and 0.0 <= r["span_f1"] <= 1.0
    assert r["min_score"] in pii_env.plugin.RULE_SCORE_FLOORS


def _recipe(name):
    return yaml.safe_load(open(os.path.join(RECIPES, name), encoding="utf-8"))


# the variants the release recipe was measured against (README.md, "The variants"): the lines that change, and nothing
# else. The pool-only variant writes the whole mix: a scene left out of `mix` keeps its default share.
POOL_ONLY_MIX = dict(code=0.0, data=0.0, prose=0.0, email=0.0, log=0.0, public=1.0)
VARIANTS = {
    "capacity": {("train", "size"): "tiny_n18"},
    "pool-only": {("data", "mix"): POOL_ONLY_MIX, ("data", "lookalikes"): "text"},
}


@pytest.mark.parametrize("variant", ["release"] + sorted(VARIANTS))
def test_the_release_recipe_and_its_variants_validate(tmp_path, variant):
    from pbtrain.factory import spec as S
    from pbtrain.model.config import ModelConfig
    t = _recipe("pii-1.0.0.yaml")
    for (block, key), value in VARIANTS.get(variant, {}).items():
        assert key in t[block], f"the template has no line `{block}.{key}` for the {variant} variant"
        t[block][key] = copy.deepcopy(value)
    p = tmp_path / "r.yaml"
    p.write_text(yaml.safe_dump(t), encoding="utf-8")
    spec = S.load(str(p))
    assert spec["output_schema"]["labels"] == ["PII"] and spec["data"]["format"] == "pii_mask"
    assert spec["train"]["seeds"] == [0, 1, 2, 3, 4, 5] and spec["train"]["decide"] is True
    assert {c["metric"] for c in spec["criteria"]["exams"]} == {"f2_mean6_seeds_mean", "f2_ai4privacy_en_seeds_mean",
                                                                  "precision_mean6_seeds_mean"}
    cfg = ModelConfig.preset(spec["train"]["size"])
    assert (cfg.d_model, cfg.ngram_orders, cfg.ngram_dim) == (256, (2, 3, 4), 64)
    assert (cfg.n_layers, cfg.ngram_buckets) == ((8, 262144) if variant == "capacity" else (4, 32768))


def test_the_pool_only_variant_is_public_documents_only(pii_env):
    """The pool-only variant as README.md writes it: every window from the public documents. The shorthand
    `mix: {public: 1.0}` would not be: a partial `mix` keeps the default share of every scene it leaves out, so it
    would still draw about half of the windows from generated scenes, prose and e-mail included."""
    from piigen.generator import DEFAULT_MIX
    t = _recipe("pii-1.0.0.yaml")["data"]
    for (block, key), value in VARIANTS["pool-only"].items():
        t[key] = copy.deepcopy(value)
    t.update(path="${PUREBYTE_DATA}/corpora/smoke-context.bin", n_train=48, n_eval=16, seq_len=128,
             public_train=dict(dir="${PUREBYTE_DATA}/pii/public_train"))
    v, p, w = build(pii_env, t)
    man = v.manifest(w, p)
    assert man["mix"] == POOL_ONLY_MIX and man["stats"]["train"]["scenes"] == {"public": 48}
    shorthand = dict(DEFAULT_MIX, public=1.0)
    assert sum(x for k, x in shorthand.items() if k != "public") / sum(shorthand.values()) > 0.45


def test_the_release_recipe_needs_no_sentence_list(pii_env):
    """Only the prose and e-mail scenes draw a sentence: without sentences_{es,en}.tsv the structured-forms data are
    the same, bit for bit, and a recipe with prose refuses to build."""
    lists = os.path.join(pii_env.data, "pii", "lists")
    names = pii_env.plugin._names
    moved = []
    try:
        for f in ("sentences_es.tsv", "sentences_en.tsv"):
            os.replace(os.path.join(lists, f), os.path.join(lists, f + ".away"))
            moved.append(f)
        names._CACHE.clear()
        v, f_, w = build(pii_env, forms_block())
        assert digest(f_) == LOCKED_FORMS_IGNORE
        assert 0.0 <= v.span_rival(f_.select, f_.test)["span_f1"] <= 1.0       # the rival reads no sentence either
        with pytest.raises(ValueError, match="sentences_es.tsv"):
            build(pii_env, mask_block(mix=dict(NO_SCENE, prose=0.5, public=0.5)))
    finally:
        for f in moved:
            os.replace(os.path.join(lists, f + ".away"), os.path.join(lists, f))
        names._CACHE.clear()


def test_the_other_recipes_validate():
    from pbtrain.factory import spec as S
    smoke = S.load(os.path.join(RECIPES, "smoke-cpu.yaml"))
    assert smoke["data"]["format"] == "pii_mask" and smoke["output_schema"]["labels"] == ["PII"]
    typed = S.load(os.path.join(RECIPES, "experimental-typed.yaml"))
    assert typed["data"]["format"] == "pii_spans" and len(typed["output_schema"]["labels"]) == 12
    assert typed["name"] != "pii" and typed["specialist"] == "pii"


def test_the_factory_builds_the_smoke_data_and_scores_the_rival(pii_env, tmp_path):
    from pbtrain.factory import spec as S
    from pbtrain.factory.stages import Context, run
    from test_pii_plugin import recipe_file
    out = run(S.load(recipe_file(tmp_path, "smoke-cpu.yaml", n_train=96, n_eval=48)), ["data", "baselines"],
              str(tmp_path / "runs"), Context(log=lambda *a: None))
    d, b = out["data"], out["baselines"]
    assert d["vertical"] == "pii_mask" and d["manifest"]["k"] == 1 and d["n_train"] == 96
    assert set(b["detail"]["per_type_test"]) == {"PII"}


def test_the_release_recipe_data_block_builds_on_small_shapes(pii_env, tmp_path):
    """The release recipe's own data block (every key as written) on the test resources, with small counts: every key
    is read, one type, the structured-forms mix (public documents plus generated records, logs and code, look-alikes
    as IGNORE)."""
    t = _recipe("pii-1.0.0.yaml")["data"]
    t.update(path="${PUREBYTE_DATA}/corpora/smoke-context.bin", n_train=48, n_eval=16, seq_len=128,
             public_train=dict(dir="${PUREBYTE_DATA}/pii/public_train"))
    v, p, w = build(pii_env, t)
    assert sorted(w.unread()) == ["control", "format", "path", "split_policy.unit"]  # no code window at this size
    assert v.manifest(w, p)["mix"]["public"] == 0.8 and set(np.unique(p.train[2]).tolist()) <= {0, 1, IGNORE}
