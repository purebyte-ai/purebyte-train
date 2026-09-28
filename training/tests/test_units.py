"""Unit tests of the pieces the golden and smoke tests do not isolate."""
import copy
import os

import numpy as np
import pytest
import torch

import golden_cases as gc
from pbtrain.data import fingerprint as fp
from pbtrain.eval.operating_point import OperatingPointError, choose
from pbtrain.export.pbformat import dequant_ngram_int4, quant_ngram_int4
from pbtrain.factory import criteria as C
from pbtrain.factory.spec import Watched
from pbtrain.model.decode import (IGNORE, IGNORE_LABEL, label_bias, mask_spans, mask_to_bioes, spans, viterbi,
                                  viterbi_batch, window_class)
from pbtrain.model.layers.bitpack import pack2, unpack2
from pbtrain.train.loop import SpanTask, train_spans
from pbtrain.train.recipe import RecipeError, SpanRecipe, from_train_block


def test_bioes_one_entity_merges_contiguous_bytes():
    row = np.array([0, 1, 1, 1, 0, 1, 0, 1, 1])
    assert mask_to_bioes(row).tolist() == [0, 1, 2, 3, 0, 4, 0, 1, 3]
    assert mask_spans(row) == [(1, 4, 0), (5, 6, 0), (7, 9, 0)]


def test_bioes_typed_masks_split_by_value_and_ignore():
    row = np.array([0, 1, 1, 2, 2, 0, IGNORE, 3])
    lab = mask_to_bioes(row, 3)
    assert lab.tolist() == [0, 1, 3, 5, 7, 0, IGNORE_LABEL, 12]
    assert window_class(row[None]).tolist() == [1]
    assert window_class(np.array([[0, IGNORE, 0]])).tolist() == [0]


def test_ignored_bytes_are_never_gold_spans():
    row = np.array([0, IGNORE, IGNORE, 0, 2, 2, IGNORE, 1])
    assert mask_spans(row, 2) == [(4, 6, 1), (7, 8, 0)]


def test_per_type_operating_point_moves_only_what_helps():
    from pbtrain.eval.operating_point import per_type
    target = [0.5, -0.5, 0.5]                          # the measure peaks at base + (0, -1, 0)
    vec, f = per_type(lambda v: -sum((a - b) ** 2 for a, b in zip(v, target)), 0.5, 3)
    assert vec == [0.5, -0.5, 0.5] and f == 0.0


def test_viterbi_batch_equals_viterbi():
    g = np.random.default_rng(0)
    lp = np.log(g.dirichlet(np.ones(9), size=(6, 40))).astype(np.float32)
    lb = label_bias(2, 0.5)
    batch = viterbi_batch(lp, 2, lb)
    for k in range(6):
        assert (batch[k] == viterbi(lp[k], 2, lb)).all()
        assert spans(batch[k], 2) == spans(viterbi(lp[k], 2, lb), 2)


def test_ternary_pack_roundtrip():
    t = torch.from_numpy(np.random.default_rng(1).integers(-1, 2, (5, 13)).astype(np.float32))
    assert (unpack2(pack2(t), 13) == t.numpy()).all()


def test_int4_tables_roundtrip_within_a_step():
    W = np.random.default_rng(2).normal(size=(64, 16)).astype(np.float32)
    packed, scale = quant_ngram_int4(W)
    back = dequant_ngram_int4(packed, scale)
    assert np.abs(back - W).max() <= scale.max() / 2 + 1e-6


def test_operating_point_refuses_the_edge_and_records_flat_sweeps():
    with pytest.raises(OperatingPointError, match="EDGE"):
        choose(lambda b: dict(span_f1=b, precision=b, recall=b), grid=(0.0, 1.0, 2.0))
    bias, f1, sweep = choose(lambda b: dict(span_f1=0.5, precision=0.5, recall=0.5), grid=(0.0, 1.0, 2.0))
    assert bias == 0.0 and "flat" in sweep[-1]["warning"]


def test_criteria_validate_and_evaluate():
    crit = dict(written="2026-01-01", synthetic=[dict(metric="test_f1_mean", op=">=", value=0.9)],
                exams=[dict(exam="x", metric="f1", op=">=", value=0.5)])
    C.validate(crit)
    v = C.evaluate(crit, synthetic=dict(test_f1_mean=0.95), exams={})
    assert [x["verdict"] for x in v] == ["PASS", "PENDING"]
    assert C.summary(v)["overall"] == "PENDING"
    with pytest.raises(C.CriteriaError):
        C.validate(dict(synthetic=[dict(metric="test_f1_mean", op=">=", value=1)]))    # no `written`


def test_recipes_refuse_unknown_and_unsupported_keys():
    with pytest.raises(RecipeError, match="unknown keys"):
        from_train_block(dict(seeds=[0], size="nano", steps=1, lr=1e-3, batch=2, step=5))
    with pytest.raises(RecipeError, match="EMA"):
        from_train_block(dict(seeds=[0], size="nano", steps=1, lr=1e-3, batch=2, ema=0.999))
    r = from_train_block(dict(seeds=[0], size="nano", steps=1, lr=1e-3, batch=2, mode="bidirectional"))
    assert r.direction == "hydra"


def test_watched_reports_unread_leaves():
    w = Watched(dict(a=1, b=dict(c=2, d=3), e=4))
    w["a"]
    w["b"]["c"]
    assert sorted(w.unread()) == ["b.d", "e"]


def test_fingerprints_ignore_padding_and_find_shared_regions():
    g = np.random.default_rng(3)
    exam = g.integers(0, 256, 100_000, dtype=np.uint8)
    f = fp.fingerprints(exam)
    X = g.integers(0, 256, (4, 512), dtype=np.uint8)
    X[1] = exam[5000:5512]
    X[2, :] = 0
    assert fp.window_hits(X, f).tolist() == [False, True, False, False]
    assert len(fp.fingerprints(np.zeros(10_000, np.uint8))) == 0


def test_k_entity_training_runs_and_heads_have_1_plus_4k_labels():
    X, Y, S = gc.Task().train
    S3 = S.copy()
    S3[::4][S3[::4] == 1] = 3                                  # some spans are of type 3
    task = SpanTask((X, window_class(S3).astype(np.int64), S3), None, None)
    m, th, ch = train_spans(task, SpanRecipe(size="mini64", steps=3, batch=8, lr=1e-3, warmup=1, n_entities=3))
    assert th.proj.out_features == 13 and th.n_entities == 3


def test_parallel_seeds_are_bit_identical_to_serial(tmp_path):
    from pbtrain.train import checkpoint
    from pbtrain.train.seeds import train_seeds
    X, Y, S = gc.Task().train
    npz = str(tmp_path / "partitions.npz")
    np.savez(npz, Xtr=X, Ytr=Y, Str=S)
    r = SpanRecipe(size="mini64", steps=4, batch=8, lr=1e-3, warmup=1)
    torch.set_num_threads(1)                                   # workers inherit the parent's thread count
    par = train_seeds(npz, r, [0, 1], device="cpu", parallel=2, cache_dir=str(tmp_path / "par"), log=lambda *a: None)
    ser = train_seeds(npz, r, [0, 1], device="cpu", parallel=1, cache_dir=str(tmp_path / "ser"), log=lambda *a: None)
    for j in par:
        a, b = checkpoint.load(par[j]), checkpoint.load(ser[j])
        assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"])
    again = train_seeds(npz, r, [0, 1], device="cpu", parallel=1, cache_dir=str(tmp_path / "ser"), log=lambda *a: None)
    assert again == ser                                        # served from the content cache
    prov = checkpoint.load(ser[(0, False)])["provenance"]
    assert prov["device"] == "cpu" and prov["torch_threads"] == 1 and prov["trainer_digest"]
    with pytest.raises(ValueError, match="repeat a seed"):
        train_seeds(npz, r, [0, 0], device="cpu", cache_dir=str(tmp_path / "dup"), log=lambda *a: None)


# ------------------------------------------------------------------------------------------- task validation
RECIPE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "specialists",
                      "secrets-code", "recipes", "smoke-cpu.yaml")


def _task(tmp_path, edit):
    import yaml
    from pbtrain.factory import spec as S
    with open(RECIPE, encoding="utf-8") as f:
        t = yaml.safe_load(f)
    edit(t)
    p = tmp_path / "t.yaml"
    p.write_text(yaml.safe_dump(t), encoding="utf-8")
    return lambda: S.load(str(p))


@pytest.mark.parametrize("edit,message", [
    (lambda t: t["train"].update(seeds=[0, 0, 0]), "repeats"),
    (lambda t: t["train"].update(seeds=[0, 1, True]), "whole numbers"),
    (lambda t: t["train"].update(seeds=3), "list"),
    (lambda t: t["train"].update(steps="12000"), "is a string"),
    (lambda t: t["train"].update(lr="3e-4"), "YAML reads"),
    (lambda t: t["train"].update(clip=0), "must be > 0"),
    (lambda t: t["train"].update(span_weight=0.5), "must be >= 1.0"),
    (lambda t: t["train"].update(beta2=1.0), "must be < 1.0"),
    (lambda t: t["train"].update(size="nano_typo"), "unknown `train.size"),
    (lambda t: t["train"].update(batch=10_000), "larger than `data.n_train"),
    (lambda t: t["train"].update(released_seed=7), "not one of the trained seeds"),
    (lambda t: t["train"].update(released_seed_reason="why"), "without `released_seed`"),
    (lambda t: t["train"].update(real_frac=0.25), "without `real_windows`"),
    (lambda t: t["train"].update(guard_every=5), "without `guard_mode`"),
    (lambda t: t["train"].update(lora_rank=4), "without `adapter: lora`"),
    (lambda t: t.update(baselines=["classic"]), "unknown top-level keys"),
    (lambda t: t.update(engine={"threads": 1, "kernel": "avx2"}), "the only key is `threads`"),
    (lambda t: t.update(engine={"threads": 0}), ">= 1"),
    (lambda t: t.update(name="../escape"), "names folders"),
    (lambda t: t.update(version="1.0"), "semantic version"),
    (lambda t: t.update(specialist="../../evil"), "folder of specialists"),
    (lambda t: t.update(exams=["a", "a"]), "twice"),
    (lambda t: t["output_schema"].update(per_type_bias=True), "one label"),
    (lambda t: t["output_schema"].update(bias_policy="precision"), "bias_policy"),
    (lambda t: t["output_schema"].update(min_select_recall=1.5), "(0, 1]"),
    (lambda t: t["output_schema"].update(reject_label=None), "not supported"),
    (lambda t: t["data"].update(n_train="256"), "whole number"),
    (lambda t: t["criteria"]["synthetic"][0].update(value="0.1"), "must be a number"),
])
def test_the_task_is_validated_before_anything_is_built(tmp_path, edit, message):
    import re
    from pbtrain.factory.chain import StageError
    with pytest.raises(StageError, match=re.escape(message)):
        _task(tmp_path, edit)()


def test_the_released_and_smoke_recipes_still_validate():
    from pbtrain import paths
    from pbtrain.factory import spec as S
    names = []
    for sp in ("secrets-code", "secrets-bin", "pii"):
        folder = os.path.join(paths.SPECIALISTS_SRC, sp, "recipes")
        for fn in sorted(os.listdir(folder)):
            S.load(os.path.join(folder, fn))
            names.append(fn)
    assert "secrets-code-1.0.0.yaml" in names and "pii-1.0.0.yaml" in names and len(names) >= 8


def test_the_task_digest_ignores_comments_and_description(tmp_path):
    from pbtrain.factory import spec as S
    with open(RECIPE, encoding="utf-8") as f:
        text = f.read()
    a = tmp_path / "a.yaml"
    b = tmp_path / "b.yaml"
    c = tmp_path / "c.yaml"
    a.write_text(text, encoding="utf-8")
    b.write_text("# another comment\n" + text + "description: free text\n", encoding="utf-8")
    c.write_text(text.replace("steps: 150", "steps: 151"), encoding="utf-8")
    da, db, dc = (S.load(str(p))["_digest"] for p in (a, b, c))
    assert da == db != dc


def test_experimental_options_are_the_ones_no_release_uses():
    r = SpanRecipe(out_lr_mult=0.25, real_windows="x.npz", real_frac=0.25, real_mode="spans")
    assert r.experimental_in_use() == []
    assert SpanRecipe(real_windows="x.npz", real_frac=0.25, real_mode="distill", beta2=0.99).experimental_in_use() \
        == ["beta2", "real_mode: distill"]


# ------------------------------------------------------------------------------------------- specialist.yaml
@pytest.mark.parametrize("edit,message", [
    (lambda c: c["export"].update(ngram_bit=4), "unknown keys in `export`"),
    (lambda c: c["release"].update(ensamble=3), "`release` takes"),
    (lambda c: c["release"].update(vote=4), "larger than"),
    (lambda c: c.update(exam=["x"]), "unknown keys"),
    (lambda c: c["export"].update(ngram_bits=8), "4 or 32"),
    (lambda c: c.update(plugin="../outside.py"), "inside the specialist's folder"),
    (lambda c: c.update(name="other-name"), "differs from its folder"),
    (lambda c: c["export"]["window"].update(stride=0), ">= 1"),
])
def test_specialist_yaml_is_validated(edit, message):
    import yaml
    from pbtrain import paths
    from pbtrain.verticals import SpecialistError, validate_config
    with open(os.path.join(paths.SPECIALISTS_SRC, "secrets-code", "specialist.yaml"), encoding="utf-8") as f:
        c = yaml.safe_load(f)
    validate_config("secrets-code", copy.deepcopy(c))
    edit(c)
    with pytest.raises(SpecialistError, match=message):
        validate_config("secrets-code", c)


def test_every_specialist_yaml_validates():
    from pbtrain.verticals import list_specialists, load_specialist
    for name in list_specialists():
        sp = load_specialist(name)
        assert sp.exams and 1 <= sp.vote <= sp.ensemble


def test_a_checkout_is_required_for_the_source_folders(tmp_path):
    from pbtrain.paths import require_source_tree
    with pytest.raises(FileNotFoundError, match="checkout of purebyte-train"):
        require_source_tree(str(tmp_path / "nowhere"))


# ------------------------------------------------------------------------------------------- leakage between splits
def _parts(train, select, test):
    from pbtrain.verticals import Partitions
    z = lambda X: (X, np.zeros(len(X), np.uint8), np.zeros_like(X))
    return Partitions(z(train), z(select), z(test))


def test_the_split_checks_compare_select_too_and_name_the_pair():
    from pbtrain.data import leakage
    g = np.random.default_rng(0)
    tr, se, te = (g.integers(0, 256, (400, 240), dtype=np.uint8) for _ in range(3))
    out = leakage.check_splits(_parts(tr, se, te), "chunks")
    assert out["enforced"] == {"train-select": 0, "train-test": 0, "select-test": 0}
    se2 = se.copy()
    se2[5, 120:240] = tr[7, 120:240]                           # the LAST whole chunk of a 240-byte window
    with pytest.raises(leakage.LeakError, match="between train and select"):
        leakage.check_splits(_parts(tr, se2, te), "chunks")
    te2 = te.copy()
    te2[:2] = se[:2]
    with pytest.raises(leakage.LeakError, match="identical windows between select and test"):
        leakage.check_splits(_parts(tr, se, te2), "windows")
    te3 = te.copy()
    te3[0] = tr[0]                                             # 1 of 400 is above 0.1 % of 400
    with pytest.raises(leakage.LeakError, match="more than 0.1% of test"):
        leakage.split_overlap(tr, te3, "windows")


def test_the_fingerprint_measurement_sees_shared_content_at_any_offset():
    from pbtrain.data import leakage
    g = np.random.default_rng(1)
    corpus = g.integers(32, 127, 200_000, dtype=np.uint8)
    tr = np.stack([corpus[o:o + 512] for o in range(0, 100 * 600, 600)])
    te = np.stack([corpus[o + 37:o + 37 + 512] for o in range(0, 20 * 600, 600)])     # shifted, not chunk-aligned
    se = g.integers(32, 127, (20, 512), dtype=np.uint8)
    m = leakage.fingerprint_overlap(dict(train=tr, select=se, test=te))
    assert m["test_in_train"]["windows"] >= 18 and m["select_in_train"]["windows"] == 0
    assert leakage.split_overlap(tr, te, "chunks") == 0         # the aligned chunks miss it: measured, not enforced


def test_the_exam_registry_and_corpus_gaps(tmp_path):
    from pbtrain.data import leakage
    d = tmp_path / "exams" / "fingerprints"
    d.mkdir(parents=True)
    content = np.random.default_rng(2).integers(0, 256, 20_000, dtype=np.uint8)
    fp.save(str(d / "e1.npz"), fp.fingerprints(content), dict(name="e1", role="dev"))
    reg = leakage.registry(str(tmp_path / "exams"))
    assert reg == {"e1": dict(role="dev", fp_sha256=leakage.fp_digest(fp.fingerprints(content)))}
    report = dict(exams={"e1": dict(fp_sha256=reg["e1"]["fp_sha256"])})
    man = dict(exclusion=dict(content=dict(sets={"e1": dict(fp_sha256="0" * 64)})))
    assert leakage.corpus_exclusion_gaps(man, report) == ["e1: its fingerprints changed since the corpus was built"]
    assert leakage.corpus_exclusion_gaps(dict(exclusion=dict(content=dict(sets={}))), report) == \
        ["e1: not excluded when the corpus was built"]
    assert leakage.corpus_exclusion_gaps(None, report) == []


# ------------------------------------------------------------------------------------------- small pieces
def test_the_gate_needs_more_than_the_spread():
    from pbtrain.eval.baselines import gate
    assert gate(0.60, 0.50, 0.10)[0] is False                  # a tie with the spread is not a win
    assert gate(0.61, 0.50, 0.10)[0] is True
    assert gate(0.50, 0.50, 0.0)[0] is False


def test_criteria_verdicts_refuse_values_that_are_not_numbers():
    crit = dict(written="2026-01-01", exams=[dict(exam="x", metric="f1", op=">=", value=0.5)])
    v = C.evaluate(crit, exams={"x": {"f1": "0.9"}})
    assert v[0]["verdict"] == "INVALID" and C.summary(v)["overall"] == "FAIL"
    with pytest.raises(C.CriteriaError, match="a number"):
        C.validate(dict(written="2026-01-01", synthetic=[dict(metric="control_f1", op="<=", value=True)]))


class NotAWeight:
    """An arbitrary pickled object: a checkpoint that holds one is refused, never executed."""


def test_checkpoints_are_loaded_without_executing_pickles(tmp_path):
    from pbtrain.train import checkpoint
    p = str(tmp_path / "evil.pt")
    torch.save({"model": {}, "payload": NotAWeight()}, p)
    with pytest.raises(ValueError, match="weights_only=True"):
        checkpoint.load(p)


def test_gguf_metadata_is_read_from_the_header_only(tmp_path):
    from pbtrain.export.gguf import read_metadata
    from pbtrain.export.pbformat import export
    m, th, ch = train_spans(gc.Task(), SpanRecipe(size="mini64", steps=1, batch=8, lr=1e-3, warmup=1), seed=0)
    p = str(tmp_path / "m.gguf")
    export(p, m.eval(), [th.eval(), ch.eval()], gating={0: 1}, labels=["secret"], operating_bias=0.5)
    whole = read_metadata(p)
    with open(p, "rb") as f:
        head = f.read(4096)
    cut = str(tmp_path / "cut.gguf")
    with open(cut, "wb") as f:
        f.write(head)                                          # the tensor data is gone, the metadata is not
    assert read_metadata(cut) == whole and whole["purebyte.head.0.operating_bias"] == pytest.approx(0.5)
