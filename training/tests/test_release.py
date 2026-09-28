"""Releases. The internal release, deterministically: a real factory chain of the secrets-code smoke recipe (three
training steps, the operating point set at bias 0, `verify` reported identical), whose `finetune` metrics are made to
pass everything but the rival margin, and a rival margin forced either way; then the
release refuses (a failed gate, too few members for the ensemble, an exam of the specialist that was not checked) or
is made (complete, read-only, never overwritten, nothing left behind by a failure), through the CLI in both orders of
the documented procedure. The public bundle: public file names, one .sha256 per file, the model card, the catalog
fragment, the private mapping line, the check of every file against the MANIFEST -- and the hygiene check that refuses
internal strings in the card or the GGUF metadata."""
import contextlib
import io
import json
import os
import stat

import pytest
import yaml

import golden_cases as gc
import smoke_data
from pbtrain import paths
from pbtrain.cli import main
from pbtrain.export.pbformat import export
from pbtrain.factory import spec as S
from pbtrain.factory.chain import StageError, sha_file
from pbtrain.factory.stages import Context, run
from pbtrain.release.package import FORBIDDEN, hygiene, internal_release, public_bundle, verify_release
from pbtrain.train.loop import train_spans
from pbtrain.train.recipe import SpanRecipe

RECIPE = os.path.join(paths.SPECIALISTS_SRC, "secrets-code", "recipes", "smoke-cpu.yaml")
QUIET = dict(log=lambda *a: None, verify_windows=6)


# ------------------------------------------------------------------------------------------------ internal release
@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("pbrelease")
    data, work = str(root / "data"), str(root / "work")
    smoke_data.make_data(data)
    with smoke_data.data_and_work(data, work):
        yield data, work


def task(folder, version, seeds=(0, 1, 2), edit=None):
    with open(RECIPE, encoding="utf-8") as f:
        t = yaml.safe_load(f)
    t["version"] = version
    t["train"].update(seeds=list(seeds), steps=3, decide=len(seeds) >= 3)
    if edit:
        edit(t)
    p = os.path.join(str(folder), f"task-{version}.yaml")
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(t, f)
    return p


def _identical(gguf, ck, X, n=None, threads=None):
    """`verify`'s comparison, reported identical without running the runtime (a three-step model may emit no span;
    the comparison itself is tested by test_factory_smoke.py and test_verify_runtime.py)."""
    n = n or len(X)
    return dict(n=n, equal=n, identical_spans=True, windows=list(range(n)), operating_bias=float(ck["bias"]),
                operating_bias_vec=ck.get("bias_vec"), runtime="runtime (not run by this test)", engine_threads=None,
                windows_with_spans=1)


def chain(p, runs, monkeypatch, passes=True, upto="report"):
    """data -> ... -> `upto` on task `p` (three training steps, the operating point set at bias 0, `verify` reported
    identical), with `eval`'s rival margin forced to `passes`: a real chain whose outcome does not depend on the
    model's quality."""
    import pbtrain.eval.baselines as rivals
    import pbtrain.factory.stages as stages
    import pbtrain.factory.verify as verify
    monkeypatch.setattr(stages, "choose", lambda measure, grid, floor, policy:
                        (0.0, 0.5, [dict(bias=0.0, f1=0.5, precision=0.5, recall=0.5)]))
    monkeypatch.setattr(verify, "compare_runtime", _identical)
    spec = S.load(p)
    run(spec, ["data", "baselines", "finetune"], runs, Context(**QUIET))
    smoke_data.fabricate_finetune_pass(runs, spec)
    monkeypatch.setattr(rivals, "gate", lambda *a: (passes, f"{'passes' if passes else 'refused'} (forced)"))
    try:
        run(spec, ["eval"], runs, Context(**QUIET))
    except StageError:
        assert not passes
    if upto != "eval":
        run(spec, ["calibrate", "export", "verify", "report"], runs, Context(**QUIET))
    return spec


def bundle_of(out):
    """The JSON the CLI prints for a public bundle (after the stages' progress lines, when stages ran)."""
    return json.loads(out[out.index("{\n"):])


def cli(*argv):
    err, out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        try:
            rc = main(["--threads", "1", *argv])
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


def test_a_refused_gate_and_a_short_ensemble_are_not_released(env, tmp_path, monkeypatch):
    runs = str(tmp_path / "runs")
    spec = chain(task(tmp_path, "0.0.10", seeds=(0,)), runs, monkeypatch, passes=False, upto="eval")
    with pytest.raises(StageError, match="gate was not passed"):
        internal_release(runs, spec)
    spec = chain(task(tmp_path, "0.0.10", seeds=(0,)), runs, monkeypatch, passes=True, upto="eval")
    with pytest.raises(StageError, match="ensemble of 3"):
        internal_release(runs, spec)
    assert not os.path.exists(os.path.join(paths.specialists_dir(), spec["name"], "0.0.10"))


def test_a_specialist_exam_that_was_not_checked_blocks_the_release(tmp_path, monkeypatch):
    data, work = str(tmp_path / "data"), str(tmp_path / "work")
    smoke_data.make_data(data, exams=("unrelated-exam",))          # the specialist's exams are NOT registered
    with smoke_data.data_and_work(data, work):
        runs = str(tmp_path / "runs")
        spec = chain(task(tmp_path, "0.0.11", seeds=(0,)), runs, monkeypatch, upto="eval")
        with pytest.raises(StageError, match="did not check them"):
            internal_release(runs, spec)


def test_the_documented_two_step_release(env, tmp_path, monkeypatch):
    """`pbtrain release TASK`, then `pbtrain release TASK --public --semver X.Y.Z` (training/docs/release.md). The
    bundles are version 1.0.0: the card template names the version its numbers were measured on."""
    monkeypatch.setenv("PUREBYTE_WORK", str(tmp_path / "work"))
    runs = str(tmp_path / "runs")
    p = task(tmp_path, "1.0.0")
    spec = chain(p, runs, monkeypatch)
    rc, out, err = cli("--work", runs, "release", p)
    assert rc == 0, err
    rel = os.path.join(paths.specialists_dir(), spec["name"], "1.0.0")
    with open(os.path.join(rel, "MANIFEST.json"), encoding="utf-8") as f:
        man = json.load(f)
    assert len(man["members"]) == 3 and man["vote"] == 2 and man["ensemble"] == 3
    assert man["task_digest"] == spec["_digest"] and man["task_sha256"] == sha_file(p)
    assert man["training"]["device"] == "cpu" and man["release_machine"]["torch_threads"] == 1
    assert man["released_seed_rule"] == "the best on select" and man["criteria"]["overall"] in ("PASS", "FAIL")
    assert set(man["files"]) == set(os.listdir(rel)) - {"MANIFEST.json"}
    assert all(not os.stat(os.path.join(rel, fn)).st_mode & stat.S_IWRITE for fn in os.listdir(rel))
    assert not [d for d in os.listdir(os.path.dirname(rel)) if ".partial" in d]
    rc, out, err = cli("--work", runs, "release", p)                  # a second internal release: refused
    assert rc == 2 and "never overwritten" in err
    rc, out, err = cli("--work", runs, "release", p, "--public", "--semver", "1.0.0")
    assert rc == 0, err
    bundle = bundle_of(out)["bundle"]
    assert sorted(os.listdir(bundle))[:2] == ["MODEL_CARD.md", "models.json.fragment"]
    assert "purebyte-secrets-code-1.0.0-e2.gguf.sha256" in os.listdir(bundle)
    # the task changed after the release: its release is not this task's
    with open(p, encoding="utf-8") as f:
        t = yaml.safe_load(f)
    t["output_schema"]["min_select_recall"] = 0.5
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(t, f)
    rc, out, err = cli("--work", runs, "release", p, "--public", "--semver", "1.0.0")
    assert rc == 2 and "another version of the task" in err
    # an altered release cannot be published
    member = os.path.join(rel, "member_1.gguf")
    os.chmod(member, stat.S_IWRITE | stat.S_IREAD)
    with open(member, "ab") as f:
        f.write(b"\0")
    with pytest.raises(StageError, match="differs from the sha256"):
        verify_release(rel)


def test_the_one_step_release_and_a_failure_that_leaves_nothing(env, tmp_path, monkeypatch):
    import pbtrain.factory.stages as stages
    monkeypatch.setenv("PUREBYTE_WORK", str(tmp_path / "work"))
    runs = str(tmp_path / "runs")
    p = task(tmp_path, "1.0.0")
    spec = chain(p, runs, monkeypatch)
    real = stages.export_checkpoint

    def broken(*a, **k):
        raise OSError("disk full (simulated)")
    monkeypatch.setattr(stages, "export_checkpoint", broken)
    with pytest.raises(OSError, match="simulated"):
        run(spec, ["release"], runs, Context(**QUIET))
    parent = os.path.join(paths.specialists_dir(), spec["name"])
    assert not os.path.exists(os.path.join(parent, "1.0.0")) and not [d for d in os.listdir(parent) if "1.0.0" in d]
    monkeypatch.setattr(stages, "export_checkpoint", real)
    rc, out, err = cli("--work", runs, "release", p, "--public", "--semver", "1.0.0")
    assert rc == 0, err
    assert os.path.exists(os.path.join(parent, "1.0.0", "MANIFEST.json")) and os.path.isdir(bundle_of(out)["bundle"])


def test_a_release_written_by_pbtrain_run_is_bundled_by_release_public(env, tmp_path, monkeypatch):
    """`pbtrain run TASK` runs every stage, `release` included; `pbtrain release TASK --public --semver X.Y.Z` then
    makes the bundle from that internal release instead of refusing it."""
    monkeypatch.setenv("PUREBYTE_WORK", str(tmp_path / "work"))
    runs = str(tmp_path / "runs")
    p = task(tmp_path, "1.0.0")
    spec = chain(p, runs, monkeypatch)
    run(spec, ["release"], runs, Context(**QUIET))
    rc, out, err = cli("--work", runs, "release", p, "--public", "--semver", "1.0.0")
    assert rc == 0 and "matches the task" in err, err
    assert len(bundle_of(out)["files"]) == 3


def test_release_options_are_checked_before_anything_runs(env, tmp_path):
    p = task(tmp_path, "0.0.5")
    for argv, msg in ((("--public",), "needs --semver"), (("--semver", "1.0.0"), "acts only with --public"),
                      (("--public", "--semver", "1.0"), "not a semantic version")):
        rc, out, err = cli("--work", str(tmp_path / "runs"), "release", p, *argv)
        assert rc != 0
    assert not os.path.exists(os.path.join(str(tmp_path / "runs"), "secrets-code-smoke", "0.0.5"))


# --------------------------------------------------------------------------------------------------- public bundle
def _fake_release(folder, name="secrets-code", info=None):
    os.makedirs(folder)
    members = []
    for i, seed in enumerate((0, 1, 2)):
        m, th, ch = train_spans(gc.Task(), SpanRecipe(size="mini64", steps=2, batch=8, lr=1e-3, warmup=1), seed=seed)
        fn = "model.gguf" if i == 0 else f"member_{i}.gguf"
        export(os.path.join(folder, fn), m.eval(), [th.eval(), ch.eval()], gating={0: 1}, labels=["secret"],
               operating_bias=0.5, profile="secrets-code", window=dict(size=512, stride=384), name=name, info=info)
        members.append(dict(file=fn, seed=seed, sha256=sha_file(os.path.join(folder, fn))))
    man = dict(name="secrets-code", version="1.0.0", specialist="secrets-code", members=members, vote=2,
               seeds=[0, 1, 2], released_seed=0, operating_bias=0.5, task_sha256="0" * 64, code_commit="abc")
    with open(os.path.join(folder, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)
    return folder


def test_public_bundle(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREBYTE_WORK", str(tmp_path / "work"))
    rel = _fake_release(str(tmp_path / "rel"))
    out = public_bundle(rel, "1.0.0")
    names = sorted(os.listdir(out["bundle"]))
    assert "purebyte-secrets-code-1.0.0.gguf" in names and "purebyte-secrets-code-1.0.0-e2.gguf.sha256" in names
    assert "MODEL_CARD.md" in names and "models.json.fragment" in names
    with open(os.path.join(out["bundle"], "MODEL_CARD.md"), encoding="utf-8") as f:
        card = f.read()
    assert "{{" not in card and "2 of 3" in card
    with open(os.path.join(tmp_path, "work", "releases", "MAPPING.jsonl"), encoding="utf-8") as f:
        line = f.read()
    assert json.loads(line)["public"] == "secrets-code 1.0.0"
    with pytest.raises(StageError, match="never overwritten"):
        public_bundle(rel, "1.0.0")


def test_a_per_type_operating_point_reaches_the_card(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREBYTE_WORK", str(tmp_path / "work"))
    rel = _fake_release(str(tmp_path / "rel"))
    with open(os.path.join(rel, "MANIFEST.json"), encoding="utf-8") as f:
        man = json.load(f)
    man.update(operating_bias_vec=[0.5, 1.5], labels=["EMAIL", "IBAN"])
    with open(os.path.join(rel, "MANIFEST.json"), "w", encoding="utf-8") as f:
        json.dump(man, f)
    tpl = tmp_path / "card.md"
    tpl.write_text("# card {{version}}\nOperating point: {{operating_bias}}\n{{files}}\n", encoding="utf-8")
    out = public_bundle(rel, "1.0.0", card_template=str(tpl))
    with open(os.path.join(out["bundle"], "MODEL_CARD.md"), encoding="utf-8") as f:
        assert "per type EMAIL 0.5, IBAN 1.5" in f.read()


def test_a_card_that_names_another_version_is_refused(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREBYTE_WORK", str(tmp_path / "work"))
    rel = _fake_release(str(tmp_path / "rel"))
    tpl = tmp_path / "card.md"
    tpl.write_text("# secrets-code {{version}}\nMeasured on secrets-code 1.0.0.\n{{files}}\n", encoding="utf-8")
    with pytest.raises(StageError, match="names secrets-code 1.0.0, and this bundle is secrets-code 1.0.1"):
        public_bundle(rel, "1.0.1", card_template=str(tpl))
    assert public_bundle(rel, "1.0.0", card_template=str(tpl))["files"]


def test_hygiene_refuses_internal_strings(tmp_path, monkeypatch):
    monkeypatch.setenv("PUREBYTE_WORK", str(tmp_path / "work"))
    for bad in ("trained on C:\\Users\\me\\runs", "model v999", "data in /root/purebyte/runs", "see /tmp/alice-run-17",
                "copied from \\\\fileserver\\share\\runs", "~/runs/secrets-code-exp3", "`/home/me/x`", "D:/runs/a",
                "(/Users/me/work)", "at /mnt/scratch/x"):
        assert hygiene({"card": bad}), bad
    for ok in ("secrets-code 1.0.0, format version 3", "https://github.com/purebyte-ai/purebyte/blob/main/x.md",
               "purebyte scan path/to/repo", "purebyte scan --model secrets-bin dist/", "notes.txt > notes.redacted.txt",
               "ratio 3:1 of files", "https://example.com/data/set"):
        assert not hygiene({"card": ok}), ok
    rel = _fake_release(str(tmp_path / "rel2"), name="secrets-code v12")
    with pytest.raises(StageError, match="internal strings"):
        public_bundle(rel, "1.0.1")
    assert len(FORBIDDEN) >= 5


@pytest.mark.parametrize("specialist", ["secrets-code", "secrets-bin", "pii"])
def test_the_model_card_templates_pass_the_hygiene_check(specialist, tmp_path, monkeypatch):
    monkeypatch.setenv("PUREBYTE_WORK", str(tmp_path / "work"))
    with open(os.path.join(paths.SPECIALISTS_SRC, specialist, "MODEL_CARD.template.md"), encoding="utf-8") as f:
        assert hygiene({"card": f.read()}) == []


def test_informative_metadata_and_version_check(tmp_path, monkeypatch):
    from pbtrain.export.gguf import read_metadata
    from pbtrain.export.pbformat import ExportError
    monkeypatch.setenv("PUREBYTE_WORK", str(tmp_path / "work"))
    info = {"version": "1.0.0", "license": "other", "license.name": "PureByte Model License"}
    rel = _fake_release(str(tmp_path / "rel3"), info=info)
    meta = read_metadata(os.path.join(rel, "model.gguf"))
    assert meta["general.version"] == "1.0.0" and meta["general.license.name"] == "PureByte Model License"
    with pytest.raises(StageError, match="general.version"):
        public_bundle(rel, "1.0.1")
    with pytest.raises(StageError, match="semantic version"):
        public_bundle(rel, "../1.0.0")
    bundle = public_bundle(rel, "1.0.0")["bundle"]
    with open(os.path.join(bundle, "purebyte-secrets-code-1.0.0.gguf.sha256"), "rb") as f:
        assert f.read().endswith(b"  purebyte-secrets-code-1.0.0.gguf\n")      # LF on every OS: sha256sum -c reads it
    m, th, ch = train_spans(gc.Task(), SpanRecipe(size="mini64", steps=2, batch=8, lr=1e-3, warmup=1), seed=0)
    with pytest.raises(ExportError, match="info key"):
        export(str(tmp_path / "x.gguf"), m.eval(), [th.eval(), ch.eval()], gating={0: 1}, info={"licence": "x"})
