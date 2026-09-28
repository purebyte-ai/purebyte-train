"""The whole factory on CPU, offline, in minutes: the secrets-code smoke recipe on a generated corpus with made-up
catalogs. Proves the pipeline (data + leak checks, rival, seeds, operating point, gate, criteria, calibration,
export, runtime equivalence through the NumPy reference, report), that the leak check FAILS a run whose data shares
content with a registered exam, and that the done.json chain refuses what the data stage did not check: a stream added
or re-mined after `data`, exams registered after `data`, a task edited after `data`, stages re-run out of order.
The release path itself is tested deterministically in test_release.py."""
import json
import os

import numpy as np
import pytest
import yaml

import smoke_data
from pbtrain import paths
from pbtrain.data import fingerprint as fp
from pbtrain.factory import spec as S
from pbtrain.factory.chain import StageError, done_path, read_done
from pbtrain.factory.stages import Context, run

RECIPE = os.path.join(paths.SPECIALISTS_SRC, "secrets-code", "recipes", "smoke-cpu.yaml")
QUIET = dict(log=lambda *a: None, verify_windows=6)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("pbdata")
    data, work = str(root / "data"), str(root / "work")
    smoke_data.make_data(data)
    with smoke_data.data_and_work(data, work):
        yield data, work


@pytest.fixture()
def fresh(tmp_path):
    """A $PUREBYTE_DATA / $PUREBYTE_WORK of this test only (its exams can be changed freely)."""
    data, work = str(tmp_path / "data"), str(tmp_path / "work")
    smoke_data.make_data(data)
    with smoke_data.data_and_work(data, work):
        yield data, work


@pytest.fixture()
def fixed_point(monkeypatch):
    """The operating point is set at bias 0 instead of chosen: the tests of the chain's mechanics train three steps,
    too few for a sweep to mean anything (the choice itself is tested by test_the_whole_chain and test_units)."""
    import pbtrain.factory.stages as stages
    monkeypatch.setattr(stages, "choose", lambda measure, grid, floor, policy:
                        (0.0, 0.5, [dict(bias=0.0, f1=0.5, precision=0.5, recall=0.5)]))


def task_file(folder, name="task.yaml", fast=True, edit=None):
    """A copy of the smoke recipe (with `fast`: one seed and three steps, so it does not decide), edited by
    `edit(task)`; returns its path."""
    with open(RECIPE, encoding="utf-8") as f:
        t = yaml.safe_load(f)
    if fast:
        t["train"].update(seeds=[0], steps=3, decide=False)
    if edit:
        edit(t)
    p = os.path.join(str(folder), name)
    with open(p, "w", encoding="utf-8") as f:
        yaml.safe_dump(t, f)
    return p


def edit_task(path, edit):
    with open(path, encoding="utf-8") as f:
        t = yaml.safe_load(f)
    edit(t)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(t, f)


def register_exam(data, name, content, role=None):
    meta = dict(name=name, files=1, bytes=len(content), **({"role": role} if role else {}))
    fp.save(os.path.join(data, "exams", "fingerprints", name + ".npz"), fp.fingerprints(content), meta)


def test_the_whole_chain(env):
    """Every stage runs. On so few steps the gate may refuse the model (then eval records the refusal and every later
    stage but `release` still runs); test_release.py covers the release itself."""
    data, work = env
    runs = os.path.join(work, "runs")
    spec = S.load(RECIPE)
    ctx = Context(**QUIET)
    out = run(spec, ["data", "baselines", "finetune"], runs, ctx)
    dd = out["data"]
    assert dd["exam_leak_check"]["exams"]["unrelated-exam"]["train"] == 0
    assert set(dd["split_overlap"]["enforced"]) == {"train-select", "train-test", "select-test"}
    assert set(dd["split_overlap"]["measured"]) == {"select_in_train", "test_in_train", "test_in_select"}
    assert set(dd["exam_registry"]) == {"unrelated-exam", *smoke_data.SECRETS_CODE_EXAMS}
    assert dd["warnings"] == [] and dd["task_digest"] == spec["_digest"] and dd["upstream_sha256"] is None
    ft = out["finetune"]
    assert ft["control_f1"] <= 0.10 and len(ft["runs"]) == 3
    assert all(r["ckpt"].endswith("final.pt") and len(r["ckpt_sha256"]) == 64 for r in ft["runs"])
    assert ft["training"]["device"] == "cpu" and ft["training"]["torch_threads"] == 1
    assert ft["training"]["trainer_digest"] and ft["released_seed_rule"] == "the best on select"
    try:
        run(spec, ["eval"], runs, ctx)
        passed = True
    except StageError as e:
        assert "GATE NOT PASSED" in str(e) or "NOT DELIVERED" in str(e)
        passed = False
    ev = read_done(runs, spec, "eval")
    assert ev["gate_passes"] is passed and bool(ev["refused"]) is not passed
    assert {v["verdict"] for v in ev["criteria"]} <= {"PASS", "FAIL"}
    out = run(spec, ["calibrate", "export", "verify", "report"], runs, ctx)
    assert out["calibrate"]["temperature"] == 1.0
    assert {"ece", "floor_p95", "resolved"} <= set(out["calibrate"]["document_head_ece"])
    assert out["export"]["format_version"] == 3 and out["export"]["ckpt_sha256"] == \
        next(r["ckpt_sha256"] for r in ft["runs"] if r["seed"] == ft["released_seed"])
    vf = out["verify"]
    assert vf["identical_spans"] and vf["n"] == 6 and vf["windows_with_spans"] > 0, vf
    with open(out["report"]["report"], encoding="utf-8") as f:
        report = f.read()
    assert "## Warnings" in report and "Between splits, measured" in report and "Trained on cpu" in report
    if not passed:
        with pytest.raises(StageError, match="gate was not passed"):
            run(spec, ["release"], runs, ctx)


def test_a_leaking_exam_fails_the_data_stage(env, tmp_path):
    data, work = env
    with open(os.path.join(data, "corpora", "smoke-context.bin"), "rb") as f:
        corpus = f.read()
    path = os.path.join(data, "exams", "fingerprints", "leaky-exam.npz")
    fp.save(path, fp.fingerprints(corpus[:150_000]), dict(name="leaky-exam", files=1, bytes=150_000))
    try:
        spec = S.load(RECIPE)
        with pytest.raises(StageError, match="shares content with an exam"):
            run(spec, ["data"], str(tmp_path / "runs"), Context(**QUIET))
    finally:
        os.remove(path)


def test_a_dev_exam_is_counted_and_reported_not_enforced(env, tmp_path):
    data, work = env
    with open(os.path.join(data, "corpora", "smoke-context.bin"), "rb") as f:
        corpus = f.read()
    register_exam(data, "dev-exam", np.frombuffer(corpus[:150_000], np.uint8), role="dev")
    try:
        dd = run(S.load(task_file(tmp_path)), ["data"], str(tmp_path / "runs"), Context(**QUIET))["data"]
    finally:
        os.remove(os.path.join(data, "exams", "fingerprints", "dev-exam.npz"))
    hits = dd["exam_leak_check"]["dev_hits"]["dev-exam"]
    assert hits["train"] > 0 and dd["exam_leak_check"]["exams"]["dev-exam"]["role"] == "dev"


def test_a_declared_key_that_nobody_reads_fails(env, tmp_path):
    p = task_file(tmp_path, edit=lambda t: t["data"].update(extra_line=5))         # a typo of extra_lines
    with pytest.raises(StageError, match="never read"):
        run(S.load(p), ["data"], str(tmp_path / "runs"), Context(**QUIET))


def test_criteria_cannot_change_after_the_data_stage(env, tmp_path):
    p = task_file(tmp_path)
    runs = str(tmp_path / "runs")
    run(S.load(p), ["data", "baselines"], runs, Context(**QUIET))
    edit_task(p, lambda t: t["criteria"]["synthetic"][0].update(value=0.5))
    with pytest.raises(StageError, match="changed after the data stage"):
        run(S.load(p), ["finetune"], runs, Context(**QUIET))


def test_criteria_exams_outside_exams_are_recorded_as_warnings(env, tmp_path):
    def edit(t):
        t["criteria"]["exams"] = [dict(exam="unrelated-exam", metric="f1", op=">=", value=0.5),
                                  dict(exam="never-fingerprinted", metric="f1", op=">=", value=0.5)]
    dd = run(S.load(task_file(tmp_path, edit=edit)), ["data"], str(tmp_path / "runs"), Context(**QUIET))["data"]
    w = " ".join(dd["warnings"])
    assert "never-fingerprinted" in w and "NOT registered" in w and "registered, so checked: ['unrelated-exam']" in w


def test_a_stream_added_after_data_is_refused(fresh, tmp_path):
    """The reviewer's case: `real_windows` cut from a registered exam is added to `train:` after `data`. finetune must
    refuse (the task changed), and the data stage on the edited task refuses the stream."""
    data, work = fresh
    exam = np.random.default_rng(11).integers(32, 127, 60_000, dtype=np.uint8)
    register_exam(data, "leaky-exam", exam)
    os.makedirs(os.path.join(data, "streams"))
    np.savez(os.path.join(data, "streams", "leaky.npz"), X=np.stack([exam[i * 128:(i + 1) * 128] for i in range(64)]))
    p = task_file(tmp_path)
    runs = str(tmp_path / "runs")
    run(S.load(p), ["data", "baselines"], runs, Context(**QUIET))
    edit_task(p, lambda t: t["train"].update(real_windows="${PUREBYTE_DATA}/streams/leaky.npz", real_frac=0.25,
                                               real_mode="neg"))
    with pytest.raises(StageError, match="another version of the task"):
        run(S.load(p), ["finetune"], runs, Context(**QUIET))
    with pytest.raises(StageError, match=r"leaky-exam: \d+ window\(s\) of real_windows"):
        run(S.load(p), ["data"], runs, Context(**QUIET))


def test_a_stream_rewritten_after_data_is_refused(fresh, tmp_path):
    """Same path, new content: the data stage checked the old file, so finetune refuses the new one."""
    data, work = fresh
    stream = os.path.join(data, "streams", "neg.npz")
    os.makedirs(os.path.dirname(stream))
    np.savez(stream, X=np.random.default_rng(3).integers(0, 256, (64, 128), dtype=np.uint8))
    p = task_file(tmp_path, edit=lambda t: t["train"].update(real_windows="${PUREBYTE_DATA}/streams/neg.npz",
                                                             real_frac=0.25, real_mode="neg"))
    runs = str(tmp_path / "runs")
    dd = run(S.load(p), ["data", "baselines"], runs, Context(**QUIET))["data"]
    assert dd["streams"]["real_windows"]["sha256"] and dd["exam_leak_check"]["exams"]["code-fp-a"]["real_windows"] == 0
    exam = np.random.default_rng(12).integers(32, 127, 60_000, dtype=np.uint8)
    np.savez(stream, X=np.stack([exam[i * 128:(i + 1) * 128] for i in range(64)]))
    with pytest.raises(StageError, match="not the file the data stage checked"):
        run(S.load(p), ["finetune"], runs, Context(**QUIET))


def test_exams_registered_after_data_are_not_ignored(fresh, tmp_path):
    data, work = fresh
    p = task_file(tmp_path)
    runs = str(tmp_path / "runs")
    run(S.load(p), ["data", "baselines"], runs, Context(**QUIET))
    register_exam(data, "late-exam", np.random.default_rng(13).integers(0, 256, 50_000, dtype=np.uint8))
    with pytest.raises(StageError, match="registered exams changed"):
        run(S.load(p), ["finetune"], runs, Context(**QUIET))


def test_a_task_edited_after_data_reaches_no_later_stage(env, tmp_path):
    """The reviewer's case: a key outside `criteria:` (here the recall floor) changes after `data`; every later stage,
    and so the release, refuses the chain (each one checks the digest of every earlier done.json, `data` first)."""
    p = task_file(tmp_path)
    runs = str(tmp_path / "runs")
    run(S.load(p), ["data"], runs, Context(**QUIET))
    edit_task(p, lambda t: t["output_schema"].update(min_select_recall=0.99))
    for stage in ("baselines", "finetune", "eval", "report", "release"):
        with pytest.raises(StageError, match="another version of the task"):
            run(S.load(p), [stage], runs, Context(**QUIET))


def test_criteria_cannot_be_refrozen_by_running_data_again(env, tmp_path):
    """The criteria of a version are frozen by its first data stage: running `data` again on edited criteria under the
    same version is refused, and the record keeps the first freeze."""
    p = task_file(tmp_path)
    runs = str(tmp_path / "runs")
    first = run(S.load(p), ["data"], runs, Context(**QUIET))["data"]
    assert first["criteria_frozen_at"]
    run(S.load(p), ["data"], runs, Context(**QUIET))                   # the same criteria: a re-run is fine
    edit_task(p, lambda t: t["criteria"]["synthetic"][0].update(value=0.99))
    with pytest.raises(StageError, match="were frozen on"):
        run(S.load(p), ["data"], runs, Context(**QUIET))
    assert os.path.exists(done_path(runs, S.load(p), "data"))        # the refusal deleted nothing
    edit_task(p, lambda t: t.update(version="0.0.2"))                 # a new version may state new criteria
    assert run(S.load(p), ["data"], runs, Context(**QUIET))["data"]["criteria_frozen_at"]


def test_a_failed_eval_never_leaves_an_earlier_pass(env, tmp_path, monkeypatch, fixed_point):
    import pbtrain.eval.baselines as rivals
    p = task_file(tmp_path)
    runs = str(tmp_path / "runs")
    spec = S.load(p)
    run(spec, ["data", "baselines", "finetune"], runs, Context(**QUIET))
    smoke_data.fabricate_finetune_pass(runs, spec)
    monkeypatch.setattr(rivals, "gate", lambda *a: (True, "passes (the test forces it)"))
    assert run(spec, ["eval"], runs, Context(**QUIET))["eval"]["gate_passes"] is True
    monkeypatch.setattr(rivals, "gate", lambda *a: (False, "refused (the test forces it)"))
    with pytest.raises(StageError, match="GATE NOT PASSED"):
        run(spec, ["eval"], runs, Context(**QUIET))
    ev = read_done(runs, spec, "eval")
    assert ev["gate_passes"] is False and ev["refused"][0]["by"] == "gate"


@pytest.mark.parametrize("seeds,scope", [(3, "INFORMS (3 seed(s): three seeds inform, six decide)"),
                                         (6, "DECIDES (6 seeds)")])
def test_a_run_decides_from_six_seeds(env, tmp_path, monkeypatch, fixed_point, seeds, scope):
    import pbtrain.eval.baselines as rivals
    p = task_file(tmp_path, edit=lambda t: t["train"].update(seeds=list(range(seeds)), decide=True))
    runs = str(tmp_path / "runs")
    spec = S.load(p)
    run(spec, ["data", "baselines", "finetune"], runs, Context(**QUIET))
    smoke_data.fabricate_finetune_pass(runs, spec)
    monkeypatch.setattr(rivals, "gate", lambda *a: (True, "passes (the test forces it)"))
    ev = run(spec, ["eval"], runs, Context(**QUIET))["eval"]
    assert ev["scope"] == scope and ev["decide"] is (seeds >= 6)


def test_the_chain_refuses_stages_out_of_order_and_edited_results(env, tmp_path, fixed_point):
    p = task_file(tmp_path)
    runs = str(tmp_path / "runs")
    spec = S.load(p)
    run(spec, ["data", "baselines", "finetune"], runs, Context(**QUIET))
    run(spec, ["baselines"], runs, Context(**QUIET))              # a re-run deletes every later done.json
    assert not os.path.exists(done_path(runs, spec, "finetune"))
    with pytest.raises(StageError, match="stage `finetune` is missing"):
        run(spec, ["eval"], runs, Context(**QUIET))
    run(spec, ["finetune"], runs, Context(**QUIET))
    bl = done_path(runs, spec, "baselines")
    with open(bl, encoding="utf-8") as f:
        d = json.load(f)
    d["rival_f1"] = 0.0                                             # an edited result
    with open(bl, "w", encoding="utf-8") as f:
        json.dump(d, f)
    with pytest.raises(StageError, match="made from another `baselines` result"):
        run(spec, ["eval"], runs, Context(**QUIET))


def test_a_forced_released_seed_is_recorded_with_the_rule_it_overrides(env, tmp_path, fixed_point):
    p = task_file(tmp_path, edit=lambda t: t["train"].update(seeds=[0, 1], released_seed=1))
    runs = str(tmp_path / "runs")
    ft = run(S.load(p), ["data", "baselines", "finetune"], runs, Context(**QUIET))["finetune"]
    assert ft["released_seed"] == 1 and ft["released_seed_rule"] == "forced by train.released_seed"
    assert ft["released_seed_reason"] is None and ft["select_best_seed"] == 0      # a tie on select: the first seed
    assert "gives no `released_seed_reason`" in ft["warnings"][0]
    edit_task(p, lambda t: t["train"].update(released_seed_reason="the fewest false alarms on a DEV set"))
    ft = run(S.load(p), ["data", "baselines", "finetune"], runs, Context(**QUIET))["finetune"]
    assert "the task's reason: the fewest false alarms on a DEV set" in ft["warnings"][0]
