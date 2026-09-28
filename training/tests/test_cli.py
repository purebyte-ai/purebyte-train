"""The command line: doctor runs without touching a GPU and verifies idle priority; new-vertical scaffolds a specialist
whose plugin builds valid data, whose smoke recipe validates and whose placeholder rival blocks the gate; status and
eval read a task; exam results are recorded append-only; options that would act on nothing are refused."""
import contextlib
import io
import json
import os
import subprocess
import sys

import pytest
import yaml

from pbtrain import paths
from pbtrain.cli import main

TRAINING = paths.TRAINING
RECIPE = os.path.join(paths.SPECIALISTS_SRC, "secrets-code", "recipes", "smoke-cpu.yaml")


def _cli(*args, env=None):
    e = dict(os.environ, PYTHONPATH=TRAINING, **(env or {}))
    return subprocess.run([sys.executable, "-m", "pbtrain", "--threads", "1", *args], capture_output=True, text=True,
                          env=e, timeout=600)


def _main(*argv):
    err, out = io.StringIO(), io.StringIO()
    with contextlib.redirect_stderr(err), contextlib.redirect_stdout(out):
        try:
            rc = main(["--threads", "1", *argv])
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue(), err.getvalue()


def test_doctor_runs(tmp_path):
    r = _cli("doctor", env=dict(PUREBYTE_DATA=str(tmp_path)))
    assert r.returncode == 0, r.stderr[-500:]
    assert "[ok] idle priority verified" in r.stdout and "specialists" in r.stdout
    assert "PyTorch build" in r.stdout and "CUDA / ROCm GPU" in r.stdout
    assert "[--] corpus code-context.bin: data/build_code_corpus.py" in r.stdout            # missing, with its builder
    assert "[--] catalog gitleaks.json" in r.stdout and "catalogue" not in r.stdout


def test_doctor_tells_made_up_stand_ins_from_real_data(tmp_path):
    import hashlib
    from pbtrain.doctor import standins
    cat = tmp_path / "catalogs" / "gitleaks.json"
    cat.parent.mkdir()
    cat.write_bytes(b'{"rules": []}')
    (tmp_path / "smoke-standins.json").write_text(json.dumps(
        {"files": {"catalogs/gitleaks.json": hashlib.sha256(b'{"rules": []}').hexdigest()}}), encoding="utf-8")
    assert standins(str(tmp_path)) == {"catalogs/gitleaks.json"}
    cat.write_bytes(b'{"rules": [1]}')                                   # rebuilt for real: no longer a stand-in
    assert standins(str(tmp_path)) == set() and standins(str(tmp_path / "none")) == set()


def test_new_vertical_scaffold_builds_and_validates(tmp_path):
    from pbtrain.factory import spec as S
    from pbtrain.factory.spec import Watched
    from pbtrain.scaffold import new_vertical
    from pbtrain.verticals import get, load_specialist
    new_vertical("toy-finder", ["token", "key"], root=str(tmp_path))
    folder = tmp_path / "toy-finder"
    assert (folder / "plugin.py").exists() and (folder / "recipes" / "smoke-cpu.yaml").exists()
    load_specialist("toy-finder", root=str(tmp_path))
    v = get("toy_finder")
    assert v.n_entities == 2
    t = v.build(None, Watched(dict(n_train=16, n_eval=8, seq_len=64, split_policy=dict(seed=0))), (0.8, 0.1, 0.1))
    t.check(64, 2)
    assert v.span_rival(t.select, t.test) is None                        # no rival until one is written
    spec = S.load(str(folder / "recipes" / "smoke-cpu.yaml"))
    assert spec["train"]["size"] == "mini64" and spec["engine"] == {"threads": 1}
    with pytest.raises(SystemExit):
        new_vertical("toy-finder", ["x"], root=str(tmp_path))          # never overwrites


def test_a_scaffolded_specialist_stops_at_the_rival_and_refuses_an_unread_path(tmp_path, monkeypatch):
    """The placeholder rival is None: `baselines` refuses, so no model of a new specialist can pass the gate before a
    real rival exists. And a `data.path` that a vertical without a corpus never reads fails the data stage."""
    from pbtrain.factory import spec as S
    from pbtrain.factory.chain import StageError
    from pbtrain.factory.stages import Context, run
    from pbtrain.scaffold import new_vertical
    from pbtrain.verticals import load_specialist
    new_vertical("toy-gate", ["token"], root=str(tmp_path))
    load_specialist("toy-gate", root=str(tmp_path))
    monkeypatch.setenv("PUREBYTE_DATA", str(tmp_path / "data"))
    recipe = str(tmp_path / "toy-gate" / "recipes" / "smoke-cpu.yaml")
    runs = str(tmp_path / "runs")
    quiet = Context(log=lambda *a: None)
    run(S.load(recipe), ["data"], runs, quiet)
    with pytest.raises(StageError, match="provides no classic rival"):
        run(S.load(recipe), ["baselines"], runs, quiet)
    with open(recipe, encoding="utf-8") as f:
        t = yaml.safe_load(f)
    t["data"]["path"] = "corpora/none.bin"
    p = tmp_path / "with-path.yaml"
    p.write_text(yaml.safe_dump(t), encoding="utf-8")
    with pytest.raises(StageError, match=r"never read by the vertical .*data\.path"):
        run(S.load(str(p)), ["data"], str(tmp_path / "runs2"), quiet)


def test_status_and_eval_on_a_fresh_task(tmp_path):
    r = _cli("--work", str(tmp_path), "status", RECIPE)
    assert r.returncode == 0, r.stderr[-500:]
    assert "----  data" in r.stdout and "PENDING" in r.stdout


def test_exam_results_are_recorded_append_only(tmp_path):
    with open(RECIPE, encoding="utf-8") as f:
        t = yaml.safe_load(f)
    t["criteria"]["exams"] = [dict(exam="code-fp-b", metric="findings_max_seed", op="<=", value=129)]
    task = tmp_path / "t.yaml"
    task.write_text(yaml.safe_dump(t), encoding="utf-8")
    work = str(tmp_path / "work")
    m1, m2 = tmp_path / "m1.json", tmp_path / "m2.json"
    m1.write_text(json.dumps({"findings_max_seed": 200}), encoding="utf-8")
    m2.write_text(json.dumps({"findings_max_seed": 100}), encoding="utf-8")
    rc, out, err = _main("--work", work, "eval", str(task), "--record", "code-fp-b", "--metrics", str(m1))
    assert rc == 0 and "FAIL" in out and "WARNING" not in err
    rc, out, err = _main("--work", work, "eval", str(task), "--record", "code-fp-b", "--metrics", str(m2))
    assert rc == 0 and "PASS" in out and "already had 1 record" in err
    from pbtrain.factory import exams as E
    from pbtrain.factory import spec as S
    hist = E.history(work, S.load(str(task)))["code-fp-b"]
    assert [h["metrics"]["findings_max_seed"] for h in hist] == [200, 100] and all(h["metrics_sha256"] for h in hist)
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"findings_max_seed": "many"}), encoding="utf-8")
    rc, out, err = _main("--work", work, "eval", str(task), "--record", "code-fp-b", "--metrics", str(bad))
    assert rc == 2 and "numbers" in err
    rc, out, err = _main("--work", work, "eval", str(task), "--record", "../escape", "--metrics", str(m1))
    assert rc == 2 and "names a file" in err


@pytest.mark.parametrize("argv,message", [
    (("data", RECIPE, "--device", "cuda"), None),                        # argparse: no such option on `data`
    (("export", RECIPE, "--parallel", "3"), None),
    (("status", RECIPE, "--backend", "radeon"), None),
    (("eval", RECIPE, "--metrics", "m.json"), "acts only with --record"),
    (("run", RECIPE, "--stages", "data,baselines", "--device", "cuda"), "run on the CPU"),
    (("run", RECIPE, "--stages", "eval", "--parallel", "2"), "run on the CPU"),
    (("train", RECIPE, "--device", "cpu", "--backend", "radeon"), "runs on a GPU"),
    (("train", RECIPE, "--device", "cpu", "--backend", "cuda"), "runs on a GPU"),
    (("train", RECIPE, "--parallel", "0"), "1 or more"),
])
def test_options_that_would_act_on_nothing_are_refused(argv, message, tmp_path):
    rc, out, err = _main("--work", str(tmp_path), *argv)
    assert rc not in (0, None)
    if message:
        assert message in (str(rc) + err)


def test_a_missing_gpu_is_a_message_not_a_traceback(tmp_path):
    """The test session hides every CUDA / ROCm device: asking for one is an error with a clear message (exit 2)."""
    rc, out, err = _main("--work", str(tmp_path), "train", RECIPE, "--device", "cuda")
    assert rc == 2 and "pbtrain: device cuda asked for" in err and "Traceback" not in err
