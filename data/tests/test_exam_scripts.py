"""fetch_exams.py and fingerprint_exams.py on synthetic evaluation sets, offline: a failing set never stops the others,
the summary names it and the exit status says so; files without a public download never stop a set (its fingerprint
file says it is incomplete); nothing is downloaded while fingerprinting; the shared options are accepted only by the
scripts that act on them."""
import hashlib
import os

import numpy as np
import pytest

import fetch_exams
import fingerprint_exams
from pbdata import cli, exams, fetch, manifest
from pbdata.util import DataError

X, Y = (np.random.default_rng(seed).integers(0, 256, 4096, dtype=np.uint8).tobytes() for seed in (1, 2))
SHA = lambda b: hashlib.sha256(b).hexdigest()  # noqa: E731


def _sets(git=True):
    doc = {"schema": 1, "kind": "exam", "name": "tiny", "sets": [
        {"name": "tiny-files", "role": "test", "entries": [
            {"id": "pub", "type": "url", "url": "https://example.invalid/x.bin", "licence": "MIT",
             "pin": {"sha256": SHA(X), "size": len(X)}, "files": [{"name": "x.bin", "sha256": SHA(X), "size": len(X)}]},
            {"id": "loc", "type": "local", "licence": "MIT", "provenance": "a local copy only",
             "files": [{"name": "y.bin", "sha256": SHA(Y), "size": len(Y)}]}]}]}
    if git:
        doc["sets"].append({"name": "tiny-git", "role": "test", "entries": [
            {"id": "repo", "type": "git", "url": "https://example.invalid/never/fetched", "licence": "MIT",
             "pin": {"commit": "a" * 40}}]})
    m = manifest.from_doc(doc, "tiny.yaml")
    return [(s, m) for s in m.sets]


@pytest.fixture
def tiny(data_home, monkeypatch):
    sets = _sets()
    monkeypatch.setattr(manifest, "exam_sets", lambda exams=None: sets)
    monkeypatch.setattr(cli, "start", lambda args: None)          # no priority or thread changes in the test process
    monkeypatch.setattr(fetch, "fetch_git", lambda *a, **k: pytest.fail("nothing may be downloaded here"))
    monkeypatch.setattr(fetch, "fetch_file", lambda *a, **k: pytest.fail("nothing may be downloaded here"))
    return data_home


def _run(module, monkeypatch, *argv):
    monkeypatch.setattr("sys.argv", [module.__name__ + ".py", *argv])
    return module.main()


def test_a_missing_local_only_file_never_stops_a_set(tiny, monkeypatch, capsys, F):
    files = tiny / "exams" / "tiny-files" / "files"
    files.mkdir(parents=True)
    (files / "x.bin").write_bytes(X)
    assert _run(fingerprint_exams, monkeypatch, "--set", "tiny-files") == 0
    fp, meta = F.load(str(tiny / "exams" / "fingerprints" / "tiny-files.npz"))
    assert meta["complete"] is False and meta["missing_count"] == 1 and meta["missing_local_only"] == 1
    assert meta["files"] == 1 and len(fp) > 0
    out = capsys.readouterr().out
    assert "INCOMPLETE" in out and "not excluded" in out


def test_a_failing_set_does_not_stop_the_others(tiny, monkeypatch, capsys, F):
    files = tiny / "exams" / "tiny-files" / "files"
    files.mkdir(parents=True)
    (files / "x.bin").write_bytes(X)
    assert _run(fingerprint_exams, monkeypatch) == 1                 # tiny-git is not fetched: status 1
    out = capsys.readouterr().out
    assert "tiny-git" in out and "NOT FINGERPRINTED" in out and "fetch_exams.py --set tiny-git" in out
    assert "--allow-missing-exams" in out
    assert os.path.exists(tiny / "exams" / "fingerprints" / "tiny-files.npz")
    assert not os.path.exists(tiny / "exams" / "fingerprints" / "tiny-git.npz")


def test_a_missing_public_file_needs_allow_partial(tiny, monkeypatch, capsys, F):
    assert _run(fingerprint_exams, monkeypatch, "--set", "tiny-files") == 1       # neither file is there
    assert "have a public download are not on disk" in capsys.readouterr().out
    (tiny / "local").mkdir()
    (tiny / "local" / "copy.bin").write_bytes(Y)
    assert _run(fingerprint_exams, monkeypatch, "--set", "tiny-files", "--allow-partial", "--from-dir",
                str(tiny / "local")) == 0
    _, meta = F.load(str(tiny / "exams" / "fingerprints" / "tiny-files.npz"))
    assert meta["missing"] == ["pub:x.bin"] and meta["missing_local_only"] == 0 and meta["source"] == "from-dir"


def test_fetching_continues_past_a_failed_set(tiny, monkeypatch, capsys):
    def fake(s, m, local, dry_run, allow_drift=False, sampled=None):
        if s["name"] == "tiny-files":
            raise DataError("x.bin has sha256 0000, the manifest pins 1111")
        return dict(set=s["name"], entries=[], missing=[])
    monkeypatch.setattr(fetch_exams, "fetch_set", fake)
    assert _run(fetch_exams, monkeypatch) == 1
    out = capsys.readouterr().out
    assert "tiny-files       FAILED: x.bin has sha256" in out and "tiny-git         fetched" in out
    assert "--allow-missing-exams" in out


def test_fetch_reports_local_only_files(tiny, monkeypatch, capsys):
    monkeypatch.setattr(fetch_exams, "fetch_set",
                        lambda s, m, local, dry_run, allow_drift=False, sampled=None: dict(set=s["name"], entries=[],
                                                                             missing=exams.local_only_files(s)))
    assert _run(fetch_exams, monkeypatch) == 0
    out = capsys.readouterr().out
    assert "fetched without 1 file(s) that have no public download" in out and "--from-dir" in out


def test_options_are_accepted_only_where_they_act(tiny, monkeypatch):
    for module, flag in ((fingerprint_exams, "--allow-missing-exams"), (fingerprint_exams, "--no-fetch"),
                         (fetch_exams, "--allow-missing-exams"), (fetch_exams, "--no-fetch")):
        with pytest.raises(SystemExit):
            _run(module, monkeypatch, flag, "--dry-run")
    ap = cli.parser("doc", "a build", excludes=True, fetches=True)
    a = ap.parse_args(["--allow-missing-exams", "--no-fetch"])
    assert a.allow_missing_exams and a.no_fetch
    with pytest.raises(SystemExit):
        cli.parser("doc", "a fetch").parse_args(["--no-fetch"])


def test_the_pii_benchmark_samples_are_cut_once_per_run(data_home, monkeypatch):
    """pii-bench-test and pii-bench-dev come from one sampling run per dataset, with the parameters pii-bench-test
    declares (pii-bench-dev does not repeat them), and every file is compared with the reference sample."""
    from pbdata import pii
    calls = []

    def fake(entry):
        calls.append((entry["dataset"], bool(entry.get("sampling"))))
        return {"test": [pii.make_doc("t-1", entry["dataset"], "en", "a test document", [])],
                "dev": [pii.make_doc("d-1", entry["dataset"], "en", "a dev document", [])]}
    monkeypatch.setattr(pii, "BENCH", {k: fake for k in pii.BENCH})
    monkeypatch.setattr(pii, "check_revision", lambda e, allow, revs: None)
    sets = {s["name"]: s for s, _ in manifest.exam_sets()}
    sampled = {}
    r1 = pii.fetch_exam_set(sets["pii-bench-test"], sampled=sampled)
    r2 = pii.fetch_exam_set(sets["pii-bench-dev"], sampled=sampled)
    assert len(calls) == 3 and all(declared for _, declared in calls)
    assert r1 == r2 and all(rec["matches_reference"] == {"test": False, "dev": False} for rec in r1)
    pii.fetch_exam_set(sets["pii-bench-dev"])                  # alone: sampled again, with the declared parameters
    assert len(calls) == 6 and all(declared for _, declared in calls)
