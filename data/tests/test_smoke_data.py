"""The smoke stand-ins (pbdata/smoke.py, build_smoke_data.py): the same bytes as the test suites' own generators, a
fresh $PUREBYTE_DATA gets everything the two smoke recipes read, real data are never replaced (nor anything deleted),
and a stand-in is recognised until a real build overwrites it. Offline, NumPy only (the factory side is in
specialists/secrets-code/tests)."""
import importlib.util
import json
import os

import pytest

from pbdata import paths, smoke
from pbdata.util import DataError, sha256_file

S = smoke.standins_module()


def _load(rel, name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(paths.REPO, *rel.split("/")))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _tree(folder):
    out = {}
    for dp, _, fns in os.walk(folder):
        for fn in fns:
            p = os.path.join(dp, fn)
            out[os.path.relpath(p, folder).replace("\\", "/")] = sha256_file(p)
    return out


def test_the_generators_are_those_of_the_test_suites(tmp_path):
    """Two copies of one generator: this test keeps them identical (a data script never imports a test)."""
    smoke_data = _load("training/tests/smoke_data.py", "smoke_data_under_test")
    lookalike = _load("specialists/pii/tests/pii_lookalike.py", "pii_lookalike_under_test")
    for n in (400_000, smoke.SMOKE_CORPUS_BYTES):
        a, b = tmp_path / f"a{n}.bin", tmp_path / f"b{n}.bin"
        smoke.code_corpus(str(a), n_bytes=n)
        smoke_data.code_corpus(str(b), n_bytes=n)
        assert a.read_bytes() == b.read_bytes()
    smoke.catalogs(str(tmp_path / "c1"))
    smoke_data.catalogs(str(tmp_path / "c2"))
    lookalike.catalogs(str(tmp_path / "c3"))
    assert _tree(tmp_path / "c1") == _tree(tmp_path / "c2") == _tree(tmp_path / "c3")
    names1 = smoke.lists(str(tmp_path / "l1"))
    names2 = lookalike.lists(str(tmp_path / "l2"))
    assert names1 == names2 and _tree(tmp_path / "l1") == _tree(tmp_path / "l2")
    assert sorted(_tree(tmp_path / "l1")) == sorted(smoke.LIST_FILES)
    smoke.public_docs(str(tmp_path / "d1"), *names1)
    lookalike.public_docs(str(tmp_path / "d2"), *names2)
    assert _tree(tmp_path / "d1") == _tree(tmp_path / "d2")


def test_a_fresh_data_folder_gets_everything_the_smoke_recipes_read(data_home):
    r = smoke.build(str(data_home), log=lambda *a: None)
    expected = {S.SMOKE_CORPUS, "catalogs/gitleaks.json", "catalogs/trufflehog.json",
                "pii/public_train/lookalike.jsonl"} | {f"pii/lists/{f}" for f in smoke.LIST_FILES}
    assert set(r["written"]) == expected and not r["kept"]
    assert os.path.getsize(data_home / "corpora" / "smoke-context.bin") == smoke.SMOKE_CORPUS_BYTES
    man = json.loads((data_home / "corpora" / "smoke-context.manifest.json").read_text(encoding="utf-8"))
    assert man["made_up"] is True and man["output"]["path"] == S.SMOKE_CORPUS
    assert set(S.in_place(str(data_home))) == expected
    before = _tree(data_home)
    again = smoke.build(str(data_home), log=lambda *a: None)             # the same bytes, every time
    assert set(again["written"]) == expected and _tree(data_home) == before


def test_real_data_are_never_replaced(data_home):
    (data_home / "catalogs").mkdir()
    (data_home / "catalogs" / "gitleaks.json").write_text('{"rules": []}', encoding="utf-8")
    (data_home / "pii" / "lists").mkdir(parents=True)
    (data_home / "pii" / "lists" / "places_us.tsv").write_text("name\tstate\n", encoding="utf-8")
    (data_home / "pii" / "public_train").mkdir()
    (data_home / "pii" / "public_train" / "real.jsonl").write_text("{}\n", encoding="utf-8")
    before = _tree(data_home)
    r = smoke.build(str(data_home), log=lambda *a: None)
    assert r["written"] == [S.SMOKE_CORPUS] and len(r["kept"]) == 3
    after = _tree(data_home)
    assert all(after[k] == v for k, v in before.items())
    assert not os.path.exists(data_home / "catalogs" / "trufflehog.json")
    assert set(S.in_place(str(data_home))) == {S.SMOKE_CORPUS}


def test_a_standin_is_real_again_once_a_real_build_replaces_it(data_home):
    smoke.build(str(data_home), log=lambda *a: None)
    (data_home / "catalogs" / "gitleaks.json").write_text('{"rules": [], "fps": []}', encoding="utf-8")
    left = S.in_place(str(data_home))
    assert "catalogs/gitleaks.json" not in left and "catalogs/trufflehog.json" in left
    (data_home / "pii" / "lists" / "places_us.tsv").write_text("name\tstate\n", encoding="utf-8")
    assert "pii/lists/places_us.tsv" not in S.in_place(str(data_home))
    assert len(S.in_place(str(data_home))) == 14                        # nothing is ever deleted, only recognised


def test_standins_serve_the_smoke_corpus_only(data_home):
    smoke.build(str(data_home), log=lambda *a: None)
    smoke_corpus = str(data_home / "corpora" / "smoke-context.bin")
    other = str(data_home / "corpora" / "code-context.bin")
    assert S.check(str(data_home), ("catalogs/",), smoke_corpus, "catalogues", "build them") == [
        "catalogs/gitleaks.json", "catalogs/trufflehog.json"]
    with pytest.raises(ValueError, match="made-up stand-ins"):
        S.check(str(data_home), ("catalogs/",), other, "catalogues", "build them")
    assert S.check(str(data_home), ("streams/",), other, "streams", "build them") == []


def test_the_creddata_stream_refuses_standin_catalogues(data_home):
    from pbdata import creddata_stream as CS
    smoke.build(str(data_home), log=lambda *a: None)
    with pytest.raises(DataError, match="made-up stand-ins"):
        CS.load_generator(str(data_home))


def test_the_script(data_home, capsys, monkeypatch):
    import build_smoke_data
    monkeypatch.setattr("sys.argv", ["build_smoke_data.py", "--dry-run"])
    assert build_smoke_data.main() == 0 and not os.listdir(data_home)
    monkeypatch.setattr("sys.argv", ["build_smoke_data.py"])
    assert build_smoke_data.main() == 0
    monkeypatch.setattr("sys.argv", ["build_smoke_data.py", "--verify"])
    assert build_smoke_data.main() == 0
    out = capsys.readouterr().out
    assert "would write corpora/smoke-context.bin" in out and "verify: the smoke stand-ins OK" in out
