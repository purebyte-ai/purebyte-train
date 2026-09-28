"""The identity half of the exclusion guarantee: no source may be an evaluation set."""
import pytest

from pbdata import identity, manifest
from pbdata.util import DataError


def src(*entries):
    return manifest.from_doc({"schema": 1, "kind": "source", "name": "s", "entries": list(entries)}, "s.yaml")


def exam(*entries):
    return manifest.from_doc({"schema": 1, "kind": "exam", "name": "x",
                              "sets": [{"name": "x-test", "role": "test", "entries": list(entries)}]}, "x.yaml")


def git(eid, url, commit="a" * 40):
    return {"id": eid, "type": "git", "url": url, "pin": {"commit": commit}, "licence": "MIT"}


def pypi(eid, name, version, sha="1" * 64):
    return {"id": eid, "type": "pypi", "package": name, "filename": f"{name}-{version}.whl",
            "url": f"https://files.example.org/{eid}.whl", "pin": {"version": version, "sha256": sha},
            "licence": "MIT"}


def deb(eid, name, version, sha="2" * 64):
    return {"id": eid, "type": "deb", "package": name, "arch": "amd64", "url": f"https://deb.example.org/{eid}.deb",
            "pin": {"version": version, "sha256": sha}, "licence": "GPL-2.0-or-later"}


def hf(eid, dataset, splits, **extra):
    return dict({"id": eid, "type": "hf", "dataset": dataset, "pin": {"revision": "b" * 40}, "splits": splits,
                 "licence": "CC-BY-4.0"}, **extra)


def test_the_real_manifests_are_disjoint():
    assert identity.check(manifest.load_sources(), manifest.load_exams()) == []


@pytest.mark.parametrize("a,b", [
    ("https://github.com/Foo/Bar", "https://github.com/foo/bar.git"),
    ("https://github.com/foo/bar/", "git@github.com:foo/bar.git"),
    ("http://www.github.com/foo/bar", "https://github.com/foo/bar"),
])
def test_repo_urls_normalise(a, b):
    assert identity.norm_repo(a) == identity.norm_repo(b)


def test_same_repository_at_another_commit_conflicts():
    c = identity.check([src(git("a", "https://github.com/foo/bar", "a" * 40))],
                       [exam(git("b", "https://github.com/Foo/Bar.git", "c" * 40))])
    assert c and c[0][0] == ("repo", "github.com/foo/bar")
    with pytest.raises(DataError):
        identity.require_disjoint([src(git("a", "https://github.com/foo/bar"))],
                                  [exam(git("b", "https://github.com/foo/bar"))])


def test_repository_listed_inside_an_exam_entry_conflicts():
    e = git("cd", "https://github.com/org/dataset")
    e["repos"] = [{"id": "00000001", "url": "https://github.com/some/upstream", "commit": "d" * 40}]
    assert identity.check([src(git("a", "https://github.com/some/upstream"))], [exam(e)])


def test_pypi_name_normalisation_and_versions():
    s = [src(pypi("a", "Scikit_Learn", "1.9.1", sha="3" * 64))]
    assert identity.check(s, [exam(pypi("b", "scikit-learn", "1.9.1", sha="4" * 64))])
    assert not identity.check(s, [exam(pypi("b", "scikit-learn", "1.9.0", sha="4" * 64))])


def test_deb_name_version_and_file_digest():
    assert identity.check([src(deb("a", "libc6", "2.41-12", sha="5" * 64))],
                          [exam(deb("b", "libc6", "2.41-12", sha="6" * 64))])
    assert not identity.check([src(deb("a", "libc6", "2.41-12", sha="5" * 64))],
                              [exam(deb("b", "libc6", "2.42-1", sha="6" * 64))])
    # the same file under another name is still the same file
    assert identity.check([src(deb("a", "x", "1", sha="7" * 64))], [exam(deb("b", "y", "2", sha="7" * 64))])


def test_member_file_digest_conflicts():
    local = {"id": "l", "type": "local", "licence": "LicenseRef-Unknown",
             "files": [{"name": "lib.so", "sha256": "8" * 64, "size": 10}]}
    e = pypi("a", "pkg", "1")
    e["files"] = [{"member": "pkg/lib.so", "sha256": "8" * 64, "size": 10}]
    assert identity.check([src(e)], [exam(local)])


def test_datasets_conflict_by_split_only():
    s = [src(hf("t", "Org/Data", ["train"]))]
    assert not identity.check(s, [exam(hf("v", "org/data", ["test"]))])
    assert identity.check(s, [exam(hf("v", "org/data", ["train", "test"]))])
    # a benchmark built from other datasets' test splits declares them
    bench = hf("bench", "other/bench", ["test"], derived_from=["hf:Org/Data:validation"])
    assert identity.check([src(hf("t", "org/data", ["validation"]))], [exam(bench)])


def test_a_custom_manifest_is_checked_and_recorded_where_it_is(tmp_path):
    """`--manifest` of the corpus builders: the manifest actually built from goes through the identity check too, and
    the build records where it is; `--out` never points inside the repository."""
    import os

    from pbdata import cli, paths
    exam_repo = next(e["url"] for m in manifest.load_exams() for s in m.sets for e in s["entries"] if e["type"] == "git"
                     and s.get("fingerprint") != "creddata")
    doc = {"schema": 1, "kind": "source", "name": "custom", "entries": [
        {"id": "sneaky", "type": "git", "url": exam_repo, "pin": {"commit": "b" * 40}, "licence": "MIT"}]}
    p = tmp_path / "custom.yaml"
    import yaml
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(DataError, match="identity exclusion failed"):
        cli.identity_preamble(built=manifest.load(str(p)))
    doc["entries"][0]["url"] = "https://example.org/not/an/exam"
    p.write_text(yaml.safe_dump(doc), encoding="utf-8")
    m = manifest.load(str(p))
    _, _, report = cli.identity_preamble(built=m)
    assert report["built_manifest"] == dict(path="custom.yaml", outside_repository=True, sha256=m.sha256)
    real = manifest.load_sources()[0]
    assert cli.manifest_ref(real)["path"].startswith("data/sources/")
    with pytest.raises(DataError, match="inside the repository"):
        cli.outside_repository(os.path.join(paths.REPO, "data", "corpus-here"), "--out")
    assert cli.outside_repository(str(tmp_path / "out"), "--out") == os.path.abspath(str(tmp_path / "out"))
