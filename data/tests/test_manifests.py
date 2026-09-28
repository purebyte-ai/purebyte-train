"""Every manifest in data/ is valid, and the validator refuses what it must refuse."""
import copy
import os
import re

import pytest

from pbdata import manifest
from pbdata.paths import DATA, EXAM_MANIFESTS, SOURCE_MANIFESTS
from pbdata.util import DataError

SOURCES = manifest.load_sources()
EXAMS = manifest.load_exams()


def test_there_are_manifests():
    assert {m.name for m in SOURCES} == {"code-context", "binary-context", "pii", "pii-pool", "secret-catalogs"}
    assert {m.name for m in EXAMS} == {"code-fp", "creddata", "binary-fp", "pii"}


@pytest.mark.parametrize("m", SOURCES + EXAMS, ids=lambda m: os.path.basename(m.path))
def test_manifest_is_valid(m):
    assert manifest.validate(m.doc, m.path) == []


def test_set_names_are_unique_across_exam_manifests():
    names = [s["name"] for m in EXAMS for s in m.sets]
    assert len(names) == len(set(names))


def test_group_shares_add_up():
    for m in SOURCES:
        groups = (m.doc.get("build") or {}).get("groups")
        if groups:
            assert abs(sum(g["share"] for g in groups.values()) - 1.0) < 1e-9, m.name


def test_code_fp_sets_and_creddata_partitions():
    sets = {s["name"]: s for m in EXAMS for s in m.sets}
    for name in ("code-fp-a", "code-fp-b", "code-fp-c", "code-fp-dev", "creddata-test", "creddata-dev"):
        assert name in sets
    test = {r["id"] for r in sets["creddata-test"]["entries"][0]["repos"]}
    dev = {r["id"] for r in sets["creddata-dev"]["entries"][0]["repos"]}
    assert len(test) == 168 and len(dev) == 169 and not test & dev


@pytest.mark.parametrize("expr,ok", [
    ("MIT", True), ("Apache-2.0 OR MIT", True), ("(MIT OR Apache-2.0) AND BSD-3-Clause", True),
    ("Apache-2.0 WITH LLVM-exception", True), ("LicenseRef-Microsoft-Windows", True),
    ("", False), ("MIT OR", False), ("OR MIT", False), ("(MIT", False), ("MIT)", False), ("M I T", False),
])
def test_spdx(expr, ok):
    assert manifest.spdx_ok(expr) is ok


GOOD_SOURCE = {
    "schema": 1, "kind": "source", "name": "tiny",
    "entries": [{"id": "a", "type": "git", "url": "https://example.org/a/b", "pin": {"commit": "a" * 40},
                 "licence": "MIT"}],
}


@pytest.mark.parametrize("mutate,needle", [
    (lambda d: d["entries"][0]["pin"].update(commit="abc"), "40-character"),
    (lambda d: d["entries"][0].update(url="http://example.org/a/b"), "https://"),
    (lambda d: d["entries"][0].update(licence="not a licence!"), "SPDX"),
    (lambda d: d["entries"][0].update(colour="red"), "unknown keys"),
    (lambda d: d["entries"].append(dict(d["entries"][0])), "duplicate entry ids"),
    (lambda d: d["entries"][0].update(type="local", files=[{"name": "x", "sha256": "0" * 64, "size": 1}]),
     "cannot be `local`"),
    (lambda d: d["entries"][0].update(type="pypi", package="x", filename="x.whl", pin={"version": "1"}), "sha256"),
])
def test_validator_refuses(mutate, needle):
    doc = copy.deepcopy(GOOD_SOURCE)
    mutate(doc)
    errors = manifest.validate(doc)
    assert any(needle in e for e in errors), errors
    with pytest.raises(DataError):
        manifest.from_doc(doc)


def test_good_source_is_accepted():
    assert manifest.validate(GOOD_SOURCE) == []


# ---------------------------------------------------------------------------------------------- repository hygiene
# The data/ folder is published: no machine paths, no private tool names, no data files.
FORBIDDEN = [r"[A-Za-z]:\\Users\\", r"/home/[a-z]+/", r"/Users/[A-Za-z]+/"]
# the maintainers' private names (tools, runs, machines) come from a file outside the repository, one regex per line
_PRIVATE = os.environ.get("PUREBYTE_PRIVATE_PATTERNS")
if _PRIVATE and os.path.exists(_PRIVATE):
    with open(_PRIVATE, encoding="utf-8") as _f:
        FORBIDDEN += [ln.strip() for ln in _f if ln.strip() and not ln.startswith("#")]
TEXT_EXT = (".py", ".yaml", ".yml", ".md", ".txt", ".toml", ".cfg", ".ini", ".json")


def _published_files():
    for dp, dns, fns in os.walk(DATA):
        dns[:] = [d for d in dns if d not in ("__pycache__", ".pytest_cache")]
        for fn in fns:
            yield os.path.join(dp, fn)


def test_no_internal_paths_or_names():
    bad = []
    for p in _published_files():
        if not p.endswith(TEXT_EXT) or os.path.basename(p) == "test_manifests.py":
            continue
        with open(p, encoding="utf-8") as f:
            text = f.read()
        for rx in FORBIDDEN:
            m = re.search(rx, text)
            if m:
                bad.append(f"{os.path.relpath(p, DATA)}: `{m.group(0)}`")
    assert not bad, bad


def test_no_data_files_in_the_folder():
    big = [p for p in _published_files() if not (p.endswith(TEXT_EXT) or os.path.basename(p) == ".gitignore")
           or os.path.getsize(p) > 256 << 10]
    assert not big, big


def test_manifest_dirs_exist():
    assert os.path.isdir(SOURCE_MANIFESTS) and os.path.isdir(EXAM_MANIFESTS)
