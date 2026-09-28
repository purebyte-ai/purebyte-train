"""Same inputs -> same corpus bytes: the code corpus builder on tiny local repositories."""
import copy
import os
import random

import numpy as np
import pytest

import build_code_corpus as B
from pbdata import exclusion, manifest
from pbdata.corpus import Candidate, CorpusWriter, dedup, interleave
from helpers import git_repo

BUILD = {
    "target_mib": 0.02, "seed": 0, "file_bytes": {"min": 16, "max": 65536}, "max_nul_bytes": 2,
    "tolerance": 1.0, "decile_tolerance": 1.0, "min_files": 0, "skip_dirs": ["node_modules", "vendor"],
    "groups": {
        "python": {"share": 0.5, "ext": [".py"]},
        "yaml_ci": {"share": 0.3, "ext": [".yml", ".yaml"], "names": ["Jenkinsfile"]},
        "shell": {"share": 0.2, "ext": [".sh"], "names": ["Makefile"]},
    },
}


def text(seed, lines=30):
    rng = np.random.default_rng(seed)
    return "".join(f"line_{seed}_{i} = {int(rng.integers(0, 10 ** 9))}  # {'x' * int(rng.integers(0, 40))}\n"
                   for i in range(lines)).encode()


@pytest.fixture
def repos(tmp_path):
    a = {f"src/mod{i}.py": text(i) for i in range(12)}
    a.update({"ci/pipeline.yml": text(100), "Makefile": text(101), "node_modules/skip.py": text(102),
              "bad.py": b"\xff\xfe not utf-8 \x00\x00\x00 binary", "LICENSE": b"Permission is hereby granted, free of charge"})
    b = {f"lib/{i}.py": text(200 + i) for i in range(8)}
    b.update({f"deploy/{i}.yaml": text(300 + i) for i in range(6)}, **{"Jenkinsfile": text(400), "run.sh": text(401),
                                                                      "dup.py": text(3)})    # same bytes as a/src/mod3.py
    ra = git_repo(str(tmp_path / "repo-a"), a)
    rb = git_repo(str(tmp_path / "repo-b"), b)
    return {"repo-a": ra, "repo-b": rb}


def tiny_manifest(repos, order=("repo-a", "repo-b")):
    doc = {"schema": 1, "kind": "source", "name": "tiny", "build": copy.deepcopy(BUILD), "entries": [
        {"id": rid, "type": "git", "url": f"https://example.org/{rid}", "pin": {"commit": repos[rid][1]},
         "licence": "MIT"} for rid in order]}
    return manifest.from_doc(doc, "tiny.yaml")


def build_once(repos, out_dir, cf=None, order=("repo-a", "repo-b")):
    man = tiny_manifest(repos, order)
    cf = cf or exclusion.ContentFilter([])
    res = B.assemble(man, man.doc["build"], cf, lambda e: repos[e["id"]][0], str(out_dir),
                     target=int(BUILD["target_mib"] * 2 ** 20), name="tiny")
    with open(os.path.join(out_dir, "tiny.bin"), "rb") as f:
        data = f.read()
    with open(os.path.join(out_dir, "tiny.index.tsv.gz"), "rb") as f:
        index = f.read()
    return res, data, index


def test_same_inputs_same_bytes(repos, tmp_path):
    r1, d1, i1 = build_once(repos, tmp_path / "out1")
    r2, d2, i2 = build_once(repos, tmp_path / "out2")
    assert d1 == d2 and i1 == i2 and len(d1) > 0
    assert r1["output"]["sha256"] == r2["output"]["sha256"]
    # the manifest records where each build was written (an --out outside $PUREBYTE_DATA: its absolute path), so that
    # --verify re-hashes the real file; everything else is the same
    for r, out in ((r1, tmp_path / "out1"), (r2, tmp_path / "out2")):
        assert os.path.samefile(r["output"]["path"], out / "tiny.bin")
        assert os.path.samefile(r["output"]["index"]["path"], out / "tiny.index.tsv.gz")

    def strip(r):
        out = dict(r["output"], path=None, index=dict(r["output"]["index"], path=None))
        return {k: (out if k == "output" else v) for k, v in r.items() if k != "environment"}
    assert strip(r1) == strip(r2)


def test_order_of_the_manifest_does_not_matter(repos, tmp_path):
    _, d1, _ = build_once(repos, tmp_path / "out1", order=("repo-a", "repo-b"))
    _, d2, _ = build_once(repos, tmp_path / "out2", order=("repo-b", "repo-a"))
    assert d1 == d2


def test_what_is_left_out(repos, tmp_path):
    res, data, _ = build_once(repos, tmp_path / "out")
    assert b"not utf-8" not in data                            # binary / invalid UTF-8
    assert text(102) not in data                               # a skipped directory
    assert data.count(text(3)) <= 1                            # duplicate content is taken once
    assert res["duplicates_left_out"] == 1
    assert {i["id"] for i in res["inputs"]} == {"repo-a", "repo-b"}
    assert all(i["licence_check"]["status"] in ("ok", "unverified") for i in res["inputs"])


def test_exam_content_never_reaches_the_corpus(repos, tmp_path, F):
    exam_text = text(5) + text(201)                            # two whole files of the sources are an exam
    ex = exclusion.ExamFingerprints("exam", "test", F.fingerprints(exam_text), {}, "<memory>")
    cf = exclusion.ContentFilter([ex])
    res, data, _ = build_once(repos, tmp_path / "out", cf=cf)
    assert cf.shared(data) == {}
    st = res["exclusion"]["content"]["filter"]["exam"]
    assert st["files_dropped"] + st["files_trimmed"] >= 1


def test_interleave_is_deterministic_and_keeps_the_mix():
    pieces = {"a": [(100, i) for i in range(50)], "b": [(50, i) for i in range(40)], "c": [(10, i) for i in range(10)]}
    o1 = interleave(pieces, ["a", "b", "c"])
    o2 = interleave({k: list(v) for k, v in pieces.items()}, ["a", "b", "c"])
    assert o1 == o2 and len(o1) == 100
    first_half = o1[:50]
    assert {g for g, _ in first_half} == {"a", "b", "c"}


def test_candidate_order_is_content_based():
    c1 = Candidate("g", "s1", "x.py", 10, "abc", None, 0)
    c2 = Candidate("g", "s2", "y.py", 10, "abc", None, 0)
    assert c1.key == c2.key                                    # the key follows the content, not the place
    c3 = Candidate("g", "s1", "x.py", 10, "abc", None, 1)
    assert c3.key != c1.key                                    # and the seed
    uniq, n = dedup([c2, c1])
    assert n == 1 and uniq[0].source == "s1"


def test_writer_is_independent_of_candidate_list_order(tmp_path):
    rng = random.Random(0)
    blobs = {f"c{i}": (f"piece {i} " * (i % 7 + 3)).encode() + b"\n" for i in range(60)}

    def run(out, shuffle):
        cands = [Candidate("g" if i % 3 else "h", "s", k, len(v), k, None, 0) for i, (k, v) in enumerate(blobs.items())]
        if shuffle:
            rng.shuffle(cands)
        w = CorpusWriter("t", str(out), {"g": {"share": 0.6}, "h": {"share": 0.4}}, target_bytes=1500)
        pieces, stats = w.select(cands, lambda pool: ((c, blobs[c.cid]) for c in pool), lambda c, raw: (raw, "ok"))
        w.write(pieces)
        with open(w.bin_path, "rb") as f:
            return f.read()

    assert run(tmp_path / "a", False) == run(tmp_path / "b", True)
