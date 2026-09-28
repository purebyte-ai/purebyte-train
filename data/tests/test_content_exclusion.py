"""The content half of the exclusion guarantee, on synthetic bytes, with the training package's fingerprint module."""
import os

import numpy as np
import pytest

from pbdata import exclusion, manifest, paths
from pbdata.corpus import Candidate, CorpusWriter
from pbdata.util import DataError


def rand_bytes(n, seed):
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


def rand_text(n_lines, seed):
    rng = np.random.default_rng(seed)
    words = ["alpha", "beta", "gamma", "delta", "config", "value", "return", "import", "server", "token_name"]
    lines = []
    for _ in range(n_lines):
        k = int(rng.integers(3, 9))
        lines.append(" ".join(words[int(i)] for i in rng.integers(0, len(words), k)) + f" {int(rng.integers(0, 10**9))}")
    return ("\n".join(lines) + "\n").encode()


def exam_set(name, content, F):
    return exclusion.ExamFingerprints(name, "test", F.fingerprints(content), {"manifest_sha256": "x"}, "<memory>")


def test_the_algorithm_is_the_training_one(F):
    assert F.__name__ == "pbtrain.data.fingerprint"
    assert paths.fingerprint_module() is F


def test_shared_region_is_detected_and_cut_from_text(F):
    exam = rand_text(400, 1)
    ex = exam_set("exam-a", exam, F)
    cf = exclusion.ContentFilter([ex])
    leaked = exam[5000:6200]
    leaked = leaked[leaked.index(b"\n") + 1:]
    candidate = rand_text(200, 2) + leaked + rand_text(200, 3)
    assert "exam-a" in cf.shared(candidate)
    clean, cut = cf.clean_text(candidate)
    assert clean is not None and cut["exam-a"] > 0
    assert cf.shared(clean) == {}
    assert len(clean) < len(candidate) and leaked not in clean
    # what was not near the shared region is kept, line for line
    assert rand_text(200, 2)[:3000] in clean
    assert cf.stats["exam-a"]["files_trimmed"] == 1


def test_clean_text_is_untouched(F):
    cf = exclusion.ContentFilter([exam_set("exam-a", rand_text(300, 1), F)])
    text = rand_text(300, 9)
    clean, cut = cf.clean_text(text)
    assert clean == text and cut == {}


def test_file_that_is_mostly_exam_is_dropped(F):
    exam = rand_text(300, 1)
    cf = exclusion.ContentFilter([exam_set("exam-a", exam, F)])
    clean, cut = cf.clean_text(exam[:8000] + b"x = 1\n")
    assert clean is None
    assert cf.stats["exam-a"]["files_dropped"] == 1 and cf.stats["exam-a"]["bytes_dropped"] > 0


def test_binary_gets_one_clean_span_or_is_dropped(F):
    exam = rand_bytes(40000, 4)
    cf = exclusion.ContentFilter([exam_set("bin", exam, F)])
    binary = rand_bytes(60000, 5) + exam[10000:14000] + rand_bytes(90000, 6)
    span = cf.clean_span(binary, span_max=32768, min_span=4096, pick=12345)
    assert span is not None
    a, z = span
    assert z - a <= 32768 and z - a >= 4096
    assert cf.shared(binary[a:z]) == {}
    # never spliced: the span is a slice of the original file
    assert binary.find(binary[a:z]) == a
    # a file made of exam bytes has no clean span: dropped whole
    assert cf.clean_span(exam, span_max=32768, min_span=4096, pick=1) is None
    assert cf.stats["bin"]["files_dropped"] == 1


def test_symbol_anchor_span_starts_at_the_anchor(F):
    cf = exclusion.ContentFilter([exam_set("bin", rand_bytes(20000, 7), F)])
    blob = rand_bytes(50000, 8)
    blob = blob[:30000] + b"\x00_ZN3foo3barEv" + blob[30000:]
    a, z = cf.clean_span(blob, span_max=8192, min_span=1024, pick=0, anchor=b"\x00_ZN")
    assert blob[a:a + 4] == b"\x00_ZN"


def exam_manifest(names):
    return manifest.from_doc({"schema": 1, "kind": "exam", "name": "tiny", "sets": [
        {"name": n, "role": "test", "entries": [
            {"id": "r", "type": "git", "url": f"https://example.org/{n}", "pin": {"commit": "a" * 40},
             "licence": "MIT"}]} for n in names]}, "tiny.yaml")


def test_missing_exam_fingerprints_fail_the_build(data_home, F):
    m = exam_manifest(["set-one", "set-two"])
    os.makedirs(paths.fingerprints_dir())
    F.save(paths.fingerprint_path("set-one"), F.fingerprints(rand_bytes(5000, 1)),
           {"name": "set-one", "role": "test", "manifest_sha256": m.sha256})
    with pytest.raises(DataError, match="set-two"):
        exclusion.load_all(exams=[m])
    sets, report = exclusion.load_all(allow_missing=True, exams=[m])
    assert [s.name for s in sets] == ["set-one"]
    assert report["missing"] == ["set-two"] and report["allow_missing_exams"] is True
    assert any("allow-missing-exams" in w for w in report["warnings"])


def test_stale_fingerprints_count_as_missing(data_home, F):
    m = exam_manifest(["set-one"])
    os.makedirs(paths.fingerprints_dir())
    F.save(paths.fingerprint_path("set-one"), F.fingerprints(rand_bytes(5000, 1)),
           {"name": "set-one", "role": "test", "manifest_sha256": "made-from-another-version"})
    with pytest.raises(DataError, match="another version"):
        exclusion.load_all(exams=[m])


def test_fingerprint_files_round_trip(data_home, F):
    fp = F.fingerprints(rand_bytes(30000, 2))
    os.makedirs(paths.fingerprints_dir())
    F.save(paths.fingerprint_path("x"), fp, {"role": "test"})
    fp2, meta = F.load(paths.fingerprint_path("x"))
    assert exclusion.fp_digest(fp) == exclusion.fp_digest(fp2)
    assert meta["algorithm"].startswith("anchored-64B")


def test_a_block_formed_at_a_junction_is_found_and_repaired(tmp_path, F):
    # an exam string whose every block straddles position 100 of `x`: neither piece holds a whole block on its own
    for seed in range(200):
        x = rand_bytes(240, seed)
        exam = x[40:140]
        if len(F.fingerprints(exam)):
            break
    else:
        pytest.skip("no anchored block found")
    a = rand_bytes(3000, 11) + x[:100]
    b = x[100:] + rand_bytes(3000, 12)
    cf = exclusion.ContentFilter([exam_set("junction", exam, F)])
    assert cf.shared(a) == {} and cf.shared(b) == {}
    assert cf.shared(a + b)
    w = CorpusWriter("tiny", str(tmp_path), {"g": {"share": 1.0}}, target_bytes=10 ** 6)
    ca, cb = Candidate("g", "s", "a", len(a), "a", None, 0), Candidate("g", "s", "b", len(b), "b", None, 0)
    ca.key, cb.key = "0", "1"                                   # a is written right before b
    content = {"a": a, "b": b}
    pieces, stats = w.select([cb, ca], lambda pool: ((c, content[c.cid]) for c in pool), lambda c, raw: (raw, "ok"))
    w.write(pieces)                                             # the same pieces written without verification...
    with open(w.bin_path, "rb") as f:
        assert cf.shared(f.read())                              # ...leak the exam through their junction
    rows, ver = w.write(pieces, cf)
    with open(w.bin_path, "rb") as f:
        out = f.read()
    assert cf.shared(out) == {}
    assert ver["pieces_removed"] >= 1 and ver["shared_regions"] == 0
