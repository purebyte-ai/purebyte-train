"""The binary corpus builder on synthetic artifacts: format recognition, member-level identity, clean spans."""
import io
import os
import zipfile

import numpy as np
import pytest

import build_binary_corpus as BB
from pbdata import exclusion
from pbdata.corpus import Candidate, CorpusWriter
from pbdata.util import DataError, sha256_bytes

BUILD = {"member_bytes": {"min": 64, "max": 1 << 24}, "group_member_min": {"pyc": 32}, "span": {"max": 4096, "min": 512},
         "symbols": {"min_count": 5}, "seed": 0}


def noise(n, seed):
    return np.random.default_rng(seed).integers(0, 256, n, dtype=np.uint8).tobytes()


def elf(n=3000, seed=1, symbols=0):
    return b"\x7fELF\x02\x01\x01" + b"\0" * 9 + noise(n, seed) + b"\x00_ZN4core6detail" * symbols


def pe(n=3000, seed=2):
    head = bytearray(b"MZ" + b"\0" * 0x3A + (0x80).to_bytes(4, "little") + b"\0" * (0x80 - 0x40))
    return bytes(head) + b"PE\0\0" + noise(n, seed)


@pytest.mark.parametrize("name,data,group", [
    ("usr/bin/tool", elf(), "elf"),
    ("usr/lib/libq.so", elf(symbols=10), "elf_symbols"),
    ("usr/lib/liba.a", b"!<arch>\n" + noise(500, 3), "ar"),
    ("x/a.cpython-313.pyc", b"\xf3\r\r\n" + noise(100, 4), "pyc"),
    ("python.exe", pe(), "pe"),
    ("MZ-but-not-pe.bin", b"MZ" + noise(300, 5), None),
    ("lib.dylib", b"\xcf\xfa\xed\xfe" + noise(300, 6), "macho"),
    ("universal", b"\xca\xfe\xba\xbe\x00\x00\x00\x02" + noise(300, 7), "macho"),
    ("a/B.class", b"\xca\xfe\xba\xbe\x00\x00\x00\x34" + noise(300, 8), "java"),
    ("lib.jar", b"PK\x03\x04" + noise(300, 9), "java"),
    ("grammar.wasm", b"\0asm\x01\0\0\0" + noise(300, 10), "wasm"),
    ("README", b"hello world" * 20, None),
])
def test_detect_group(name, data, group):
    assert BB.detect_group(name, data, None, symbols_min=5) == group


def test_the_format_comes_from_the_entry_not_the_cache_path():
    from pbdata.archives import entry_format
    assert entry_format({"type": "deb", "url": "https://snapshot.example.org/file/0123abcd"}) == "deb"
    assert entry_format({"type": "pypi", "url": "https://files.example.org/x-1.0-py3-none-any.whl"}) == "zip"
    assert entry_format({"type": "url", "url": "https://example.org/go1.27.1.darwin-arm64.tar.gz"}) == "tar"
    assert entry_format({"type": "url", "url": "https://example.org/tree-sitter-c.wasm"}) == "raw"
    assert entry_format({"type": "url", "url": "https://example.org/a.jar", "format": "raw"}) == "raw"


def test_forced_group_wins():
    assert BB.detect_group("bios.bin", noise(400, 1), "firmware") == "firmware"


def _artifact(tmp_path, members):
    p = tmp_path / ("0" * 64)                     # named like the download cache: by digest, without an extension
    with zipfile.ZipFile(p, "w") as z:
        for name, data in members.items():
            z.writestr(name, data)
    return str(p)


def entry(**kw):
    e = {"id": "art", "type": "url", "url": "https://example.org/artifact.zip", "pin": {"sha256": "0" * 64},
         "licence": "MIT", "take": {"members": ["**/*"]}}
    e.update(kw)
    return e


def test_enumerate_recognises_and_caches(data_home, tmp_path):
    members = {"bin/tool": elf(seed=11), "bin/tool.exe": pe(seed=12), "lib/libq.so": elf(seed=13, symbols=10),
               "doc/readme.txt": b"text " * 100, "bin/copy": elf(seed=11)}
    cands, rec = BB.enumerate_entry(entry(), _artifact(tmp_path, members), BUILD, 0, exam_digests={})
    groups = sorted((c.path, c.group) for c in cands)
    assert groups == [("bin/copy", "elf"), ("bin/tool", "elf"), ("bin/tool.exe", "pe"), ("lib/libq.so", "elf_symbols")]
    assert all(os.path.exists(c.ref[0]) for c in cands)
    assert rec["candidates"] == 4 and rec["candidate_groups"] == {"elf": 2, "elf_symbols": 1, "pe": 1}


def test_a_member_that_is_an_exam_file_stops_the_build(data_home, tmp_path):
    secret_exam_file = pe(seed=21)
    with pytest.raises(DataError, match="identity exclusion failed"):
        BB.enumerate_entry(entry(), _artifact(tmp_path, {"x.dll": secret_exam_file}), BUILD, 0,
                           exam_digests={sha256_bytes(secret_exam_file): "bin-set:e:x.dll"})


def test_spans_avoid_exam_blocks_and_symbol_spans_start_at_a_symbol(data_home, tmp_path, F):
    exam_bytes = noise(20000, 30)
    ex = exclusion.ExamFingerprints("bin", "test", F.fingerprints(exam_bytes), {}, "<memory>")
    cf = exclusion.ContentFilter([ex])
    shared_file = b"\x7fELF" + noise(3000, 31) + exam_bytes[:6000] + noise(9000, 32)
    sym_file = b"\x7fELF" + noise(2000, 33) + b"\x00_ZN3foo3barEv\x00_ZN3foo3bazEv" * 40 + noise(2000, 34)
    t = BB.make_transform(cf, BUILD, "clean-span")
    c1 = Candidate("elf", "s", "a", len(shared_file), "a", None, 0)
    out, why = t(c1, shared_file)
    assert out is not None and cf.shared(out) == {} and len(out) <= 4096
    c2 = Candidate("elf_symbols", "s", "b", len(sym_file), "b", None, 0)
    out2, _ = t(c2, sym_file)
    assert out2.startswith(b"\x00_ZN")
    # the literal policy drops every file that shares anything
    t2 = BB.make_transform(exclusion.ContentFilter([ex]), BUILD, "drop-file")
    assert t2(c1, shared_file) == (None, "exam-overlap")


def test_region_of_large_members_is_deterministic():
    big = noise(BB.REGION_MAX + 5000, 40)
    a = BB.region_of(big, "elf", 123456789)
    b = BB.region_of(big, "elf", 123456789)
    assert a == b and len(a[1]) == BB.REGION_MAX
    small = noise(1000, 41)
    assert BB.region_of(small, "elf", 5) == (0, small)


def test_binary_writer_output_is_clean_and_deterministic(data_home, tmp_path, F):
    exam_bytes = noise(30000, 50)
    ex = exclusion.ExamFingerprints("bin", "test", F.fingerprints(exam_bytes), {}, "<memory>")
    members = {f"f{i}.so": elf(n=6000, seed=100 + i) for i in range(6)}
    members["mixed.so"] = b"\x7fELF" + noise(5000, 60) + exam_bytes[:8000] + noise(5000, 61)
    cands, _ = BB.enumerate_entry(entry(), _artifact(tmp_path, members), BUILD, 0, exam_digests={})

    def run(out):
        cf = exclusion.ContentFilter([ex])
        w = CorpusWriter("b", str(out), {"elf": {"share": 1.0}}, target_bytes=40000)
        pieces, stats = w.select(cands, BB.cached_stream, BB.make_transform(cf, BUILD, "clean-span"))
        w.write(pieces, cf)
        with open(w.bin_path, "rb") as f:
            data = f.read()
        return data, cf

    d1, cf1 = run(tmp_path / "o1")
    d2, _ = run(tmp_path / "o2")
    assert d1 == d2 and cf1.shared(d1) == {}
