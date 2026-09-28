"""The PII pool (pbdata/pii_pool.py, build_pii_pool.py) offline: the four levels of the hard overlap check, the samplers
of the four sources on made-up rows, the writer (deduplication, exclusion, LF lines, digests), the local sources of the
base sample, the content-block helpers and the manifest. No network, no pyarrow, no benchmark file: every text here is
made up (words of syllables), and no value is a real one."""
import json
import os

import numpy as np
import pytest

from pbdata import identity, manifest, pii, pii_pool
from pbdata.paths import SOURCE_MANIFESTS

SYL = ("ka", "lo", "mi", "ne", "ru", "sa", "ti", "vo", "ze", "pa")


def word(g, n=3):
    return "".join(SYL[int(i)] for i in g.integers(0, len(SYL), n)).capitalize()


LONG = ("The committee reviewed the quarterly report and approved the new schedule for all regional offices "
        "before the end of the month.")


def doc_with(name, city):
    """A made-up document and its spans (character offsets): a sentence with a person and a place, then boilerplate."""
    first = f"Dear {name}, your parcel for {city} is ready."
    text = first + " " + LONG
    a = text.index(name)
    b = text.index(city)
    return text, [(a, a + len(name), "NAME"), (b, b + len(city), "CITY")]


@pytest.fixture
def index():
    idx = pii_pool.TestIndex()
    text, spans = doc_with("Kalomi Ruvo", "Satize")
    idx.add("test", text, spans)
    return idx


def test_the_four_levels(index):
    text, spans = doc_with("Kalomi Ruvo", "Satize")
    assert index.reason(text, spans) == "exact_doc"
    assert index.reason("  " + text.upper() + "\n", spans) == "exact_doc", "normalised: case and white space"
    other = f"Something else entirely happened here. {LONG}"
    assert index.reason(other, []) == "exact_unit", "a shared sentence of 40 characters or more"
    t2, s2 = doc_with("Tizena Vopa", "Mirulo")
    t2 = t2.replace(LONG, "A short ending.")
    base_t, base_s = doc_with("Kalomi Ruvo", "Satize")
    idx = pii_pool.TestIndex()
    idx.add("test", base_t.replace(LONG, "A short ending."), base_s)
    assert idx.reason(t2, s2) == "masked_doc", "the same template with other values"
    tmpl = "Please confirm your attendance at the annual meeting by replying to Kalomi Ruvo before noon."
    a = tmpl.index("Kalomi Ruvo")
    idx2 = pii_pool.TestIndex()
    idx2.add("test", "Unrelated opening line. " + tmpl, [(24 + a, 24 + a + 11)])
    t3 = tmpl.replace("Kalomi Ruvo", "Zenatiru Palo") + " Another closing remark follows."
    b = t3.index("Zenatiru Palo")
    assert idx2.reason(t3, [(b, b + 13, "NAME")]) == "masked_unit", "a shared masked sentence with a placeholder"
    assert idx2.reason(t3, []) is None, "without its label the sentence is another sentence"
    assert index.reason("A clean document with nothing in common. Really nothing at all here.", []) is None


def test_short_units_do_not_count(index):
    assert index.reason("Dear Tizena, see you soon.", []) is None
    assert pii_pool.units("tiny. line.", pii_pool.MIN_UNIT) == []


def test_masked_templates():
    assert pii_pool.masked("abcdefgh", [(1, 6), (2, 4)]) == "a § gh", "a span inside another adds nothing"
    assert pii_pool.masked("abcdefgh", [(1, 4), (2, 6)]) == "a §  § gh"
    assert pii_pool.masked("abcdefgh", [(1, 2), (2, 3)]) == "a §  § defgh"


def test_byte_to_char_rounds_outwards():
    text = "añb€c"
    b = text.encode("utf-8")
    assert pii_pool.byte_to_char(text, [(0, len(b))]) == [(0, 5)]
    assert pii_pool.byte_to_char(text, [(2, 3)]) == [(1, 2)], "a boundary inside ñ takes the whole character"
    assert pii_pool.byte_to_char(text, [(4, 7)]) == [(3, 4)]


def test_the_benchmark_and_bench_readers(tmp_path):
    g = np.random.default_rng(1)
    piimb = tmp_path / "piimb"
    piimb.mkdir()
    rows = []
    for i in range(4):
        name = f"{word(g)} {word(g)}"
        text = f"Contact {name} today."
        rows.append(dict(uid=f"t:{i}", task_name="gretel", text=text,
                         entities=[dict(start=8, end=8 + len(name), label="name")]))
    for fn in ("test.jsonl", "test_sentences.jsonl"):
        (piimb / fn).write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    bench = tmp_path / "bench"
    bench.mkdir()
    name = f"{word(g)} {word(g)}"
    d = dict(id="b1", source="nvidia/Nemotron-PII", lang="en", text=f"Patient {name} works at the {word(g)} lab.",
             labels=[[8, 8 + len(name), "first_name"]])
    (bench / "nemotron-pii.jsonl").write_text(json.dumps(d) + "\n", encoding="utf-8")
    idx = pii_pool.TestIndex()
    pii_pool.index_piimb(idx, str(piimb))
    from pbdata import paths
    pii_pool.index_bench(idx, str(bench), "pii-bench-test", paths.pii_public_map())
    assert idx.counts == {"piimb/test.jsonl": 4, "piimb/test_sentences.jsonl": 4, "pii-bench-test": 1}
    assert idx.reason(rows[0]["text"], [(8, 8 + len(rows[0]["text"]) - 15)]) == "exact_doc"
    with pytest.raises(Exception, match="fetch"):
        pii_pool.index_piimb(pii_pool.TestIndex(), str(tmp_path / "missing"))


def test_openpii_candidates_are_drawn_by_id_and_capped(tmp_path, monkeypatch):
    monkeypatch.setattr(pii_pool, "OPENPII_ROWS", {"en": 100, "de": 100})
    monkeypatch.setattr(pii_pool, "OPENPII_OTHER_ROWS", 100)
    g = np.random.default_rng(2)
    p = tmp_path / "train.jsonl"
    with open(p, "w", encoding="utf-8", newline="\n") as f:
        for i in range(400):
            lang = ("en", "de", "el")[i % 3]
            name = word(g)
            t = f"Hello {name}, welcome."
            f.write(json.dumps(dict(uid=str(1000 + i), language=lang, split="validation" if i % 17 == 0 else "train",
                                    source_text=t, privacy_mask=[dict(start=6, end=6 + len(name), value=name,
                                                                      label="GIVENNAME")])) + "\n")
        f.write("not json\n")
    quota = pii_pool.openpii_quota(20, 5)
    kept = pii_pool.openpii_candidates(str(p), quota, 2.5)
    again = pii_pool.openpii_candidates(str(p), quota, 2.5)
    assert kept == again and set(kept) == {"en", "de", "el"}
    assert all(len(v) <= int(2 * 2.5 * quota(k)) for k, v in kept.items())
    docs = pii_pool.openpii_order(kept)
    assert all(d["id"].startswith("ai4p-train-") and d["labels"][0][2] == "GIVENNAME" for d in docs)
    ids = {int(d["id"].rsplit("-", 1)[1]) - 1000 for d in docs}
    assert not any(i % 17 == 0 for i in ids), "rows of another split are never read"
    w = pii_pool.PoolWriter(str(tmp_path), pii_pool.TestIndex())
    out, pre = w.prefilter(docs, quota)
    per = {}
    for d in out:
        per[d["lang"]] = per.get(d["lang"], 0) + 1
    assert per == {"en": 20, "de": 5, "el": 5} and pre == {}


def test_the_parquet_sources_on_made_up_rows():
    nem = [dict(uid="u1", text="Call Kalo at home.",
                spans=json.dumps([dict(start=5, end=9, text="Kalo", label="first_name")])),
           dict(uid="u2", text="Mail Ruvo now.", spans=str([dict(start=0, end=3, text="Ruvo", label="first_name")]))]
    d1, d2 = list(pii_pool.nemotron_docs(nem))
    assert d1["labels"] == [[5, 9, "first_name"]] and d2["labels"] == [[5, 9, "first_name"]], "realigned by its text"
    assert d1["id"] == "Nemotron-PII-train-uid-u1" and d1["lang"] == "en"
    gen = [dict(uid="g1", text="Satize met Satize.", entities=json.dumps([dict(entity="Satize", types=["name"])]))]
    (d,) = list(pii_pool.gretel_en_docs(gen))
    assert d["labels"] == [[0, 6, "name"], [11, 17, "name"]], "every occurrence of the entity's text"
    fin = [dict(level_0=0, index=7, language="France", generated_text="IBAN of Vopa.",
                pii_spans=json.dumps([dict(start=8, end=12, label="name")]))]
    (d,) = list(pii_pool.gretel_finance_docs(fin))
    assert d["lang"] == "fr" and d["id"].endswith("-idx-7") and d["labels"] == [[8, 12, "name"]]
    blank = pii_pool.pool_doc("x", "s", "en", "ab  cd", [(2, 4, "L"), (0, 9, "L"), (0, 2, "L")])
    assert blank["labels"] == [[0, 2, "L"]], "labels outside the text or on blank text are dropped"


def test_the_writer_deduplicates_excludes_and_writes_lf(tmp_path, index):
    text, spans = doc_with("Kalomi Ruvo", "Satize")
    docs = [pii_pool.pool_doc("a", "s", "en", "One clean document. Nothing shared.", []),
            pii_pool.pool_doc("b", "s", "en", "ONE clean   document. Nothing shared.", []),
            pii_pool.pool_doc("c", "s", "en", text, spans)]
    w = pii_pool.PoolWriter(str(tmp_path), index)
    st = w.emit("part", docs)
    assert (st["written"], st["duplicates"], st["excluded"]) == (1, 1, {"exact_doc": 1})
    raw = (tmp_path / "part.jsonl").read_bytes()
    assert b"\r" not in raw and raw.count(b"\n") == 1
    import hashlib
    assert hashlib.sha256(raw).hexdigest() == st["sha256"]
    assert w.excluded == [dict(id="c", source="s", reason="exact_doc")]
    w2 = pii_pool.PoolWriter(str(tmp_path), pii_pool.TestIndex(),
                             block_filter=lambda t: "piimb" if "Nothing" in t else None)
    st2 = w2.emit("part2", docs[:1])
    assert st2["excluded"] == {"exam_block:piimb": 1} and st2["written"] == 0


def test_local_ranges_read_like_http_ranges(tmp_path):
    p = tmp_path / "f.bin"
    p.write_bytes(bytes(range(100)))
    r = pii.LocalRanges(str(p))
    assert r.size == 100 and r.get(10, 12) == bytes([10, 11, 12]) and r.get(98, 500) == bytes([98, 99])


def test_the_paged_sampler_reads_a_local_rows_function():
    rows = [dict(text=f"Row {i} of {word(np.random.default_rng(i))}.", spans="[]") for i in range(250)]

    def local(dataset, split, offset, length):
        return rows[offset:offset + length], len(rows)
    docs = pii.train_paged("owner/set", "train", 150, seed=1, text_of=lambda r: r["text"],
                           labels_of=lambda r, t: [], rows=local)
    again = pii.train_paged("owner/set", "train", 150, seed=1, text_of=lambda r: r["text"],
                            labels_of=lambda r, t: [], rows=local)
    assert docs == again and len(docs) == 150 and docs[0]["id"].startswith("set-train-")


def test_content_blocks(F):
    g = np.random.default_rng(5)
    shared = " ".join(word(g, 4) for _ in range(150))
    fp = pii_pool.fingerprint_texts(["prefix words " + shared, "an unrelated text " * 20])
    hit = pii_pool.texts_sharing_blocks([shared + " tail", "nothing of it here " * 10, "tiny"], fp)
    assert hit.tolist() == [True, False, False]


def test_the_pool_manifest():
    m = manifest.load(os.path.join(SOURCE_MANIFESTS, "pii-pool.yaml"))
    b = m.doc["build"]
    assert set(b["reference_sha256"]) == {f"{n}.jsonl" for n in pii_pool.OUTPUTS}
    assert b["evaluation_sets"] == ["piimb", "pii-bench-test", "pii-bench-dev"]
    for e in m.entries:
        assert e["type"] == "url" and "/resolve/" in e["url"] and e["pin"]["sha256"] == e["files"][0]["sha256"]
        assert any(s.startswith("hf:") and s.endswith(":train") for s in e["identities"])
    exams = manifest.load_exams()
    assert identity.check([m], exams) == [], "TRAIN files only: no identity in common with an evaluation set"
