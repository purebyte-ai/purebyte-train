"""The release pool of the pii specialist: labelled documents of the TRAIN splits of four public PII datasets, rebuilt
from pinned downloads, and the HARD OVERLAP CHECK that keeps every evaluation set out of it (data/sources/pii-pool.yaml,
build_pii_pool.py).

The pool (`pii/public_train_xl/`, one JSONL file per source; `pii/public_train_clean/` = its first file alone):
    base_public_train   the sample of data/sources/pii.yaml (fetch_pii.py train), drawn here from the same pinned files
    openpii1m_xl        OpenPII-1M TRAIN in its 23 languages: English 48,000, every other language 7,000 (a hash of the
                        document id decides, not the file order), after the check
    nemotron_xl         every Nemotron-PII TRAIN document
    gretel_en_v1_xl     every Gretel PII masking EN v1 TRAIN document
    gretel_finance_xl   every Gretel synthetic PII finance TRAIN document (7 languages)
A document whose normalised text is already in the pool is not written twice. Every line: {id, source, lang, text,
labels: [[char start, char end, source label]]}, the format of `pii/public_train/`.

The hard overlap check, every candidate, before it is written (a document that fails is never written; its id goes to
HARD_CHECK_EXCLUDED.json with the level):
    exact_doc     its normalised text (lower case, runs of white space as one space) is an evaluation document's
    exact_unit    one of its units (a sentence or a line, normalised, of 40 characters or more) is an evaluation unit
    masked_doc    its MASKED TEMPLATE (every labelled span replaced by one placeholder) is an evaluation document's
    masked_unit   one of its masked units (60 characters or more, holding a placeholder) is a masked evaluation unit:
                  the same sentence with other values, a template shared by a generator's train and test splits
The evaluation side: the PII masking benchmark (documents and sentences, their entities), the pii-bench-test and
pii-bench-dev documents (their labels mapped by the pii specialist's `public_map`, as the evaluation reads them) and,
optionally, more documents (`--extra-test`). Units are cut the same way on both sides (sentence end or line break), so
an identical region gives identical units.

On top of it the data scripts' usual content check runs (64-byte blocks against the exam fingerprints, data/README.md):
with `--exam-overlap keep` it is REPORTED (the pool is then the one the release was trained with, and the training
factory refuses it: see data/README.md), with `drop` every document that shares a block with a `test` exam is left
out too.
"""
import ast
import bisect
import hashlib
import json
import os
import re

import numpy as np

from .util import DataError

MIN_UNIT, MIN_MASKED_UNIT = 40, 60
LEVELS = ("exact_doc", "exact_unit", "masked_doc", "masked_unit")
RULE = dict(exact_doc="normalised text == a test document",
            exact_unit=f"a unit >= {MIN_UNIT} chars == a test unit",
            masked_doc="masked template == a test document's",
            masked_unit=f"a masked unit >= {MIN_MASKED_UNIT} chars with a placeholder == a masked test unit")
OUTPUTS = ("base_public_train", "openpii1m_xl", "nemotron_xl", "gretel_en_v1_xl", "gretel_finance_xl")
# OpenPII-1M: rows per language of the dataset's distribution (train + validation); the languages not listed have about
# 45,500. A row is a candidate when a hash of its id falls under oversample x quota / (0.8 x rows): the check and the
# cap absorb the excess, so an excluded document never leaves a language short.
OPENPII_ROWS = {"en": 179419, "fr": 131173, "de": 120105, "es": 90254, "it": 89304}
OPENPII_OTHER_ROWS = 45500
FINANCE_LANG = {"English": "en", "Spanish": "es", "French": "fr", "France": "fr", "German": "de", "Italian": "it",
                "Dutch": "nl", "Swedish": "sv"}                   # the dataset writes «France» for its French rows
_SPLIT = re.compile(r"(?<=[.!?])\s+|\s*\n\s*")


# ------------------------------------------------------------------------------------------------ keys of a document
def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "little")


def norm(s):
    return " ".join(s.lower().split())


def masked(text, spans):
    """Every labelled span (character offsets) -> a placeholder; a span inside an earlier one adds nothing."""
    out, last = [], 0
    for a, b in sorted((int(a), int(b)) for a, b, *_ in spans):
        if b <= last:
            continue
        a = max(a, last)
        out.append(text[last:a])
        out.append(" § ")
        last = b
    out.append(text[last:])
    return "".join(out)


def units(text, minlen, need_mask=False):
    out = []
    for u in _SPLIT.split(text):
        n = norm(u)
        if len(n) >= minlen and (not need_mask or "§" in n):
            out.append(h64(n))
    return out


def keys_of(text, spans):
    """(document key, unit keys, masked document key, masked unit keys)."""
    m = masked(text, spans)
    return h64(norm(text)), units(text, MIN_UNIT), h64(norm(m)), units(m, MIN_MASKED_UNIT, need_mask=True)


def byte_to_char(text, items):
    """[(byte start, byte end, ...)] of the UTF-8 text -> [(char start, char end)], rounding outwards."""
    cum, acc = [0], 0
    for c in text:
        acc += len(c.encode("utf-8"))
        cum.append(acc)
    return [(bisect.bisect_right(cum, a) - 1, bisect.bisect_left(cum, b)) for a, b, *_ in items]


class TestIndex:
    """The four kinds of key of every evaluation document; `reason()` is the check."""

    def __init__(self):
        self.doc, self.unit, self.mdoc, self.munit = set(), set(), set(), set()
        self.counts = {}

    def add(self, name, text, spans):
        d, u, md, mu = keys_of(text, spans)
        self.doc.add(d)
        self.unit.update(u)
        self.mdoc.add(md)
        self.munit.update(mu)
        self.counts[name] = self.counts.get(name, 0) + 1

    def reason(self, text, spans):
        """The first level a document fails (in the order of LEVELS), or None."""
        d, u, md, mu = keys_of(text, spans)
        if d in self.doc:
            return "exact_doc"
        if any(x in self.unit for x in u):
            return "exact_unit"
        if md in self.mdoc:
            return "masked_doc"
        if any(x in self.munit for x in mu):
            return "masked_unit"
        return None

    def summary(self):
        return dict(documents=dict(self.counts), units=len(self.unit), masked_units=len(self.munit))


def index_piimb(idx, root):
    """The benchmark's documents and sentences with their entities (character offsets)."""
    for fn in ("test.jsonl", "test_sentences.jsonl"):
        p = os.path.join(root, fn)
        if not os.path.exists(p):
            raise DataError(f"{p} is missing: fetch the evaluation sets first (python data/fetch_exams.py --set piimb)")
        with open(p, encoding="utf-8") as f:
            for ln in f:
                r = json.loads(ln)
                idx.add(f"piimb/{fn}", r["text"], [(e["start"], e["end"]) for e in r["entities"]])


def index_bench(idx, root, name, public_map):
    """A pii-bench set: every JSONL file of the set, its SOURCE labels mapped as the evaluation reads them (the typed
    spans and the ignored regions of the pii specialist's `public_map`), in characters."""
    files = sorted(f for f in os.listdir(root) if f.endswith(".jsonl")) if os.path.isdir(root) else []
    if not files:
        raise DataError(f"{root} holds no documents: fetch the evaluation sets first (python data/fetch_exams.py "
                        f"--set {name})")
    for fn in files:
        with open(os.path.join(root, fn), encoding="utf-8") as f:
            for ln in f:
                if not ln.strip():
                    continue
                d = json.loads(ln)
                text = d["text"]
                sp, ig = public_map.map_doc(text, [tuple(x) for x in d["labels"]], d["source"])
                b = public_map.to_bytes(text, sp) + public_map.to_bytes(text, ig)
                idx.add(name, text, byte_to_char(text, b))


def index_extra(idx, path, name="extra"):
    """More evaluation documents: one JSON object per line with `text` (or `data_b64`, bytes read as UTF-8)."""
    import base64
    with open(path, encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            r = json.loads(ln)
            text = r["text"] if "text" in r else base64.b64decode(r["data_b64"]).decode("utf-8", "replace")
            idx.add(name, text, [])


# ------------------------------------------------------------------------------------------------ the sources
def pool_doc(uid, source, lang, text, labs):
    """A pool document: labels outside the text or blank are dropped, the rest sorted."""
    out = [[int(a), int(b), str(lab)] for a, b, lab in labs if 0 <= int(a) < int(b) <= len(text)
           and text[int(a):int(b)].strip()]
    return dict(id=uid, source=source, lang=lang, text=text, labels=sorted(out))


def openpii_quota(en, per_lang):
    return lambda lang: en if lang == "en" else (per_lang if lang else 0)


def openpii_candidates(path, quota, oversample):
    """{language: [document]} of OpenPII-1M TRAIN rows (the file's own `split` field), drawn by a hash of the row id."""
    kept = {}
    with open(path, encoding="utf-8") as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except ValueError:
                continue
            if r.get("split", "train") != "train":
                continue
            lang = r.get("language")
            q = quota(lang)
            if not q:
                continue
            total = OPENPII_ROWS.get(lang, OPENPII_OTHER_ROWS) * 0.8
            rate = min(1.0, oversample * q / total)
            uid = str(r.get("uid"))
            if (h64("openpii:" + uid) % 1_000_000) / 1e6 >= rate:
                continue
            n = kept.setdefault(lang, [])
            if len(n) >= int(2 * oversample * q):                    # a memory guard only: the rate targets the quota
                continue
            t = r["source_text"]
            labs = []
            for m in r.get("privacy_mask") or []:
                a, b = int(m["start"]), int(m["end"])
                if t[a:b] != m["value"]:
                    j = t.find(m["value"])
                    if j < 0:
                        continue
                    a, b = j, j + len(m["value"])
                labs.append((a, b, m["label"]))
            n.append(pool_doc(f"ai4p-train-{uid}", "ai4privacy/pii-masking-openpii-1m", lang, t, labs))
    return kept


def openpii_order(kept):
    """The candidates in a seeded order: by language, then by a hash of the id (never the file's order)."""
    docs = []
    for lang in sorted(kept):
        docs += sorted(kept[lang], key=lambda d: h64("order:" + d["id"]))
    return docs


def nemotron_docs(rows):
    for r in rows:
        t, sp = r["text"], r["spans"]
        if isinstance(sp, str):
            sp = json.loads(sp) if sp.startswith('[{"') else ast.literal_eval(sp)
        labs = []
        for s in sp or []:
            a, b = int(s["start"]), int(s["end"])
            st = None if s.get("text") is None else str(s["text"])
            if st is not None and t[a:b] != st:
                j = t.find(st, max(0, a - 20))
                if j < 0:
                    continue
                a, b = j, j + len(st)
            labs.append((a, b, s["label"]))
        yield pool_doc(f"Nemotron-PII-train-uid-{r['uid']}", "nvidia/Nemotron-PII", "en", t, labs)


def gretel_en_docs(rows):
    """Gretel PII masking EN v1 lists entities without offsets: every occurrence of an entity's text is labelled."""
    for r in rows:
        t, ents = r["text"], r["entities"]
        if isinstance(ents, str):
            try:
                ents = json.loads(ents)
            except ValueError:
                ents = ast.literal_eval(ents)
        labs = []
        for e in ents or []:
            s = str(e.get("entity", "")).strip()
            typ = (e.get("types") or [e.get("type", "pii")])[0]
            if len(s) < 2:
                continue
            for m in re.finditer(re.escape(s), t):
                labs.append((m.start(), m.end(), typ))
        yield pool_doc(f"gretel-pii-masking-en-v1-train-uid-{r['uid']}", "gretelai/gretel-pii-masking-en-v1", "en", t,
                       labs)


def gretel_finance_docs(rows):
    for r in rows:
        t, sp = r["generated_text"], r["pii_spans"]
        sp = json.loads(sp) if isinstance(sp, str) else sp
        labs = [(int(s["start"]), int(s["end"]), s["label"]) for s in sp or []]
        yield pool_doc(f"synthetic_pii_finance_multilingual-train-idx-{r['index']}",
                       "gretelai/synthetic_pii_finance_multilingual", FINANCE_LANG.get(r["language"], "?"), t, labs)


def parquet_rows(path, columns=None):
    """The rows of a Parquet file as dicts, in file order (pyarrow; only this builder needs it)."""
    try:
        import pyarrow.parquet as pq
    except ImportError:
        raise DataError("reading the Parquet files of the pool needs pyarrow (`pip install pyarrow`); it is the one "
                        "extra package of the data scripts, used by build_pii_pool.py only") from None
    return pq.read_table(path, columns=columns).to_pylist()


def rows_api(path):
    """rows(dataset, split, offset, length) -> (rows, total) over a local Parquet file: what `pii.train_paged` reads
    from the rows API, in the same order, without the network."""
    cache = {}

    def rows(dataset, split, offset, length):
        if "t" not in cache:
            cache["t"] = parquet_rows(path)
        t = cache["t"]
        return t[offset:offset + length], len(t)
    return rows


# ------------------------------------------------------------------------------------------------ writing
def jsonl_line(d):
    return json.dumps(d, ensure_ascii=False) + "\n"


def text_key(text):
    """The 16-hex key of a document's normalised text (what the pool deduplicates on)."""
    return f"{h64(norm(text)):016x}"


class PoolWriter:
    """Writes the pool's files: every candidate is deduplicated against the documents already written (normalised
    text), checked against the test index, against `listed` (text keys of documents to leave out, with the reason
    `listed`) and, optionally, against the exam fingerprints (`block_filter(text)` -> exam name or None); the rest is
    written, LF line ends."""

    def __init__(self, out_dir, index, block_filter=None, listed=()):
        self.out_dir, self.index, self.block_filter = out_dir, index, block_filter
        self.listed = set(listed)
        self.seen, self.excluded, self.stats = set(), [], {}

    def _why(self, d):
        why = self.index.reason(d["text"], d["labels"])
        if why is None and self.listed and text_key(d["text"]) in self.listed:
            why = "listed"
        if why is None and self.block_filter is not None:
            hit = self.block_filter(d["text"])
            why = f"exam_block:{hit}" if hit else None
        return why

    def emit(self, name, docs, **extra):
        p = os.path.join(self.out_dir, f"{name}.jsonl")
        st = dict(candidates=0, written=0, excluded={}, duplicates=0, langs={})
        h = hashlib.sha256()
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for d in docs:
                st["candidates"] += 1
                k = h64(norm(d["text"]))
                if k in self.seen:
                    st["duplicates"] += 1
                    continue
                why = self._why(d)
                if why:
                    st["excluded"][why] = st["excluded"].get(why, 0) + 1
                    self.excluded.append(dict(id=d["id"], source=d["source"], reason=why))
                    continue
                self.seen.add(k)
                line = jsonl_line(d)
                f.write(line)
                h.update(line.encode("utf-8"))
                st["written"] += 1
                st["langs"][d["lang"]] = st["langs"].get(d["lang"], 0) + 1
        st.update(sha256=h.hexdigest(), **extra)
        self.stats[name] = st
        return st

    def prefilter(self, docs, quota):
        """OpenPII: the check BEFORE the per-language cap, so the cap counts clean documents only."""
        ok, out, pre = {}, [], {}
        for d in docs:
            n = ok.get(d["lang"], 0)
            if n >= quota(d["lang"]):
                continue
            why = self._why(d)
            if why:
                self.excluded.append(dict(id=d["id"], source=d["source"], reason=why))
                pre[why] = pre.get(why, 0) + 1
                continue
            ok[d["lang"]] = n + 1
            out.append(d)
        return out, pre


# ------------------------------------------------------------------------------------------------ content blocks
def texts_sharing_blocks(texts, fp):
    """bool [len(texts)]: the text shares at least one 64-byte fingerprint block with `fp` (sorted unique uint64, the
    training package's algorithm)."""
    from . import paths
    F = paths.fingerprint_module()
    fp = np.asarray(fp, dtype=np.uint64)
    hit = np.zeros(len(texts), dtype=bool)
    if not len(fp):
        return hit
    for i, t in enumerate(texts):
        f = F.fingerprints(t.encode("utf-8") if isinstance(t, str) else t)
        if len(f):
            j = np.searchsorted(fp, f)
            j[j >= len(fp)] = len(fp) - 1
            hit[i] = bool((fp[j] == f).any())
    return hit


def fingerprint_texts(texts):
    """Sorted unique fingerprints of many texts, each fingerprinted on its own (no block spans two texts)."""
    from . import paths
    F = paths.fingerprint_module()
    sep = b"\0" * 80                                   # blocks holding a NUL match no text: none spans two documents
    parts = [t.encode("utf-8") if isinstance(t, str) else t for t in texts]
    return F.fingerprints(np.frombuffer(sep.join(parts), dtype=np.uint8)) if parts else np.zeros(0, np.uint64)
