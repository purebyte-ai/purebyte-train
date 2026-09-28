"""Made-up stand-ins for the two CPU smoke recipes, written by `data/build_smoke_data.py`.

The smoke recipes (specialists/secrets-code/recipes/smoke-cpu.yaml, specialists/pii/recipes/smoke-cpu.yaml) prove the
factory's machinery on a CPU in minutes. They read, under $PUREBYTE_DATA:

    corpora/smoke-context.bin     a code-like corpus (both recipes)
    catalogs/*.json               credential-format catalogues (the secrets generator; the pii rules)
    pii/lists/*.tsv               the twelve lists of the pii generator
    pii/public_train/*.jsonl      labelled documents in the public TRAIN format (the pii smoke recipe)

This module generates all of them: nothing is downloaded, nothing real is used. They are the generators the test
suites use -- `training/tests/smoke_data.py` for the corpus and the catalogues, `specialists/pii/tests/pii_lookalike.py`
for the lists and the documents -- reproduced here so that a data script never imports a test; data/tests/test_smoke_data
checks that both give the same bytes. The corpus is larger (1.2 MB instead of 400 KB): the pii recipe cuts three
splits of at least 64 KiB each from it.

Nothing here is a real name, street, document or credential: words are made of random syllables, numbers are random
digits, and the two credential formats (`demo_...`, `ACMEX...`) belong to no provider.
"""
import json
import os
import sys
import tempfile

import numpy as np

from . import paths
from .util import ensure_dir, sha256_file, write_json

SMOKE_CORPUS_BYTES = 1_200_000

# ------------------------------------------------------------------------------ corpus and catalogues (smoke_data.py)
WORDS = ("user", "total", "items", "config", "result", "value", "count", "index", "buffer", "request", "response",
         "client", "server", "handler", "cache", "timeout", "retries", "path", "name", "status", "payload", "session")


def code_corpus(path, n_bytes=400_000, seed=0):
    """Deterministic pseudo source code: functions, assignments, config lines, comments, and high-entropy literals that
    are not credentials (digests, ids), which is what makes the classic rival stumble."""
    g = np.random.default_rng(seed)
    w = lambda: WORDS[int(g.integers(0, len(WORDS)))]  # noqa: E731
    out, size = [], 0
    hexd = "0123456789abcdef"
    rnd_hex = lambda n: "".join(hexd[int(i)] for i in g.integers(0, 16, n))  # noqa: E731
    while size < n_bytes:
        k = int(g.integers(0, 9))
        if k >= 6:
            name = ("checksum", "etag", "commit_sha", "digest", "build_id", "revision")[int(g.integers(0, 6))]
            out.append(f'{name} = "{rnd_hex(int(g.choice([32, 40, 64])))}"\n')
            size += len(out[-1])
            continue
        if k == 0:
            s = f"def {w()}_{w()}({w()}, {w()}):\n    {w()} = {w()} * {int(g.integers(1, 99))}\n    return {w()}\n"
        elif k == 1:
            s = f'{w()}_{w()} = "{w()}-{w()}"\n'
        elif k == 2:
            s = f"# {w()} the {w()} before {w()}\n"
        elif k == 3:
            s = f"{w()}:\n  {w()}: {int(g.integers(0, 1000))}\n  {w()}: {w()}\n"
        elif k == 4:
            s = f"if {w()} > {int(g.integers(0, 50))}:\n    {w()}.append({w()})\n"
        else:
            s = f'{w().upper()}_{w().upper()}={w()}\n'
        out.append(s)
        size += len(s)
    data = "".join(out).encode()[:n_bytes]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def catalogs(folder, seed=0):
    """Two made-up formats (`demo_` + 24 lower-alnum, `ACMEX` + 20 upper-alnum) and one example value, generated."""
    g = np.random.default_rng(seed)
    alnum = "abcdefghijklmnopqrstuvwxyz0123456789"
    example = "demo_" + "".join(alnum[int(i)] for i in g.integers(0, len(alnum), 28))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "gitleaks.json"), "w", encoding="utf-8") as f:
        json.dump({"rules": [{"id": "demo-token", "regex": r"\bdemo_[a-z0-9]{24}\b", "kw": ["demotoken"]},
                             {"id": "acmex-key", "regex": r"ACMEX[A-Z0-9]{20}", "kw": ["acmexkey"]}],
                   "tps": [], "fps": ["demo_placeholder_value_for_docs"]}, f)
    with open(os.path.join(folder, "trufflehog.json"), "w", encoding="utf-8") as f:
        json.dump({"providers": ["demoprov", "acmexprov"], "patterns": [], "examples": [example]}, f)
    return folder


# ------------------------------------------------------------------------------ PII lists and documents (pii_lookalike)
SYL = ("ba", "be", "bi", "bo", "bu", "da", "de", "di", "do", "fa", "fe", "fi", "ga", "go", "ka", "ke", "ko", "la",
       "le", "li", "lo", "lu", "ma", "me", "mi", "mo", "na", "ne", "ni", "no", "pa", "pe", "po", "ra", "re", "ri",
       "ro", "sa", "se", "si", "so", "ta", "te", "ti", "to", "va", "ve", "vi", "za", "zo")
STREET_TYPES = ("Calle", "Avenida", "Plaza", "Paseo", "Camino", "Ronda")
STATES = ("CA", "NY", "TX", "WA", "OR", "FL")
LIST_FILES = ("names_es_male.tsv", "names_es_female.tsv", "surnames_es.tsv", "names_en_male.tsv",
              "names_en_female.tsv", "surnames_en.tsv", "streets_es.tsv", "places_us.tsv", "municipalities_es.tsv",
              "street_affixes_en.tsv", "sentences_es.tsv", "sentences_en.tsv")


def word(g, lo=2, hi=4):
    """A capitalised made-up word of `lo`..`hi` syllables."""
    return "".join(SYL[int(i)] for i in g.integers(0, len(SYL), int(g.integers(lo, hi + 1)))).capitalize()


def digits(g, n):
    return "".join(str(int(x)) for x in g.integers(0, 10, n))


def _tsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write("\t".join(header) + "\n")
        for r in rows:
            f.write("\t".join(str(x) for x in r) + "\n")


def _unique(g, n):
    out = set()
    while len(out) < n:
        out.add(word(g))
    return sorted(out)


def lists(folder, seed=0, n=120):
    """The twelve list files the pii generator reads, with made-up entries. Returns (given names, surnames)."""
    g = np.random.default_rng(seed)
    os.makedirs(folder, exist_ok=True)
    firsts, surnames = [], []
    for fn in ("names_es_male", "names_es_female", "names_en_male", "names_en_female"):
        items = _unique(g, n)
        firsts += items
        _tsv(os.path.join(folder, fn + ".tsv"), ("name", "frequency"), [(w, int(g.integers(1, 5000))) for w in items])
    for fn in ("surnames_es", "surnames_en"):
        items = _unique(g, 2 * n)
        surnames += items
        _tsv(os.path.join(folder, fn + ".tsv"), ("surname", "frequency"), [(w, int(g.integers(1, 5000))) for w in items])
    _tsv(os.path.join(folder, "streets_es.tsv"), ("type", "name", "cp", "municipality", "province"),
         [(STREET_TYPES[i % len(STREET_TYPES)], f"{word(g)} {word(g)}", f"{int(g.integers(1000, 52999)):05d}", word(g),
           word(g)) for i in range(n)])
    _tsv(os.path.join(folder, "places_us.tsv"), ("name", "state"), [(word(g), STATES[i % len(STATES)]) for i in range(n)])
    _tsv(os.path.join(folder, "municipalities_es.tsv"), ("code", "name", "province"),
         [(f"{i:05d}", word(g), word(g)) for i in range(n)])
    _tsv(os.path.join(folder, "street_affixes_en.tsv"), ("suffix",),
         [(s,) for s in ("Street", "Avenue", "Road", "Lane", "Drive", "Court")])
    for lang, (verb, prep) in (("es", ("fue a", "con")), ("en", ("went to", "with"))):
        rows = []
        for _ in range(n):
            rows.append(("clean", f"{word(g)} {verb} {word(g).lower()} {prep} {word(g).lower()}."))
            rows.append(("holes", f"{{P}} {verb} {word(g).lower()} {prep} {word(g).lower()}."))
        _tsv(os.path.join(folder, f"sentences_{lang}.tsv"), ("kind", "text"), rows)
    return firsts, surnames


def public_docs(folder, firsts, surnames, n=240, seed=0):
    """Labelled documents in the public TRAIN format ({id, source, lang, text, labels: [[char a, char b, label]]}),
    with the label names of one public dataset (names, e-mail and phone; ssn and city, which only the typed policy
    ignores). Written as `lookalike.jsonl`."""
    g = np.random.default_rng(seed)
    pick = lambda seq: seq[int(g.integers(0, len(seq)))]  # noqa: E731
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "lookalike.jsonl"), "w", encoding="utf-8", newline="\n") as f:
        for i in range(n):
            labels, text = [], ""
            first, last = pick(firsts), pick(surnames)
            fields = [("Customer ", None), (first, "first_name"), (" ", None), (last, "last_name"), (" (", None),
                      (f"{first.lower()}.{last.lower()}@{word(g).lower()}.com", "email"), (") called from ", None),
                      (f"+1 {digits(g, 3)} {digits(g, 3)} {digits(g, 4)}", "phone_number"), (". SSN on file: ", None),
                      (f"{digits(g, 3)}-{digits(g, 2)}-{digits(g, 4)}", "ssn"), (". City: ", None), (word(g), "city"),
                      (".", None)]
            for piece, lab in fields:
                if lab:
                    labels.append([len(text), len(text) + len(piece), lab])
                text += piece
            f.write(json.dumps(dict(id=f"lookalike-{i:05d}", source="nvidia/Nemotron-PII", lang="en", text=text,
                                    labels=labels)) + "\n")
    return folder


# ------------------------------------------------------------------------------ writing them into $PUREBYTE_DATA
def standins_module():
    """`secretsgen.standins` of the secrets-code specialist: the marker file, and how the builds that could read a
    stand-in recognise one (it lives with the generator because all of those builds import the generator's package)."""
    d = os.path.join(paths.REPO, "specialists", "secrets-code")
    if d not in sys.path:
        sys.path.insert(0, d)
    from secretsgen import standins
    return standins


def _same_or_standin(path, rel, recorded):
    """A file that may be (re)written: absent, or a stand-in still in place (its bytes are the recorded ones)."""
    return not os.path.exists(path) or rel in recorded


def build(data_dir, log=print, dry_run=False):
    """Writes the stand-ins the smoke recipes need into `data_dir`, never over real data: the corpus always (it is
    used by nothing else), each of catalogues, lists and documents only when that folder holds none yet (or only
    earlier stand-ins). Records every stand-in in the marker file. Returns {written, kept, marker}."""
    S = standins_module()
    recorded = dict(S.in_place(data_dir))
    written, kept = {}, []

    def note(rel, path):
        written[rel] = sha256_file(path)

    # 1. the corpus: both recipes read it, nothing else does
    corpus = os.path.join(data_dir, *S.SMOKE_CORPUS.split("/"))
    if dry_run:
        log(f"  would write {S.SMOKE_CORPUS} ({SMOKE_CORPUS_BYTES:,} bytes of generated pseudo-code)")
    else:
        ensure_dir(os.path.dirname(corpus))
        tmp = corpus + ".tmp"
        code_corpus(tmp, n_bytes=SMOKE_CORPUS_BYTES)
        if not os.path.exists(corpus) or S.SMOKE_CORPUS in recorded or sha256_file(corpus) == sha256_file(tmp):
            os.replace(tmp, corpus)
            note(S.SMOKE_CORPUS, corpus)
            write_json(os.path.splitext(corpus)[0] + ".manifest.json", dict(
                builder="data/build_smoke_data.py", made_up=True,
                what="generated pseudo-code for the CPU smoke recipes; no real repository, no real credential",
                output=dict(path=S.SMOKE_CORPUS, bytes=os.path.getsize(corpus), sha256=written[S.SMOKE_CORPUS])))
        else:
            os.remove(tmp)
            kept.append(f"{S.SMOKE_CORPUS} (a file that is not the stand-in is there: kept)")

    # 2. the catalogues
    cat_dir = os.path.join(data_dir, "catalogs")
    cats = ("gitleaks.json", "trufflehog.json")
    if all(_same_or_standin(os.path.join(cat_dir, c), f"catalogs/{c}", recorded) for c in cats):
        if dry_run:
            log("  would write catalogs/gitleaks.json and catalogs/trufflehog.json (two made-up formats)")
        else:
            catalogs(cat_dir)
            for c in cats:
                note(f"catalogs/{c}", os.path.join(cat_dir, c))
    else:
        kept.append("catalogs/ (catalogues are there: kept, the smoke recipe uses them)")

    # 3. the PII lists
    list_dir = os.path.join(data_dir, "pii", "lists")
    names = None
    if all(_same_or_standin(os.path.join(list_dir, f), f"pii/lists/{f}", recorded) for f in LIST_FILES):
        if dry_run:
            log(f"  would write the {len(LIST_FILES)} lists of pii/lists/ (made-up words)")
        else:
            names = lists(list_dir)
            for f in LIST_FILES:
                note(f"pii/lists/{f}", os.path.join(list_dir, f))
    else:
        kept.append("pii/lists/ (lists are there: kept)")

    # 4. the labelled documents of the pii smoke recipe
    pub_dir = os.path.join(data_dir, "pii", "public_train")
    present = sorted(f for f in os.listdir(pub_dir) if f.endswith(".jsonl")) if os.path.isdir(pub_dir) else []
    if all(f"pii/public_train/{f}" in recorded for f in present):
        if dry_run:
            log("  would write pii/public_train/lookalike.jsonl (240 made-up labelled documents)")
        else:
            if names is None:                      # real lists are there: the documents still use made-up names
                with tempfile.TemporaryDirectory() as t:
                    names = lists(t)
            public_docs(pub_dir, *names)
            note("pii/public_train/lookalike.jsonl", os.path.join(pub_dir, "lookalike.jsonl"))
    else:
        kept.append("pii/public_train/ (documents are there: kept)")

    marker = os.path.join(data_dir, S.MARKER)
    if not dry_run:
        files = dict(recorded, **written)
        write_json(marker, dict(
            builder="data/build_smoke_data.py",
            what="made-up stand-ins for the CPU smoke recipes. A file listed here is a stand-in while its sha256 is the "
                 "one recorded; the plugins refuse stand-ins unless the recipe's corpus is corpora/smoke-context.bin",
            files=dict(sorted(files.items()))))
    return dict(written=sorted(written), kept=kept, marker=marker)
