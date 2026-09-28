"""Offline stand-ins for $PUREBYTE_DATA in the pii tests, generated at test time.

Made-up words (random syllables) for the public lists, a code-like corpus, labelled documents in the public TRAIN
format and two made-up credential formats. Nothing here is a real name, street, document or credential, and the
repository holds none of the generated values: they are assembled from syllables and random digits when a test runs.
"""
import json
import os

import numpy as np

SYL = ("ba", "be", "bi", "bo", "bu", "da", "de", "di", "do", "fa", "fe", "fi", "ga", "go", "ka", "ke", "ko", "la",
       "le", "li", "lo", "lu", "ma", "me", "mi", "mo", "na", "ne", "ni", "no", "pa", "pe", "po", "ra", "re", "ri",
       "ro", "sa", "se", "si", "so", "ta", "te", "ti", "to", "va", "ve", "vi", "za", "zo")
STREET_TYPES = ("Calle", "Avenida", "Plaza", "Paseo", "Camino", "Ronda")
STATES = ("CA", "NY", "TX", "WA", "OR", "FL")


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
    """The twelve list files the generator reads (piigen/names.py), with made-up entries. Returns (given names, surnames)."""
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


def code_corpus(path, firsts, surnames, n_bytes=1_200_000, seed=0):
    """Deterministic pseudo source code, with author lines, e-mail addresses and IP addresses the code scene re-plants."""
    g = np.random.default_rng(seed)
    pick = lambda seq: seq[int(g.integers(0, len(seq)))]
    out, size = [], 0
    while size < n_bytes:
        k = int(g.integers(0, 9))
        w1, w2 = word(g).lower(), word(g).lower()
        if k == 0:
            s = f"def {w1}_{w2}({w2}, n):\n    {w1} = {w2} * {int(g.integers(1, 99))}\n    return {w1}\n"
        elif k == 1:
            s = f'{w1}_{w2} = "{w2}-{w1}"\n'
        elif k == 2:
            s = f"# {w1} the {w2} before it is used\n"
        elif k == 3:
            s = f"{w1}:\n  {w2}: {int(g.integers(0, 1000))}\n  enabled: true\n"
        elif k == 4:
            s = f"if {w1} > {int(g.integers(0, 50))}:\n    {w2}.append({w1})\n"
        elif k == 5:
            s = f"# Copyright (c) 20{int(g.integers(10, 26))} {pick(firsts)} {pick(surnames)}\n"
        elif k == 6:
            s = f'maintainer = "{w1}.{w2}@{word(g).lower()}.com"\n'
        elif k == 7:
            s = f"upstream = 10.{int(g.integers(0, 256))}.{int(g.integers(0, 256))}.{int(g.integers(1, 255))}\n"
        else:
            s = f"{w1.upper()}_{w2.upper()}={w1}\n"
        out.append(s)
        size += len(s)
    data = "".join(out).encode()[:n_bytes]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def public_docs(folder, firsts, surnames, n=240, seed=0):
    """Labelled documents in the public TRAIN format ({id, source, lang, text, labels: [[char a, char b, label]]}),
    with the label names of one public dataset: names, e-mail and phone map to types; ssn and city are outside the
    taxonomy (IGNORE)."""
    g = np.random.default_rng(seed)
    pick = lambda seq: seq[int(g.integers(0, len(seq)))]
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


def catalogs(folder, seed=0):
    """Two made-up credential formats and one example value, generated (the secrets-code catalogue files)."""
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


def write_all(data):
    """Everything the pii vertical reads from $PUREBYTE_DATA, under `data`."""
    firsts, surnames = lists(os.path.join(data, "pii", "lists"))
    code_corpus(os.path.join(data, "corpora", "smoke-context.bin"), firsts, surnames)
    public_docs(os.path.join(data, "pii", "public_train"), firsts, surnames)
    catalogs(os.path.join(data, "catalogs"))
    return firsts, surnames
