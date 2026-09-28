"""Made-up stand-ins for the CPU smoke recipes, and how a build recognises them.

`python data/build_smoke_data.py` writes into $PUREBYTE_DATA a generated code-like corpus (`corpora/smoke-context.bin`)
and, only where nothing is there yet, made-up credential-format catalogues (`catalogs/`), look-alike PII lists
(`pii/lists/`) and look-alike labelled documents (`pii/public_train/`). It records every file it wrote, with its
sha256, in `$PUREBYTE_DATA/smoke-standins.json`. A file counts as a stand-in while its bytes are the recorded ones: the
real build that overwrites it (data/build_secret_catalogs.py, data/fetch_pii.py lists) makes it real again. Nothing
here, or in any data script, deletes a stand-in.

The stand-ins exist so that the smoke recipes run on a fresh checkout, and they must train nothing else: the
`secrets_synthetic` and `pii_*` verticals refuse them unless the recipe's corpus is the smoke corpus, and
data/build_creddata_stream.py refuses stand-in catalogues. This module lives with the generator because every one of
those readers imports the generator's package; it reads files only and draws no random number.
"""
import hashlib
import json
import os

MARKER = "smoke-standins.json"
SMOKE_CORPUS = "corpora/smoke-context.bin"


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def in_place(data_dir):
    """{relative path: sha256} of the stand-ins still in place under `data_dir` (none without the marker file)."""
    p = os.path.join(data_dir, MARKER)
    if not os.path.exists(p):
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            files = json.load(f).get("files") or {}
    except (OSError, ValueError):
        return {}
    out = {}
    for rel, sha in sorted(files.items()):
        q = os.path.join(data_dir, *rel.split("/"))
        if os.path.isfile(q) and _sha256(q) == sha:
            out[rel] = sha
    return out


def is_smoke_corpus(data_dir, corpus_path):
    """True when `corpus_path` is `$PUREBYTE_DATA/corpora/smoke-context.bin`, the corpus of the smoke recipes."""
    smoke = os.path.join(data_dir, *SMOKE_CORPUS.split("/"))
    return corpus_path is not None and os.path.normcase(os.path.abspath(corpus_path)) == os.path.normcase(
        os.path.abspath(smoke))


def check(data_dir, prefixes, corpus_path, what, how):
    """The stand-ins under `prefixes` (folders relative to `data_dir`, ending with "/") that a build is about to use.
    Raises ValueError when there is one and the build's corpus is not the smoke corpus; returns their sorted relative
    paths otherwise (recorded in the build's manifest)."""
    used = sorted(rel for rel in in_place(data_dir) if any(rel.startswith(p) for p in prefixes))
    if used and not is_smoke_corpus(data_dir, corpus_path):
        raise ValueError(f"{what} under {data_dir} are made-up stand-ins written by data/build_smoke_data.py "
                         f"({', '.join(used[:3])}{', ...' if len(used) > 3 else ''}): they serve the CPU smoke recipes "
                         f"only (corpus {SMOKE_CORPUS}). Build the real ones first: {how}")
    return used
