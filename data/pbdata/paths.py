"""Where the data scripts read and write.

The root of the data directory is owned by the training package (`pbtrain.paths.data_dir()`: `$PUREBYTE_DATA`, default
`~/.purebyte/data`), so that the scripts that build data and the code that reads it can never disagree. This module
only adds the sub-directories the data scripts write:

    $PUREBYTE_DATA/
        cache/                   downloads and git objects, content-addressed (safe to delete; re-fetched on demand)
        corpora/                 code-context.bin, binary-context.bin and their manifests
        exams/<set>/             evaluation sets (never used for training)
        exams/fingerprints/      <set>.npz: the content fingerprints of every evaluation set
        pii/                     lists and public TRAIN documents of the PII specialist
        catalogs/                credential-format catalogs derived from gitleaks and trufflehog

It is also the one bridge to `training/`: the fingerprint algorithm is imported from `pbtrain.data.fingerprint`, never
re-implemented here.
"""
import os
import sys

DATA = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))      # <repo>/data
REPO = os.path.dirname(DATA)
TRAINING = os.path.join(REPO, "training")
SOURCE_MANIFESTS = os.path.join(DATA, "sources")
EXAM_MANIFESTS = os.path.join(DATA, "exams")
CREDDATA_SCORER = os.path.join(REPO, "benchmarks", "creddata", "score.py")


def _training_on_path():
    if not os.path.isdir(os.path.join(TRAINING, "pbtrain")):
        raise ImportError(f"the training package is missing: expected {TRAINING}/pbtrain (the data scripts import the "
                          f"fingerprint algorithm and the data-directory layout from it)")
    if TRAINING not in sys.path:
        sys.path.insert(0, TRAINING)


def fingerprint_module():
    """`pbtrain.data.fingerprint`: fingerprints(bytes), window_hits(X, fp), save(path, fp, meta), load(path)."""
    _training_on_path()
    from pbtrain.data import fingerprint
    return fingerprint


def pii_public_map():
    """`piigen.public_map` of the pii specialist (specialists/pii/piigen): how the labels of public PII datasets map to
    its types. The PII pool's overlap check reads the pii-bench sets through it, as the evaluation does."""
    d = os.path.join(REPO, "specialists", "pii")
    if not os.path.isdir(os.path.join(d, "piigen")):
        raise ImportError(f"the pii specialist is missing: expected {d}/piigen")
    if d not in sys.path:
        sys.path.insert(0, d)
    from piigen import public_map
    return public_map


def data_dir():
    _training_on_path()
    from pbtrain.paths import data_dir as root
    return os.path.abspath(root())


def cache_dir():
    return os.path.join(data_dir(), "cache")


def corpora_dir():
    return os.path.join(data_dir(), "corpora")


def exams_dir():
    _training_on_path()
    from pbtrain.paths import exams_dir as root
    return os.path.abspath(root())


def fingerprints_dir():
    return os.path.join(exams_dir(), "fingerprints")


def fingerprint_path(set_name):
    return os.path.join(fingerprints_dir(), f"{set_name}.npz")


def exam_dir(set_name):
    return os.path.join(exams_dir(), set_name)


def pii_dir():
    return os.path.join(data_dir(), "pii")


def catalogs_dir():
    return os.path.join(data_dir(), "catalogs")


def rel(path):
    """A path under $PUREBYTE_DATA written relative to it (what manifests record: no machine-specific prefix)."""
    root = data_dir()
    path = os.path.abspath(path)
    try:
        r = os.path.relpath(path, root)
    except ValueError:                      # another drive on Windows
        return path.replace("\\", "/")
    return ("$PUREBYTE_DATA/" + r).replace("\\", "/") if not r.startswith("..") else path.replace("\\", "/")
