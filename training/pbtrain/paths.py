"""Where things live. The ONLY module that knows the on-disk layout; nothing else hard-codes a directory.

Code and recipes are in git; data and artefacts never are:

    <repo>/training/            this package
    <repo>/specialists/<name>/  the recipes, vertical plugins and evaluation configs of each specialist
    <repo>/data/                download / build scripts and manifests (URLs, commits, sha256, licenses)
    $PUREBYTE_DATA              everything downloaded or built from those manifests (default ~/.purebyte/data):
        corpora/                context corpora the generators draw real bytes from (`data/build_code_corpus.py`,
                                `data/build_binary_corpus.py`), each with its `<name>.manifest.json`
        exams/                  evaluation sets and their fingerprints (`data/fetch_exams.py`, `fingerprint_exams.py`)
        catalogs/               credential-format catalogs (`data/build_secret_catalogs.py`)
        pii/                    PII lists and public TRAIN documents (`data/fetch_pii.py`)
        streams/                real-window sets (.npz) mined or labeled for training
    $PUREBYTE_WORK              what training produces (default ~/.purebyte/work):
        runs/<name>/<version>/<stage>/done.json      factory work dirs
        specialists/<name>/<version>/                 immutable released artefacts (never overwritten)
        releases/<name>-<semver>/                     public bundles (GGUF + card + sha256), staged for upload
        locks/gpu.lock                                the single-GPU-job lock

Paths declared in a task are expanded (`$PUREBYTE_DATA/...`); relative ones are read against $PUREBYTE_DATA.

pbtrain runs from a checkout of the repository (`pip install -e training`): the specialists, the NumPy reference and
the data scripts live next to `training/`. An installed copy outside the checkout (a plain `pip install ./training`)
has none of them; `require_source_tree` then says so instead of failing somewhere deep.
"""
import os
import re

TRAINING = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO = os.path.dirname(TRAINING)
SPECIALISTS_SRC = os.path.join(REPO, "specialists")
REFERENCE_SRC = os.path.join(REPO, "reference")
DATA_SCRIPTS = os.path.join(REPO, "data")
_HOME = os.path.join(os.path.expanduser("~"), ".purebyte")


def require_source_tree(path=SPECIALISTS_SRC):
    """`path` (a folder of the repository checkout); a clear error when pbtrain is not running from a checkout."""
    if not os.path.isdir(path):
        raise FileNotFoundError(
            f"{path} does not exist: pbtrain runs from a checkout of purebyte-train (git clone, then `pip install -e "
            f"training`); an installed copy outside the repository has no specialists/, reference/ or data/ folder")
    return path


def data_dir():
    return os.environ.get("PUREBYTE_DATA", os.path.join(_HOME, "data"))


def work_dir():
    return os.environ.get("PUREBYTE_WORK", os.path.join(_HOME, "work"))


def runs_dir():
    return os.path.join(work_dir(), "runs")


def specialists_dir():
    return os.path.join(work_dir(), "specialists")


def releases_dir():
    return os.path.join(work_dir(), "releases")


def locks_dir():
    return os.path.join(work_dir(), "locks")


def exams_dir():
    return os.path.join(data_dir(), "exams")


def expand(path, must_exist=True):
    """A path declared in a task: `$VARS` expanded (an undefined one raises, naming it); relative -> $PUREBYTE_DATA."""
    raw = str(path)
    if "$" in raw and "PUREBYTE_DATA" in raw and "PUREBYTE_DATA" not in os.environ:
        raw = raw.replace("${PUREBYTE_DATA}", data_dir()).replace("$PUREBYTE_DATA", data_dir())
    expanded = os.path.expandvars(raw)
    if "$" in expanded:
        missing = re.findall(r"\$\{?([A-Z_][A-Z0-9_]*)\}?", expanded)
        raise FileNotFoundError(f"`{path}` needs the environment variable(s) {missing}, which are not defined")
    if not os.path.isabs(expanded):
        expanded = os.path.join(data_dir(), expanded)
    if must_exist and not os.path.exists(expanded):
        raise FileNotFoundError(f"`{path}` does not exist (looked for `{expanded}`). Build it with the scripts in "
                                f"data/ (see docs/training.md), or set PUREBYTE_DATA (now {data_dir()}).")
    return expanded
