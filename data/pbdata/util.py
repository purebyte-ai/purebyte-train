"""Small helpers shared by every data script: hashing, atomic writes, canonical JSON, logging, polite processes."""
import hashlib
import json
import os
import sys
import tempfile


class DataError(Exception):
    """A build or fetch that must stop. The message says what happened and what to do next."""


def log(msg=""):
    print(msg, flush=True)


def warn(msg):
    print(f"WARNING: {msg}", file=sys.stderr, flush=True)


def human(n):
    n = float(n)
    for unit in ("B", "KiB", "MiB", "GiB"):
        if abs(n) < 1024 or unit == "GiB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} GiB"


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def order_key(*parts):
    """A deterministic pseudo-random sort key: the same on every platform and Python version (no RNG involved)."""
    h = hashlib.sha256()
    for p in parts:
        h.update(str(p).encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def atomic_write_bytes(path, data):
    ensure_dir(os.path.dirname(os.path.abspath(path)))
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", dir=os.path.dirname(os.path.abspath(path)))
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def dumps(obj):
    """Canonical JSON: sorted keys, one space indent, UTF-8, trailing newline. Same object -> same bytes."""
    return json.dumps(obj, sort_keys=True, indent=1, ensure_ascii=False) + "\n"


def write_json(path, obj):
    atomic_write_bytes(path, dumps(obj).encode("utf-8"))


def read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def polite(threads=2):
    """Idle priority and capped threads: data builds run on machines that are busy with other work.

    Uses `pbtrain.runtime.low_priority()` (typed Windows calls, nice 19 elsewhere) when the training package is
    importable; otherwise only the thread caps are set."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[var] = str(threads)
    try:
        from . import paths
        paths._training_on_path()
        from pbtrain.runtime import low_priority
        return bool(low_priority())
    except Exception:                                   # priority is a courtesy, never a reason to stop
        return False


def check_hex(value, n):
    return isinstance(value, str) and len(value) == n and all(c in "0123456789abcdef" for c in value)
