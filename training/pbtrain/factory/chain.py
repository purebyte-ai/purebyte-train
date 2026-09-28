"""The done.json chain: each stage writes `<work>/<name>/<version>/<stage>/done.json` with what it read and produced.

A chain is valid only as a whole:
- every done.json records the digest of the task it ran on (`task_digest`: the sha256 of the task's parsed content,
  `description` excepted) and the sha256 of the previous stage's done.json as it was when the stage ran
  (`upstream_sha256`);
- `begin(stage)` runs before a stage does anything: every earlier done.json must exist, carry the current task's digest
  and still hash to what its successor recorded; then the done.json of this stage and of every later one is DELETED.
So a stage that fails, a stage re-run with other inputs or a task edited after `data` can never leave a stale "done"
that a later stage or a release would trust. An artefact whose chain is incomplete or broken cannot be released."""
import contextlib
import datetime
import hashlib
import json
import os
import time

import numpy as np

ORDER = ["data", "baselines", "finetune", "eval", "calibrate", "export", "verify", "report", "release"]


class StageError(RuntimeError, ValueError):
    pass


def stage_dir(work, spec, stage):
    d = os.path.join(work, spec["name"], str(spec["version"]), stage)
    os.makedirs(d, exist_ok=True)
    return d


def done_path(work, spec, stage):
    return os.path.join(work, spec["name"], str(spec["version"]), stage, "done.json")


def _load(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


def verify_chain(work, spec, stage):
    """Checks every link from `data` to the stage BEFORE `stage`. Returns the sha256 of that previous done.json (None
    for `data`). Raises StageError naming the first broken link."""
    i = ORDER.index(stage)
    prev = None
    for j, s in enumerate(ORDER[:i]):
        p = done_path(work, spec, s)
        if not os.path.exists(p):
            raise StageError(f"stage `{s}` is missing: the stages run in order ({' -> '.join(ORDER)})")
        d = _load(p)
        if d.get("task_digest") != spec["_digest"]:
            raise StageError(f"stage `{s}` ran on another version of the task (task digest "
                             f"{str(d.get('task_digest'))[:16]}, now {spec['_digest'][:16]}): a task that changed is "
                             f"run again from `data`, or gets a new version")
        if j and d.get("upstream_sha256") != prev:
            raise StageError(f"stage `{s}` was made from another `{ORDER[j - 1]}` result than the one on disk (it was "
                             f"re-run or edited afterwards): run the stages again from `{s}`")
        prev = sha_file(p)
    return prev


def require_previous(work, spec, stage):
    """The chain up to `stage` holds (see `verify_chain`)."""
    return verify_chain(work, spec, stage)


def begin(work, spec, stage):
    """Before `stage` runs: the chain up to it must hold; its own done.json and every later one are deleted."""
    verify_chain(work, spec, stage)
    for s in ORDER[ORDER.index(stage):]:
        with contextlib.suppress(FileNotFoundError):
            os.remove(done_path(work, spec, s))


def mark_done(work, spec, stage, data):
    """Writes the stage's done.json (atomically), linked to the previous stage's as it is now."""
    upstream = verify_chain(work, spec, stage)
    stage_dir(work, spec, stage)
    data = dict(data, stage=stage, ts=time.strftime("%Y-%m-%dT%H:%M:%S"), task_sha256=spec["_sha256"],
                task_digest=spec["_digest"], upstream_sha256=upstream)
    p = done_path(work, spec, stage)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=_json_default)
    os.replace(p + ".tmp", p)
    return data


def read_done(work, spec, stage):
    p = done_path(work, spec, stage)
    if not os.path.exists(p):
        raise StageError(f"stage `{stage}` has not been run")
    return _load(p)


def hash_array(a):
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()[:16]


def sha_file(path, n=None):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for blk in iter(lambda: f.read(1 << 24), b""):
            h.update(blk)
    return h.hexdigest() if n is None else h.hexdigest()[:n]


def _json_default(o):
    if isinstance(o, (datetime.date, datetime.datetime)):
        return o.isoformat()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"not JSON serializable: {type(o)}")
