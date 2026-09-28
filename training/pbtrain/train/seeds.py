"""Training several seeds (and the control) of one recipe: in-process, or in parallel worker processes.

A worker trains exactly what the same seed would train alone (every run is seeded by `train_spans` itself, and a worker
computes with the parent's thread count), so a seed trained in parallel is bit-identical to the same seed trained
serially. Results go to a CONTENT-ADDRESSED cache: the key is the sha256 of the training arrays + every trainer
argument, with every file argument (base, real windows, n-gram memory, guard windows) hashed by CONTENT, plus what
decides the bits besides the arguments: the device, the GPU backend, the trainer's source (`provenance.trainer_digest`),
the PyTorch and NumPy versions and the torch thread count. A changed task, changed data, a re-mined file at the same
path, other trainer code or another device can never pick up stale weights. Every checkpoint records its `provenance`
(device, backend, software, when it was trained).

    runs = train_seeds(npz_path, recipe, seeds=[0, 1, 2], control_seed=0, device="cuda", parallel=3)
    runs[(seed, shuffled)] -> checkpoint path

Parallel runs on one GPU multiply VRAM: size `parallel` from the budget table in training/docs/gpu.md.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

import numpy as np

from .. import provenance
from ..paths import TRAINING
from . import checkpoint
from .recipe import SpanRecipe

_FILE_ARGS = ("base", "real_windows", "ngram_init", "guard_windows")
_SHA_CACHE = {}


class WorkerError(RuntimeError):
    """A parallel training worker failed (its log file says why)."""


def _sha_file(path):
    st = os.stat(path)
    key = (os.path.abspath(path), st.st_size, st.st_mtime_ns)
    if key not in _SHA_CACHE:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for blk in iter(lambda: f.read(1 << 24), b""):
                h.update(blk)
        _SHA_CACHE[key] = h.hexdigest()
    return _SHA_CACHE[key]


def data_sha(npz_path):
    with np.load(npz_path) as z:
        h = hashlib.sha256()
        for k in ("Xtr", "Ytr", "Str"):
            h.update(np.ascontiguousarray(z[k]).tobytes())
    return h.hexdigest()


def cache_key(data_hash, recipe: SpanRecipe, seed, shuffled, backend="cpu", device="cpu"):
    import torch
    args = recipe.replace(shuffle_spans=bool(shuffled)).as_dict()
    for k in _FILE_ARGS:
        if args.get(k):
            args[k] = _sha_file(args[k])
    env = dict(device=str(device).split(":")[0], backend=backend, trainer=provenance.trainer_digest(),
               torch=str(torch.__version__), numpy=str(np.__version__), threads=torch.get_num_threads())
    payload = json.dumps(dict(args=args, seed=int(seed), **env), sort_keys=True, default=str)
    return hashlib.sha256((data_hash + payload).encode()).hexdigest()[:24]


def _train_one(npz_path, recipe, seed, shuffled, device, out, backend="cpu"):
    from .loop import SpanTask, train_spans
    with np.load(npz_path) as z:
        task = SpanTask.from_npz(z)
    t0 = time.time()
    m, th, ch = train_spans(task, recipe.replace(shuffle_spans=bool(shuffled)), seed=seed, device=device)
    extra = {"guard": m.guard_log} if getattr(m, "guard_log", None) else {}
    prov = dict(provenance.environment(device, backend), trained_at=time.strftime("%Y-%m-%dT%H:%M:%S"))
    checkpoint.save(out, m, th, ch, seed=seed, init_seed=recipe.init_seed, trained_seconds=round(time.time() - t0, 1),
                    provenance=prov, **extra)
    return out


def train_seeds(npz_path, recipe, seeds, control_seed=None, device="cpu", parallel=1, cache_dir=None,
                backend="cpu", log=print):
    """Trains every (seed, shuffled) job not already in the cache. Returns {(seed, shuffled): checkpoint path}."""
    if len(set(int(s) for s in seeds)) != len(seeds):
        raise ValueError(f"seeds {list(seeds)} repeat a seed: every seed is one run")
    cache_dir = cache_dir or os.path.join(os.path.dirname(os.path.abspath(npz_path)), "_seed_cache")
    os.makedirs(cache_dir, exist_ok=True)
    dh = data_sha(npz_path)
    jobs = [(int(s), False) for s in seeds] + ([(int(control_seed), True)] if control_seed is not None else [])
    paths = {j: os.path.join(cache_dir, cache_key(dh, recipe, j[0], j[1], backend, device) + ".pt") for j in jobs}
    todo = [j for j in jobs if not os.path.exists(paths[j])]
    if len(todo) < len(jobs):
        log(f"    {len(jobs) - len(todo)} run(s) reused from the cache (same data, arguments, device, backend, "
            f"trainer code and library versions)")
    if parallel <= 1 or len(todo) <= 1:
        for s, c in todo:
            log(f"    training seed {s}{' (control)' if c else ''}")
            _train_one(npz_path, recipe, s, c, device, paths[(s, c)], backend)
        return paths
    import torch
    threads = torch.get_num_threads()
    log(f"    training {len(todo)} run(s), {parallel} at a time, {threads} thread(s) each")
    running, queue = [], list(todo)
    while queue or running:
        while queue and len(running) < parallel:
            s, c = queue.pop(0)
            job = dict(npz=npz_path, recipe=recipe.as_dict(), seed=s, shuffled=c, device=device, out=paths[(s, c)],
                       backend=backend)
            jf = paths[(s, c)] + ".job.json"
            with open(jf, "w", encoding="utf-8") as f:
                json.dump(job, f)
            lf = open(paths[(s, c)] + ".log", "w", encoding="utf-8")
            pp = os.pathsep.join([TRAINING] + [x for x in os.environ.get("PYTHONPATH", "").split(os.pathsep) if x])
            # a worker computes with the PARENT's thread count: CPU reductions depend on it, and a seed trained in
            # parallel must be bit-identical to the same seed trained here
            env = dict(os.environ, PYTHONUTF8="1", PYTHONPATH=pp, PBTRAIN_THREADS=str(threads))
            p = subprocess.Popen([sys.executable, "-u", "-m", "pbtrain.train.seeds", jf], stdout=lf,
                                 stderr=subprocess.STDOUT, env=env)
            running.append((p, lf, (s, c)))
        time.sleep(1.0)
        for r in list(running):
            if r[0].poll() is not None:
                running.remove(r)
                r[1].close()
                if r[0].returncode != 0:
                    for o in running:
                        o[0].kill()
                        o[0].wait()
                        o[1].close()
                    raise WorkerError(f"the worker of seed {r[2][0]}{' (control)' if r[2][1] else ''} failed: see "
                                      f"{r[1].name}")
    return paths


def _worker(job_file):
    with open(job_file, encoding="utf-8") as f:
        job = json.load(f)
    from .. import runtime
    runtime.setup(threads=int(os.environ.get("PBTRAIN_THREADS", "2")), gpu=job["device"] != "cpu")
    if job.get("backend") == "radeon":
        from ..gpu import enable
        enable("radeon")
    _train_one(job["npz"], SpanRecipe(**job["recipe"]), job["seed"], job["shuffled"], job["device"], job["out"],
               job.get("backend", "cpu"))


if __name__ == "__main__":
    _worker(sys.argv[1])
