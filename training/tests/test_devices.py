"""Devices, everywhere (CPU, with torch's device queries faked where needed): the rules of `--device` and `--backend`
(every combination), what `pbtrain train` / `run` do with them (auto says what it chose, a GPU run takes the lock and
asks for deterministic algorithms, a CPU run does neither), the GPU lock (refusal while held, release when its owner
dies, owner reported), the environment of GPU processes and workers, the cache key per device and backend, and the
Radeon backend's platform and window limits.

On a machine with a GPU, opt-in (it touches the GPU): PBTRAIN_GPU_TESTS=1 python -m pytest -q -rs
training/tests/test_devices.py. Then every device found (CUDA: NVIDIA or AMD ROCm; MPS: Apple silicon) must
- train like the CPU: `train_spans` itself (optimizer, schedule, clipping) for a few steps gives the CPU's losses, and
  `forward_windows` the CPU's log-probabilities, within the tolerances written in the test (float32 kernels in another
  summation order: close, not equal);
- train deterministically: the same job in two fresh processes gives the same bits, with the stock backend and (ROCm on
  Windows) the radeon backend, and seeds trained in parallel give the serial run's bits.
Each device runs in its own process: the test session itself hides every CUDA / ROCm device (conftest.py); MPS cannot
be hidden by an environment variable, and no test here places a tensor on it without PBTRAIN_GPU_TESTS=1.
"""
import argparse
import json
import os
import subprocess
import sys
import types

import numpy as np
import pytest

from pbtrain import runtime

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
GPU = pytest.mark.skipif(os.environ.get("PBTRAIN_GPU_TESTS") != "1", reason="touches the GPU: PBTRAIN_GPU_TESTS=1")


def _fake_torch(monkeypatch, cuda, mps):
    import torch
    monkeypatch.setattr(torch.cuda, "is_available", lambda: cuda)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: int(cuda))
    monkeypatch.setattr(torch.backends, "mps", types.SimpleNamespace(is_available=lambda: mps, is_built=lambda: mps))


def test_cpu_is_the_default_and_needs_no_gpu():
    assert runtime.resolve_device(None) == "cpu"
    assert runtime.resolve_device("cpu") == "cpu"


def test_unknown_device_is_refused():
    with pytest.raises(ValueError):
        runtime.resolve_device("tpu")


@pytest.mark.parametrize("cuda,mps,want", [(True, True, "cuda"), (True, False, "cuda"), (False, True, "mps"),
                                           (False, False, "cpu")])
def test_auto_takes_the_first_device_found(monkeypatch, cuda, mps, want):
    _fake_torch(monkeypatch, cuda, mps)
    assert runtime.resolve_device("auto") == want


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_a_missing_gpu_is_an_error_not_a_silent_cpu_run(monkeypatch, device):
    _fake_torch(monkeypatch, cuda=False, mps=False)
    with pytest.raises(RuntimeError, match="doctor"):
        runtime.resolve_device(device)


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_an_available_gpu_is_used(monkeypatch, device):
    _fake_torch(monkeypatch, cuda=device == "cuda", mps=device == "mps")
    assert runtime.resolve_device(device) == device


_PROBE = r'''
import json, torch
mps = getattr(torch.backends, "mps", None)
print(json.dumps([d for d, ok in (("cuda", torch.cuda.is_available() and torch.cuda.device_count() > 0),
                                  ("mps", bool(mps is not None and mps.is_available()))) if ok]))
'''

# The trainer itself on one device: `train_spans` (AdamW groups, WSD schedule, clipping) for a few steps, then
# `forward_windows`; weights and data are drawn on the CPU, so they are identical on every device.
_TRAIN = r'''
import hashlib, json, sys
sys.path.insert(0, sys.argv[1])
import numpy as np, torch
dev, out, backend = sys.argv[2], sys.argv[3], sys.argv[4]
torch.set_num_threads(1)
if backend == "radeon":
    from pbtrain.gpu import enable
    enable("radeon")
from pbtrain.eval.spans import forward_windows
from pbtrain.train.loop import SpanTask, train_spans
from pbtrain.train.recipe import SpanRecipe
g = np.random.default_rng(3)
X = g.integers(0, 256, (64, 512), dtype=np.uint8)
S = np.zeros((64, 512), np.uint8)
Y = np.zeros(64, np.int64)
for i in range(0, 64, 2):
    a = int(g.integers(0, 492)); S[i, a:a + 12] = 1; Y[i] = 1
losses = []
m, th, ch = train_spans(SpanTask((X, Y, S), None, None), SpanRecipe(size="nano", steps=6, batch=8, lr=1e-3, warmup=2),
                        seed=0, device=dev, log=lambda d: losses.append(d["loss"]))
lp, doc = forward_windows(m, th, ch, X[:8], device=dev)
h = hashlib.sha256()
for mod in (m, th, ch):
    for k, t in mod.state_dict().items():
        h.update(t.detach().cpu().float().contiguous().numpy().tobytes())
np.save(out + ".lp.npy", lp)
with open(out, "w") as f:
    json.dump(dict(losses=losses, sha=h.hexdigest(), deterministic=torch.are_deterministic_algorithms_enabled()), f)
'''

# `train_seeds` on one device, serially and with two worker processes: the same bits
_PARALLEL = r'''
import sys, os
sys.path.insert(0, sys.argv[1])
import numpy as np, torch
from pbtrain.train import checkpoint
from pbtrain.train.recipe import SpanRecipe
from pbtrain.train.seeds import train_seeds
dev, d = sys.argv[2], sys.argv[3]
torch.set_num_threads(1)
g = np.random.default_rng(4)
X = g.integers(0, 256, (48, 512), dtype=np.uint8); S = np.zeros((48, 512), np.uint8); S[::2, 100:110] = 1
np.savez(os.path.join(d, "p.npz"), Xtr=X, Ytr=S.any(1).astype(np.int64), Str=S)
r = SpanRecipe(size="nano", steps=4, batch=8, lr=1e-3, warmup=1)
par = train_seeds(os.path.join(d, "p.npz"), r, [0, 1], device=dev, parallel=2, cache_dir=os.path.join(d, "par"), log=print)
ser = train_seeds(os.path.join(d, "p.npz"), r, [0, 1], device=dev, parallel=1, cache_dir=os.path.join(d, "ser"), log=print)
for j in par:
    a, b = checkpoint.load(par[j]), checkpoint.load(ser[j])
    assert all(torch.equal(a["model"][k], b["model"][k]) for k in a["model"]), j
print("EQUAL")
'''


def _gpu_env():
    env = dict(os.environ)
    for var in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES"):
        env.pop(var, None)                             # the session hides the GPUs; the children must see them
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
    return env


def _found(env):
    found = json.loads(subprocess.run([sys.executable, "-c", _PROBE], env=env, check=True, capture_output=True,
                                      text=True).stdout)
    if not found:
        pytest.skip("no CUDA, ROCm or MPS device on this machine")
    return found


def _train(tmp_path, dev, env, backend="stock", tag=""):
    p = str(tmp_path / f"{dev}-{backend}{tag}.json")
    subprocess.run([sys.executable, "-c", _TRAIN, TRAINING, dev, p, backend], check=True, env=env)
    with open(p) as f:
        return json.load(f), np.load(p + ".lp.npy")


def _radeon_here(env):
    code = "import os, torch; print(int(os.name == 'nt' and bool(getattr(torch.version, 'hip', None))))"
    return subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True).stdout.strip() == "1"


# the tolerances of the parity test: float32 kernels in another summation order (no TF32), over six optimizer steps.
# Measured on the RX 6800 XT (ROCm on Windows): losses 1.2e-7 apart (relative), log-probabilities 6.2e-6.
LOSS_RTOL, LOGP_ATOL = 1e-4, 1e-4


@GPU
def test_every_gpu_trains_like_the_cpu(tmp_path):
    env = _gpu_env()
    found = _found(env)
    with runtime.gpu_lock("test_devices parity"):
        ref, ref_lp = _train(tmp_path, "cpu", env)
        got = {dev: _train(tmp_path, dev, env) for dev in found}
    assert ref["deterministic"] is False                  # the CPU run never switches PyTorch to that mode
    for dev, (res, lp) in got.items():
        assert res["deterministic"] is True, dev          # a GPU run asks for deterministic algorithms
        for step, (a, b) in enumerate(zip(ref["losses"], res["losses"])):
            assert abs(a - b) <= LOSS_RTOL * max(1.0, abs(a)), f"{dev}, step {step + 1}: loss {b} vs the CPU's {a}"
        assert np.abs(lp - ref_lp).max() <= LOGP_ATOL, f"{dev}: log-probabilities differ by {np.abs(lp - ref_lp).max()}"
        print(f"{dev}: largest loss difference {max(abs(a - b) / max(1.0, abs(a)) for a, b in zip(ref['losses'], res['losses'])):.2e} "
              f"(relative), largest log-probability difference {np.abs(lp - ref_lp).max():.2e}")


@GPU
def test_gpu_training_is_deterministic(tmp_path):
    """The same job in two fresh processes gives the same bits, with every backend this machine has."""
    env = _gpu_env()
    found = _found(env)
    jobs = [(dev, "stock") for dev in found] + ([("cuda", "radeon")] if "cuda" in found and _radeon_here(env) else [])
    with runtime.gpu_lock("test_devices determinism"):
        for dev, backend in jobs:
            a, _ = _train(tmp_path, dev, env, backend, "-a")
            b, _ = _train(tmp_path, dev, env, backend, "-b")
            assert a["sha"] == b["sha"], f"{dev} / {backend}: two runs of the same job gave different weights"


@GPU
def test_parallel_seeds_on_a_gpu_are_the_serial_ones(tmp_path):
    env = _gpu_env()
    found = _found(env)
    with runtime.gpu_lock("test_devices parallel"):
        for dev in found:
            d = tmp_path / dev
            d.mkdir()
            r = subprocess.run([sys.executable, "-c", _PARALLEL, TRAINING, dev, str(d)], env=env, capture_output=True,
                               text=True)
            assert r.returncode == 0 and "EQUAL" in r.stdout, r.stdout[-800:] + r.stderr[-800:]


# ----------------------------------------------------------------------------------- the rules, on the CPU
RECIPE = os.path.join(os.path.dirname(TRAINING), "specialists", "secrets-code", "recipes", "smoke-cpu.yaml")


def _args(**kw):
    return argparse.Namespace(**dict(dict(cmd="train", task=RECIPE, device="cpu", parallel=1, backend=None, work=None),
                                     **kw))


def test_every_device_and_backend_combination():
    from pbtrain.gpu import backend_for
    ok = {("cpu", None): "cpu", ("cpu", "cpu"): "cpu", ("cuda", None): "cuda", ("cuda", "cuda"): "cuda",
          ("cuda", "radeon"): "radeon", ("mps", None): "mps", ("mps", "mps"): "mps"}
    for device in ("cpu", "cuda", "mps"):
        for backend in (None, "cpu", "cuda", "mps", "radeon"):
            if (device, backend) in ok:
                assert backend_for(device, backend) == ok[(device, backend)]
            else:
                with pytest.raises(runtime.DeviceError):
                    backend_for(device, backend)


def test_the_seed_cache_key_holds_the_device_and_the_backend():
    from pbtrain.train.recipe import SpanRecipe
    from pbtrain.train.seeds import cache_key
    r = SpanRecipe(size="mini64", steps=2)
    keys = [cache_key("d", r, 0, False, b, d) for d, b in (("cpu", "cpu"), ("cuda", "cuda"), ("cuda", "radeon"),
                                                           ("mps", "mps"), ("cuda:0", "cuda"))]
    assert len(set(keys)) == 4 and keys[1] == keys[4]           # cuda and cuda:0 are one device kind


def test_a_gpu_run_says_what_it_chose_takes_the_lock_and_asks_for_determinism(monkeypatch, capsys):
    import contextlib
    import pbtrain.cli as cli
    import pbtrain.factory.stages as stages
    calls = {}

    @contextlib.contextmanager
    def lock(what, path=None):
        calls["lock"] = what
        yield "gpu.lock"
    monkeypatch.setattr(runtime, "resolve_device", lambda d: "cuda")
    monkeypatch.setattr(runtime, "gpu_lock", lock)
    monkeypatch.setattr(runtime, "gpu_determinism", lambda: calls.setdefault("deterministic", True))
    monkeypatch.setattr(stages, "run", lambda spec, st, work, ctx: calls.setdefault("ctx", ctx))
    cli._run(_args(device="auto", parallel=3), ["baselines", "finetune"])
    assert "--device auto chose cuda" in capsys.readouterr().err
    assert calls["lock"].startswith("pbtrain train") and calls["deterministic"]
    assert (calls["ctx"].device, calls["ctx"].backend, calls["ctx"].parallel) == ("cuda", "cuda", 3)


def test_a_cpu_run_takes_no_lock_and_says_what_auto_chose(monkeypatch, capsys):
    import pbtrain.cli as cli
    import pbtrain.factory.stages as stages
    calls = {}
    monkeypatch.setattr(runtime, "resolve_device", lambda d: "cpu")
    monkeypatch.setattr(runtime, "gpu_lock", lambda *a, **k: calls.setdefault("lock", True))
    monkeypatch.setattr(runtime, "gpu_determinism", lambda: calls.setdefault("deterministic", True))
    monkeypatch.setattr(stages, "run", lambda spec, st, work, ctx: calls.setdefault("ctx", ctx))
    cli._run(_args(device="auto"), ["baselines", "finetune"])
    assert "--device auto chose cpu" in capsys.readouterr().err
    assert "lock" not in calls and "deterministic" not in calls and calls["ctx"].backend == "cpu"


def test_the_radeon_backend_refuses_long_windows_and_other_platforms(monkeypatch):
    import torch
    import pbtrain.cli as cli
    from pbtrain.gpu.radeon import fast_ssd, hipk
    with pytest.raises(SystemExit, match="512-byte windows"):
        cli._ctx(_args(device="cuda", backend="radeon"), {"data": {"seq_len": 1024}})
    monkeypatch.setattr(hipk, "_state", {"dll": None, "lib": None})
    monkeypatch.setattr(torch.version, "hip", None)
    with pytest.raises(runtime.DeviceError, match="PyTorch ROCm for Windows"):
        hipk.setup()
    fake = lambda T: types.SimpleNamespace(is_cuda=True, dtype=torch.float32, shape=(2, T, 768),
                                           numel=lambda: 2 * T * 768)
    dt = types.SimpleNamespace(dtype=torch.float32)
    assert fast_ssd.supported(fake(512), dt, None, None, 64) and not fast_ssd.supported(fake(1024), dt, None, None, 64)


def test_a_worker_sets_up_its_device_and_backend_like_the_parent(tmp_path, monkeypatch):
    import pbtrain.gpu as gpu
    import pbtrain.train.seeds as seeds
    from pbtrain.train.recipe import SpanRecipe
    got = {}
    monkeypatch.setattr(runtime, "setup", lambda threads, gpu, **k: got.update(threads=threads, gpu=gpu))
    monkeypatch.setattr(gpu, "enable", lambda name: got.update(enabled=name))
    monkeypatch.setattr(seeds, "_train_one", lambda *a: got.update(args=a))
    monkeypatch.setenv("PBTRAIN_THREADS", "3")
    job = tmp_path / "job.json"
    job.write_text(json.dumps(dict(npz="x.npz", recipe=SpanRecipe(size="mini64").as_dict(), seed=1, shuffled=False,
                                   device="cuda", out="o.pt", backend="radeon")), encoding="utf-8")
    seeds._worker(str(job))
    assert (got["gpu"], got["threads"], got["enabled"]) == (True, 3, "radeon")
    assert (got["args"][4], got["args"][6]) == ("cuda", "radeon")


_ENV = r'''
import json, os, sys
sys.path.insert(0, sys.argv[1])
from pbtrain import runtime
runtime.setup(threads=1, lowprio=False, gpu=sys.argv[2] == "gpu")
import torch
from pbtrain.train.loop import SpanTask, train_spans
from pbtrain.train.recipe import SpanRecipe
import numpy as np
X = np.zeros((8, 64), np.uint8); S = np.zeros((8, 64), np.uint8); S[::2, 3:9] = 1
train_spans(SpanTask((X, S.any(1).astype(np.int64), S), None, None), SpanRecipe(size="mini64", steps=1, batch=4))
out = dict(cuda=os.environ.get("CUDA_VISIBLE_DEVICES"), cublas=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
           mps_fallback=os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
           deterministic_after_cpu_training=torch.are_deterministic_algorithms_enabled())
runtime.gpu_determinism()
out.update(deterministic=torch.are_deterministic_algorithms_enabled(), cudnn=[torch.backends.cudnn.deterministic,
                                                                             torch.backends.cudnn.benchmark,
                                                                             torch.backends.cudnn.allow_tf32])
print(json.dumps(out))
'''


@pytest.mark.parametrize("kind", ["cpu", "gpu"])
def test_the_process_environment_of_cpu_and_gpu_jobs(kind):
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="7", HIP_VISIBLE_DEVICES="7")   # no device 7: nothing is touched
    env.pop("CUBLAS_WORKSPACE_CONFIG", None)
    env.pop("PYTORCH_ENABLE_MPS_FALLBACK", None)
    r = subprocess.run([sys.executable, "-c", _ENV, TRAINING, kind], env=env, capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-800:]
    out = json.loads(r.stdout.strip().splitlines()[-1])
    assert out["deterministic_after_cpu_training"] is False       # a CPU run never switches that mode on
    assert out["deterministic"] is True and out["cudnn"] == [True, False, False]
    if kind == "gpu":
        assert (out["cuda"], out["cublas"], out["mps_fallback"]) == ("7", ":4096:8", "1")
    else:
        assert (out["cuda"], out["cublas"], out["mps_fallback"]) == ("", None, None)


def test_synchronize_is_harmless_without_a_gpu():
    runtime.synchronize("cpu")
    runtime.synchronize("cuda")                              # hidden in the test session: a no-op, not an error


def test_a_hidden_gpu_is_not_a_usable_one():
    """ROCm builds can say `is_available()` with every device hidden; pbtrain also asks for a device count."""
    import torch
    assert torch.cuda.device_count() == 0                    # the session hides the CUDA / ROCm devices
    assert runtime.has_cuda(torch) is False
    with pytest.raises(runtime.DeviceError, match="no CUDA or ROCm GPU is visible"):
        runtime.resolve_device("cuda")


_HOLDER = r'''
import sys, time
sys.path.insert(0, sys.argv[1])
from pbtrain.runtime import gpu_lock
with gpu_lock("holder job", path=sys.argv[2]):
    print("held", flush=True)
    time.sleep(float(sys.argv[3]))
'''


def _take(path, what, tries=50):
    """gpu_lock, retried for a few seconds (the OS may take a moment to release a killed owner's lock)."""
    import time
    for _ in range(tries):
        try:
            with runtime.gpu_lock(what, path=path):
                with open(path, encoding="utf-8") as f:
                    return json.load(f)
        except runtime.GpuBusyError:
            time.sleep(0.1)
    raise AssertionError(f"{path} was never released")


def test_the_gpu_lock_refuses_a_second_job_and_is_freed_when_its_owner_dies(tmp_path):
    lock = str(tmp_path / "gpu.lock")
    p = subprocess.Popen([sys.executable, "-c", _HOLDER, TRAINING, lock, "120"], stdout=subprocess.PIPE, text=True)
    try:
        assert p.stdout.readline().strip() == "held"
        with pytest.raises(runtime.GpuBusyError, match=r"taken by pid \d+ \(holder job"):   # (a venv launcher's
            # child holds it on Windows: its pid is not p.pid)
            with runtime.gpu_lock("second job", path=lock):
                pass
    finally:
        p.kill()
        p.wait()
        p.stdout.close()
    assert _take(lock, "third job")["what"] == "third job"     # killed, not stale: the kernel released it
    assert _take(lock, "fourth job", tries=1)["what"] == "fourth job"


def test_the_gpu_lock_holds_within_one_process_and_ignores_leftover_content(tmp_path):
    lock = tmp_path / "gpu.lock"
    lock.write_text('{"pid": 4', encoding="utf-8")            # what a crash in the middle of a write could leave
    with runtime.gpu_lock("outer", path=str(lock)):
        with pytest.raises(runtime.GpuBusyError):
            with runtime.gpu_lock("inner", path=str(lock)):
                pass
    with runtime.gpu_lock("again", path=str(lock)):
        pass


def test_liveness_counts_other_users_processes(monkeypatch):
    assert runtime._alive(os.getpid()) and not runtime._alive(-1)
    if os.name != "nt":
        def denied(pid, sig):
            raise PermissionError
        monkeypatch.setattr(os, "kill", denied)
        assert runtime._alive(12345)
