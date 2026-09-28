"""`pbtrain doctor`: what this machine has, what the training stack finds, and what is missing.

The doctor's own process keeps CUDA and ROCm devices hidden (like every pbtrain command without --device). Whether a
CUDA or ROCm GPU is usable is asked in a CHILD process that sees the devices the user's environment makes visible
(`runtime.user_gpu_env`): it counts them and reads their names (no memory is allocated, no kernel runs). Apple's MPS is
asked in this process (asking does not initialize it).
"""
import glob
import json
import os
import platform
import shutil
import subprocess
import sys


def _line(ok, what, detail=""):
    print(f"  [{'ok' if ok else '--'}] {what}" + (f": {detail}" if detail else ""))
    return ok


_PROBE = r'''
import json, torch
out = dict(available=False, count=0, names=[])
try:
    out["available"] = bool(torch.cuda.is_available()) and torch.cuda.device_count() > 0
    if out["available"]:
        out["count"] = torch.cuda.device_count()
        out["names"] = [torch.cuda.get_device_name(i) for i in range(out["count"])]
except Exception as e:
    out["error"] = f"{type(e).__name__}: {e}"[:300]
print(json.dumps(out))
'''


def standins(data_dir):
    """The made-up stand-ins of data/build_smoke_data.py still in place (relative paths): the files that
    `$PUREBYTE_DATA/smoke-standins.json` records and whose bytes are still the recorded ones. They serve the smoke
    recipes only."""
    import hashlib
    try:
        with open(os.path.join(data_dir, "smoke-standins.json"), encoding="utf-8") as f:
            files = json.load(f).get("files") or {}
    except (OSError, ValueError, AttributeError):
        return set()
    out = set()
    for rel, digest in files.items():
        p = os.path.join(data_dir, *rel.split("/"))
        if os.path.isfile(p):
            with open(p, "rb") as f:
                if hashlib.sha256(f.read()).hexdigest() == digest:
                    out.add(rel)
    return out


def probe_gpus(timeout=120):
    """{available, count, names[, error]} of the CUDA / ROCm devices, asked in a child process that sees what the
    user's environment makes visible."""
    from .runtime import user_gpu_env
    env = user_gpu_env()
    try:
        r = subprocess.run([sys.executable, "-c", _PROBE], env=env, capture_output=True, text=True, timeout=timeout)
        return json.loads(r.stdout.strip().splitlines()[-1]) if r.returncode == 0 and r.stdout.strip() else \
            dict(available=False, count=0, names=[], error=(r.stderr or "no output").strip()[-300:])
    except (OSError, subprocess.SubprocessError, ValueError) as e:
        return dict(available=False, count=0, names=[], error=f"{type(e).__name__}: {e}"[:300])


def doctor():
    from . import paths, runtime
    print("pbtrain doctor")
    print(f"  python {platform.python_version()} on {platform.system()} {platform.machine()}")
    torch = None
    try:
        import numpy
        import torch
        from .provenance import cpu_capability
        _line(True, "torch / numpy", f"{torch.__version__} / {numpy.__version__}; torch threads "
                                     f"{torch.get_num_threads()}; CPU kernels {cpu_capability()}")
    except ImportError as e:
        _line(False, "torch / numpy", str(e))
    for mod in ("yaml", "sklearn"):
        try:
            __import__(mod)
            _line(True, mod)
        except ImportError:
            _line(False, mod, "pip install pyyaml scikit-learn")
    ok = runtime.low_priority()
    _line(ok, "idle priority verified" if ok else "idle priority NOT verified", "" if ok else
          "this process could not lower its own priority: jobs will compete with interactive work")
    if torch is not None:
        hip, cuda = getattr(torch.version, "hip", None), getattr(torch.version, "cuda", None)
        build = "ROCm " + hip if hip else "CUDA " + cuda if cuda else "CPU only"
        mps = getattr(torch.backends, "mps", None)
        apple = bool(mps is not None and mps.is_built())
        _line(True, "PyTorch build", build + (", MPS (Apple silicon)" if apple else ""))
        if hip or cuda:
            g = probe_gpus()
            if g["available"]:
                _line(True, "CUDA / ROCm GPU", f"{g['count']} usable ({', '.join(g['names'])}): --device cuda")
            else:
                _line(False, "CUDA / ROCm GPU", "none usable" + (f" ({g['error']})" if g.get("error") else
                                                                 " (no device, or a driver problem)"))
        else:
            _line(False, "CUDA / ROCm GPU", "this PyTorch build has neither: install the build for your GPU "
                                            "(training/docs/gpu.md)")
        if apple:
            _line(mps.is_available(), "MPS device", "available: --device mps" if mps.is_available()
                  else "built but not available (Apple silicon and macOS 12.3 or newer)")
    d, w = paths.data_dir(), paths.work_dir()
    _line(os.path.isdir(d), "PUREBYTE_DATA", d)
    _line(True, "PUREBYTE_WORK", w)
    fake = standins(d)
    smoke = " (a made-up stand-in: smoke recipes only)"
    for c in sorted(glob.glob(os.path.join(d, "corpora", "*.bin"))):
        man = os.path.exists(c + ".manifest.json") or os.path.exists(os.path.splitext(c)[0] + ".manifest.json")
        _line(True, "corpus", f"{os.path.basename(c)} {os.path.getsize(c):,} B"
              + (smoke if f"corpora/{os.path.basename(c)}" in fake else "" if man else " (NO build manifest)"))
    for n, how in (("smoke-context.bin", "data/build_smoke_data.py (the smoke recipes)"),
                   ("code-context.bin", "data/build_code_corpus.py"),
                   ("binary-context.bin", "data/build_binary_corpus.py (secrets-bin)")):
        if not os.path.exists(os.path.join(d, "corpora", n)):
            _line(False, f"corpus {n}", how)
    for sub, pattern, how in (("lists", "*.tsv", "data/fetch_pii.py lists"),
                              ("public_train", "*.jsonl", "data/fetch_pii.py train"),
                              ("public_train_xl", "*.jsonl", "data/build_pii_pool.py (the pii release pool)")):
        files = glob.glob(os.path.join(d, "pii", sub, pattern))
        n_fake = sum(1 for f in files if f"pii/{sub}/{os.path.basename(f)}" in fake)
        _line(len(files) > n_fake, f"pii/{sub}", (f"{len(files)} file(s)" + (f", {n_fake} of them made-up stand-ins "
                                                                              f"(smoke recipes only): {how}"
                                                                              if n_fake else ""))
              if files else how)
    for cat in ("gitleaks.json", "trufflehog.json"):
        p = os.path.join(d, "catalogs", cat)
        real = os.path.exists(p) and f"catalogs/{cat}" not in fake
        _line(real, f"catalog {cat}", p if real else (p + smoke + ": data/build_secret_catalogs.py" if os.path.exists(p)
                                                        else "data/build_secret_catalogs.py"))
    streams = sorted(glob.glob(os.path.join(d, "streams", "*.npz")))
    for s in streams:
        man = os.path.exists(os.path.splitext(s)[0] + ".manifest.json")
        _line(True, "real-window stream",
              f"{os.path.basename(s)} {os.path.getsize(s):,} B" + ("" if man else " (NO build manifest)"))
    if not any(os.path.basename(s) == "creddata-dev-spans.npz" for s in streams):
        _line(False, "real-window stream creddata-dev-spans.npz", "data/build_creddata_stream.py (secrets-code 1.0.0)")
    fps = sorted(glob.glob(os.path.join(paths.exams_dir(), "fingerprints", "*.npz")))
    _line(bool(fps), "exam fingerprints", ", ".join(os.path.basename(f)[:-4] for f in fps) or "data/fingerprint_exams.py")
    from .verticals import list_specialists
    try:
        _line(True, "specialists", ", ".join(list_specialists()))
    except FileNotFoundError as e:
        _line(False, "specialists", str(e))
    ref = os.path.join(paths.REFERENCE_SRC, "purebyte_ref")
    _line(os.path.isdir(ref), "NumPy reference (verify stage)", ref)
    tool = os.environ.get("PUREBYTE_MODEL_DUMP")
    _line(bool(tool and os.path.exists(tool)), "engine model_dump (fast verify)", tool or "set PUREBYTE_MODEL_DUMP")
    _line(bool(shutil.which("git")), "git (code commit in release manifests)")
    return 0
