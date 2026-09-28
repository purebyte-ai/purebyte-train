"""Where a result comes from: the code commit, the digest of the trainer's source, and the software and hardware that
computed it. Recorded in every checkpoint, in `finetune/done.json` and in the release MANIFEST; the trainer's digest,
the device, the backend, the PyTorch and NumPy versions and the thread count are also part of the seed cache key
(train/seeds.py), so weights computed by other code or on another device are never reused."""
import functools
import hashlib
import os
import platform
import subprocess

from .paths import REPO, TRAINING

# The code that decides the weights: the model, the trainer, the GPU backends and the process settings (thread caps,
# GPU determinism). A change to any of these files changes `trainer_digest()`.
TRAINER_SOURCES = ("pbtrain/model", "pbtrain/train", "pbtrain/gpu", "pbtrain/runtime.py")


@functools.lru_cache(maxsize=1)
def trainer_digest():
    """sha256 of the trainer's source files (relative path + content, LF line endings, in sorted order)."""
    files = []
    for rel in TRAINER_SOURCES:
        p = os.path.join(TRAINING, *rel.split("/"))
        if os.path.isfile(p):
            files.append(p)
        for root, _, names in os.walk(p):
            files += [os.path.join(root, n) for n in names if n.endswith(".py")]
    h = hashlib.sha256()
    for p in sorted(set(files)):
        with open(p, "rb") as f:
            data = f.read().replace(b"\r\n", b"\n")
        h.update(os.path.relpath(p, TRAINING).replace(os.sep, "/").encode() + b"\0")
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


def code_commit():
    """The git commit of the repository, `+dirty` when `training/` or `specialists/` has local changes; a plain
    statement when this is not a git checkout or git is missing."""
    try:
        r = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO, timeout=30)
        if r.returncode != 0 or not r.stdout.strip():
            return "not a git checkout"
        dirty = subprocess.run(["git", "status", "--porcelain", "--", "training", "specialists"], capture_output=True,
                               text=True, cwd=REPO, timeout=60)
        return r.stdout.strip() + ("+dirty" if dirty.returncode != 0 or dirty.stdout.strip() else "")
    except (OSError, subprocess.SubprocessError) as e:
        return f"no git ({type(e).__name__})"


def cpu_capability():
    """The CPU kernel path PyTorch dispatches to (for example AVX2 or AVX512): CPU results can differ between them."""
    import torch
    fn = getattr(getattr(torch.backends, "cpu", None), "get_cpu_capability", None)
    return fn() if fn is not None else "unknown"


def environment(device="cpu", backend="cpu"):
    """The software and hardware of THIS process, for the record: device and backend, library versions, torch threads
    and CPU kernel path, the trainer's digest."""
    import numpy
    import torch
    kind = str(device).split(":")[0]
    gpu = None
    if kind == "cuda" and torch.cuda.is_available() and torch.cuda.device_count() > 0:
        gpu = torch.cuda.get_device_name(0)
    hip, cuda = getattr(torch.version, "hip", None), getattr(torch.version, "cuda", None)
    return dict(device=str(device), backend=str(backend), gpu=gpu, torch=str(torch.__version__),
                torch_hip=None if hip is None else str(hip), torch_cuda=None if cuda is None else str(cuda),
                numpy=str(numpy.__version__), python=platform.python_version(),
                os=f"{platform.system()} {platform.release()}", machine=platform.machine(),
                cpu=platform.processor() or platform.machine(), cpu_capability=cpu_capability(),
                torch_threads=torch.get_num_threads(), trainer_digest=trainer_digest())
