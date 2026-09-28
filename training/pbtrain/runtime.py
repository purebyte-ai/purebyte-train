"""Process hygiene for a shared machine: low priority, capped threads, no CUDA/ROCm GPU unless asked, one GPU job at a
time, deterministic algorithms on a GPU.

Rules that come from incidents (training/docs/pitfalls.md):
- ONE GPU job at a time. Overflowing VRAM under ROCm on Windows does not fail cleanly: it spills into shared memory
  and can hang the whole machine. The GPU is taken through `gpu_lock()`, an operating-system lock that the kernel
  releases when its process ends, however it ends.
- CPU work runs at idle priority with few threads (the box is shared with GPU queues and other agents).
- CPU reductions depend on the thread count: bit-identity checks pin it (the tests use 1).
- On a GPU, `gpu_determinism()` asks PyTorch for deterministic algorithms; the CPU is never switched to that mode.

Call `setup(...)` BEFORE importing torch when you need the environment variables to take effect (the CLI does).
"""
import contextlib
import json
import os
import sys
import time

IDLE_PRIORITY_CLASS = 0x40
_LOCK_BYTE = 1 << 20          # Windows locks this byte range (past the owner's JSON, so the JSON stays readable)


class DeviceError(RuntimeError, ValueError):
    """A device was asked for and is not usable here (never a silent CPU run), or a device and backend do not go
    together."""


class GpuBusyError(RuntimeError):
    """Another live process holds the GPU lock."""


def _k32():
    """kernel32 with TYPED signatures. Untyped ctypes truncates the HANDLE of GetCurrentProcess() to a 32-bit int and
    SetPriorityClass then fails silently: that is how an "idle" job once ran at normal priority (training/docs/pitfalls.md)."""
    import ctypes
    from ctypes import wintypes
    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.SetPriorityClass.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    k32.SetPriorityClass.restype = wintypes.BOOL
    k32.GetPriorityClass.argtypes = [wintypes.HANDLE]
    k32.GetPriorityClass.restype = wintypes.DWORD
    k32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    k32.OpenProcess.restype = wintypes.HANDLE
    k32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    k32.GetExitCodeProcess.restype = wintypes.BOOL
    k32.CloseHandle.argtypes = [wintypes.HANDLE]
    k32.CloseHandle.restype = wintypes.BOOL
    return k32


def low_priority():
    """Idle priority on Windows, nice 19 elsewhere. Returns True when the priority is verified, False otherwise."""
    try:
        if os.name == "nt":
            k32 = _k32()
            me = k32.GetCurrentProcess()
            k32.SetPriorityClass(me, IDLE_PRIORITY_CLASS)
            return k32.GetPriorityClass(me) == IDLE_PRIORITY_CLASS
        os.nice(19 - os.nice(0))
        return os.nice(0) == 19
    except Exception:
        return False


_GPU_VARS = ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES")
_BEFORE_HIDING = {}


def hide_gpus():
    """No CUDA or ROCm device is visible to this process. Must run before torch initializes a device. Apple's MPS has
    no such variable: it stays visible, and pbtrain places nothing on it unless `--device mps` (or `auto`) says so."""
    for var in _GPU_VARS:
        _BEFORE_HIDING.setdefault(var, os.environ.get(var))
        os.environ[var] = ""


def user_gpu_env():
    """This process's environment with the GPU visibility the user gave it (what `hide_gpus` changed is undone): for a
    child process that must see the devices the user can see, and only those."""
    env = dict(os.environ)
    for var, value in _BEFORE_HIDING.items():
        if value is None:
            env.pop(var, None)
        else:
            env[var] = value
    return env


def setup(threads=2, lowprio=True, gpu=False):
    """Environment for a pbtrain process: thread caps (env + torch), priority, GPU visibility. With `gpu`, the
    variables that must precede the first GPU call are set: a fixed cuBLAS workspace (deterministic GEMMs on CUDA) and,
    on Apple GPUs, the CPU fallback for operations Metal lacks."""
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[var] = str(threads)
    os.environ.setdefault("PYTHONUTF8", "1")
    if not gpu:
        hide_gpus()
    else:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        # Apple GPUs: an operation that Metal (MPS) does not implement runs on the CPU instead of stopping the run
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if lowprio:
        low_priority()
    if "torch" in sys.modules:
        import torch
        torch.set_num_threads(int(threads))


def gpu_determinism():
    """What pbtrain asks of PyTorch whenever it trains or evaluates on a GPU: deterministic algorithms (an operation
    without one is a warning, `warn_only`), cuDNN / MIOpen without autotuning, no TF32 (float32 stays float32), and a
    fixed cuBLAS workspace (effective when set before the first GEMM: `setup(gpu=True)` sets it at process start).
    Never called for a CPU run: the CPU reference and its golden fingerprints are untouched."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    import torch
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False             # float32 means float32 (NVIDIA TF32 rounds convolutions)
    torch.backends.cuda.matmul.allow_tf32 = False


def _alive(pid):
    """Whether process `pid` is running (a process of another user counts as running). `gpu_lock` does not need it:
    the operating system releases its lock when the owner ends."""
    if pid <= 0:
        return False
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        k32 = _k32()
        h = k32.OpenProcess(0x1000, False, int(pid))                          # PROCESS_QUERY_LIMITED_INFORMATION
        if not h:
            return ctypes.get_last_error() == 5                               # ERROR_ACCESS_DENIED: it exists
        code = wintypes.DWORD()
        k32.GetExitCodeProcess(h, ctypes.byref(code))
        k32.CloseHandle(h)
        return code.value == 259                                              # STILL_ACTIVE
    try:
        os.kill(pid, 0)
        return True
    except PermissionError:                                                   # it exists and belongs to someone else
        return True
    except OSError:
        return False


def _try_lock(fd):
    """Takes the OS lock on `fd` without waiting; False when another process holds it."""
    try:
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, _LOCK_BYTE, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _unlock(fd):
    with contextlib.suppress(OSError):
        if os.name == "nt":
            import msvcrt
            os.lseek(fd, _LOCK_BYTE, os.SEEK_SET)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)


def _owner(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.loads(f.read() or "{}")
    except (OSError, ValueError):
        return {}


@contextlib.contextmanager
def gpu_lock(what="training", path=None):
    """The single-GPU-job rule as code: an exclusive operating-system lock (fcntl.flock on Linux and macOS,
    msvcrt.locking on Windows) on `$PUREBYTE_WORK/locks/gpu.lock`, held while the block runs. The kernel releases it when
    the owner ends, even when it is killed, so there are no stale locks to judge; the file holds the owner's PID, job and
    start time, for the message a refused job prints. A second job, in this process or another one, is refused
    (GpuBusyError) while the lock is held."""
    from .paths import locks_dir
    path = path or os.path.join(locks_dir(), "gpu.lock")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        if not _try_lock(fd):
            owner = _owner(path)
            raise GpuBusyError(f"the GPU is taken by pid {owner.get('pid', '?')} ({owner.get('what', 'unknown job')}, "
                               f"since {owner.get('since', '?')}): one GPU job at a time (training/docs/pitfalls.md)")
        try:
            me = json.dumps(dict(pid=os.getpid(), what=what, since=time.strftime("%Y-%m-%dT%H:%M:%S")))
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, me.encode("utf-8"))
            yield path
        finally:
            with contextlib.suppress(OSError):
                os.ftruncate(fd, 0)
            _unlock(fd)
    finally:
        os.close(fd)


DEVICES = ("cpu", "cuda", "mps", "auto")


def _has_mps(torch):
    mps = getattr(torch.backends, "mps", None)
    return bool(mps is not None and mps.is_available())


def has_cuda(torch):
    """A usable CUDA / ROCm device: `is_available()` AND at least one device (ROCm builds can report available with
    every device hidden)."""
    return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)


def resolve_device(device):
    """The device a job runs on. "cpu" is the reference and the default; "cuda" is an NVIDIA (CUDA) or AMD (ROCm) GPU;
    "mps" is the GPU of Apple silicon (Metal); "auto" takes the first of cuda, mps and cpu that this machine has.
    A GPU is only used when asked for, and asking for one that is not there is an error (DeviceError), never a silent
    CPU run."""
    if device in (None, "cpu"):
        return "cpu"
    if device not in DEVICES:
        raise DeviceError(f"device {device!r}: use one of {', '.join(DEVICES)}")
    import torch
    cuda, mps = has_cuda(torch), _has_mps(torch)
    if device == "auto":
        return "cuda" if cuda else "mps" if mps else "cpu"
    if device == "cuda" and not cuda:
        raise DeviceError("device cuda asked for and no CUDA or ROCm GPU is visible: a CPU-only PyTorch build, or "
                          "HIP_VISIBLE_DEVICES / CUDA_VISIBLE_DEVICES hide it (python -m pbtrain doctor)")
    if device == "mps" and not mps:
        raise DeviceError("device mps asked for and Metal is not available: it needs Apple silicon, macOS 12.3 or "
                          "newer and a PyTorch build with MPS (python -m pbtrain doctor)")
    return device


def synchronize(device):
    """Wait until the kernels queued on `device` have run (so that a timer measures only what follows)."""
    import torch
    kind = str(device).split(":")[0]
    if kind == "cuda" and has_cuda(torch):
        torch.cuda.synchronize()
    elif kind == "mps" and _has_mps(torch):
        torch.mps.synchronize()
