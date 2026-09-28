"""GPU backends. The CPU (the stock PyTorch code) is the REFERENCE: the golden fingerprints are CPU fingerprints. A
backend replaces hot paths at run time with kernels that compute the same maths in another summation order: close to
the CPU's results, not bit for bit.

    from pbtrain.gpu import enable
    enable("radeon")          # before building the model; only affects CUDA/ROCm tensors

Backends:
    cuda     stock PyTorch on any CUDA (NVIDIA) or ROCm (AMD) device (nothing to install)
    mps      stock PyTorch on the GPU of Apple silicon, through Metal (nothing to install)
    radeon   fused HIP kernels for PyTorch ROCm on Windows, measured on an RX 6800 XT (gfx1030); see radeon/__init__.py

Which backend goes with which device is decided by `backend_for` (the CLI's `--device` / `--backend`): `cpu` only with
the CPU, `cuda` and `radeon` only with `--device cuda`, `mps` only with `--device mps`.

Determinism: whenever pbtrain trains or evaluates on a GPU it asks PyTorch for deterministic algorithms
(`pbtrain.runtime.gpu_determinism`). What was MEASURED: on the RX 6800 XT (PyTorch ROCm on Windows), two runs of the same
job give the same bits with the stock and with the radeon backend, and seeds trained in parallel give the serial run's
bits (training/tests/test_devices.py with PBTRAIN_GPU_TESTS=1). Other GPUs have not been measured by the maintainers.

Rules (training/docs/gpu.md): ONE GPU job at a time (`pbtrain.runtime.gpu_lock`; VRAM overflow under ROCm on Windows
can hang the machine); size parallel seeds from the VRAM table; never compare a GPU-trained model with a CPU golden.
"""
from ..runtime import DeviceError

BACKENDS = ("cpu", "cuda", "mps", "radeon")
RADEON_MAX_WINDOW = 512       # the fused kernels are validated up to 512-byte windows; 1,024 crashed


def backend_for(device, backend=None):
    """The backend a job on `device` runs with; DeviceError for a combination that would not do what it says."""
    kind = str(device).split(":")[0]
    if backend is not None and backend not in BACKENDS:
        raise DeviceError(f"unknown backend {backend!r}; known: {BACKENDS}")
    if kind == "cpu":
        if backend not in (None, "cpu"):
            raise DeviceError(f"--backend {backend} runs on a GPU: add --device {'mps' if backend == 'mps' else 'cuda'}")
        return "cpu"
    if kind == "cuda":
        if backend in (None, "cuda", "radeon"):
            return backend or "cuda"
        raise DeviceError(f"--backend {backend} does not run on --device cuda (stock `cuda`, or `radeon` on ROCm for "
                          f"Windows); the CPU's code is `--device cpu`")
    if kind == "mps":
        if backend in (None, "mps"):
            return "mps"
        raise DeviceError(f"--backend {backend} does not run on --device mps (only the stock `mps` backend does)")
    raise DeviceError(f"unknown device {device!r}")


def enable(name):
    if name not in BACKENDS:
        raise DeviceError(f"unknown backend {name!r}; known: {BACKENDS}")
    if name == "radeon":
        from .radeon import install
        install()
    return name
