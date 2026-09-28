# Training on any hardware

AI Specialists are small (1.9 to 54 million parameters for the released examples), so they train on whatever you
have: a laptop CPU, an NVIDIA or AMD gaming GPU, or the GPU of an Apple silicon Mac. So far the maintainers have
trained on CPUs and on one AMD Radeon under Windows; the other GPUs run the same stock PyTorch code (below). The CPU is
the reference: the golden fingerprints are CPU fingerprints of small test runs (compared on the platform that recorded
them). The released
specialists were trained on a GPU (an RX 6800 XT, below): retraining a released recipe, on a CPU or a GPU, reproduces
its weights only approximately. A GPU gives the same maths faster, with kernel-level rounding differences.

```bash
python -m pbtrain run TASK                                              # CPU, the reference (default)
python -m pbtrain run TASK --device auto --parallel 3                   # the best device found: cuda, then mps, then cpu
python -m pbtrain run TASK --device cuda --parallel 3                   # NVIDIA (CUDA) or AMD (ROCm)
python -m pbtrain run TASK --device mps                                 # Apple silicon (Metal)
python -m pbtrain run TASK --device cuda --backend radeon --parallel 3  # fused HIP kernels (ROCm on Windows)
```

`python -m pbtrain doctor` tells you which PyTorch build you have and which devices it can use (it asks a child process
for the CUDA / ROCm devices, and whether MPS is available). `--device` and `--backend` exist on `train` and `run`; the
other commands run on the CPU.

## Hardware

| Hardware | PyTorch build to install | `--device` | Backend | Status |
|---|---|---|---|---|
| Any x86-64 or Arm CPU (Linux, Windows, macOS) | CPU wheels (`training/requirements-ci.txt`) | `cpu` | stock | The reference; tested in CI on Linux, Windows and macOS |
| AMD Radeon, Windows (ROCm for Windows) | ROCm wheels ([the build tested](#the-tested-rocm-build-on-windows)) | `cuda` | stock or `radeon` | Tested: the released specialists were trained on an RX 6800 XT (gfx1030), and `test_devices.py` and `test_gpu_backend.py` pass on it (PyTorch 2.10 ROCm) |
| AMD, Linux (ROCm) | ROCm wheels from pytorch.org | `cuda` | stock | Same PyTorch code path as above (`torch.cuda`); not run by the maintainers |
| NVIDIA (CUDA) | CUDA wheels from pytorch.org | `cuda` | stock | Same PyTorch code path as ROCm (`torch.cuda`); this release of pbtrain has not been run on NVIDIA by the maintainers |
| Apple silicon (M1 or newer, macOS 12.3+) | the default macOS wheels | `mps` | stock | float32 only, which is what the trainer uses; operations Metal lacks fall back to the CPU (`PYTORCH_ENABLE_MPS_FALLBACK=1`). The model code was developed partly on an Apple M3 Pro (PyTorch 2.12.1), where two numerical problems were found and are worked around (`pbtrain/model/layers/act.py`, `ssd.py`); this release of pbtrain and its tests have not been run on Apple silicon by the maintainers |

"Not run" means what it says: the code uses only stock PyTorch operations in float32 on these devices, and
`training/tests/test_devices.py` (`PBTRAIN_GPU_TESTS=1 python -m pytest -q -rs -s training/tests/test_devices.py`)
checks, on every GPU it finds, that `train_spans` gives the CPU's losses over six steps within 1e-4 (relative) and the
CPU's log-probabilities within 1e-4 (the RX 6800 XT: 1.2e-7 and 6.2e-6), that two runs of the same job give the same
bits, and that seeds trained in parallel give the serial run's bits. If you run it on one of them, a report with its
output and that of `python -m pbtrain doctor` is very welcome. In PowerShell, set the variable first:
`$env:PBTRAIN_GPU_TESTS = "1"; python -m pytest -q -rs -s training/tests/test_devices.py`.

Install PyTorch for your hardware with the selector at <https://pytorch.org/get-started/locally/> (in a virtual
environment, as in the [Quick start](../../README.md#quick-start)), then `python -m pip install -e "training[test]"`.

### The tested ROCm build on Windows

The RX 6800 XT runs used a nightly build of AMD's TheRock project for the gfx103X family, which bundles ROCm (no HIP
SDK to install), with Python 3.12 on Windows 10 22H2 (display driver 32.0.21025.1024 at the time of writing):

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install "torch==2.10.0+rocm7.13.0a20260421" --index-url https://rocm.nightlies.amd.com/v2-staging/gfx103X-dgpu/ --extra-index-url https://pypi.org/simple
```

Nightly builds are replaced over time: when this one is no longer served, a later build of the same index is expected
to work, but it is not the one tested.

## Rules

- **One GPU job at a time**: any `--device` other than `cpu` takes `$PUREBYTE_WORK/locks/gpu.lock`, an operating-system
  lock (`fcntl.flock`, or `msvcrt.locking` on Windows) that the kernel releases when its owner ends, however it ends; a
  second job is refused, naming the owner, while the lock is held. Overflowing VRAM under ROCm on Windows can hang the
  machine.
- **A missing device is an error**: `--device cuda` or `--device mps` on a machine without that GPU stops at once with a
  message; it never falls back to a silent CPU run (a ROCm build that reports CUDA available with every device hidden
  counts as no device). `--device auto` says on standard error which device it chose, the CPU included.
- **Device and backend go together**: `--backend cpu` only with `--device cpu`, `cuda` or `radeon` only with `--device
  cuda`, `mps` only with `--device mps`; any other pair is refused. The seed cache keeps runs of different devices and
  backends apart.
- **`--parallel N`** trains N seeds at once in worker processes (the control is one more job). Workers are seeded like a
  serial run and compute with the parent's thread count: parallel and serial give the same weights (tested on the CPU,
  and on the RX 6800 XT with `PBTRAIN_GPU_TESTS=1`). Memory multiplies by N.
- **Determinism**: on a GPU, pbtrain asks PyTorch for deterministic algorithms (`torch.use_deterministic_algorithms(True,
  warn_only=True)`: an operation without a deterministic implementation is a warning), cuDNN / MIOpen without
  autotuning, no TF32, and a fixed cuBLAS workspace (`CUBLAS_WORKSPACE_CONFIG=:4096:8`); a CPU run never changes those
  settings. The radeon backend's GEMM formulations are chosen by a static rule on the shape, never by timing. Measured:
  two runs of the same job give the same bits on the RX 6800 XT, stock and radeon backends
  (`test_devices.py`). On other GPUs this has not been measured: expect agreement within float32 tolerance, and bits
  only when the test passes there. The attention presets (`nano_a`, `nano_an`) call PyTorch's fused attention, whose
  backward can have no deterministic implementation on CUDA and ROCm: there expect the warning, and runs that differ in
  the last bits. Random numbers are drawn on the CPU, so the initial weights are the same on every device.
- **Never compare a GPU-trained model with a CPU golden**: the goldens are CPU fingerprints.

## Memory per job (measured, float32, RX 6800 XT)

| Size | Window x batch | VRAM | Parallel jobs on 16 GB |
|---|---|---|---|
| `nano` | 512 x 16 | 1.8 GB | 4 |
| `tiny_n18` | 512 x 16 | 3.9 GB | 3 (4 overflowed) |
| `nano` | 1,024 x 32 | 5.9 GB | 2 |

On Apple silicon the GPU shares the Mac's unified memory: the same sizes apply to the memory left free.

## The Radeon backend

`pbtrain.gpu.enable("radeon")` (or `--device cuda --backend radeon`) installs fused HIP kernels compiled at run time
with hiprtc (no compiler toolchain needed), for an RX 6800 XT (gfx1030) with PyTorch ROCm on Windows. Anywhere else
(another platform, no ROCm SDK package) it stops with a message; on another AMD GPU it warns that it was validated on
gfx1030 only:

| Piece | What it fuses |
|---|---|
| `fast_conv` | depthwise conv1d + SiLU |
| `fast_ssd` | the whole chunked SSD scan, with a hand-derived backward |
| `fast_norm` | the z / xBC split, gate + grouped RMSNorm |
| `fast_linear` | deterministic GEMM formulations for the backward of the linears |

Measured on `nano`: 74 -> 32 ms per step for one process; four processes 22.6 ms per step aggregate (3.3x). Only the
training path is replaced (GPU tensors, float32, conv width 4, no carried state, windows of at most 512 bytes);
everything else falls through to the reference code. 1,024-byte windows crashed the kernels: pbtrain refuses the
backend for a task whose `seq_len` is above 512 (extraction tasks train without it). The block sizes of the kernels'
reductions fix the summation order, so they are constants, not settings.

Tried and dropped on this GPU: HIP graphs (no gain), TunableOp (-10 % time, breaks determinism), fp16 / bf16 (not
accelerated on RDNA2, changes the numbers), torch.compile / Triton (not available on ROCm for Windows), a fused plain
RMSNorm (slower than eager), chunk size tuning (no effect).

Rules for new kernels on this GPU: neighboring threads read neighboring memory; one thread per output element;
reductions are left to torch; measure the aggregate of several processes, not one.

**Any change inside a block must be mirrored in `pbtrain/gpu/radeon/__init__.py`**: the backend replaces the block's
forward. `PBTRAIN_GPU_TESTS=1 python -m pytest training/tests/test_gpu_backend.py` (the backend runs on Windows only:
in PowerShell, `$env:PBTRAIN_GPU_TESTS = "1"; python -m pytest training/tests/test_gpu_backend.py`) compares the loss
and every gradient, element by element (|radeon - reference| <= 1e-4 x max|reference| + 1e-3 x |reference| per tensor), with
the reference on the GPU, for a causal and a hydra model (on the RX 6800 XT the largest difference is 6 % of its
tolerance).
