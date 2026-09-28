"""Runtime-compiled HIP kernels on PyTorch ROCm for Windows, with no compiler toolchain.

`torch.cuda._compile_kernel` works through hiprtc, but the PyTorch ROCm wheels for Windows look for `hiprtc0701.dll`
while the ROCm SDK wheel ships `hiprtc07013.dll`. `setup()` finds the real one in the SDK and hands it to torch: it
replaces the private `torch.cuda._utils._get_hiprtc_library`, and sets ROCM_HOME, ROCM_PATH, HIP_PATH and CUDA_HOME to
the SDK when they are unset (hiprtc looks for its headers there). Importing this module changes nothing; `setup()` runs
when the Radeon backend is installed and stops with a clear message anywhere else than PyTorch ROCm on Windows with the
ROCm SDK package (`_rocm_sdk_core`).

Shape-specialised kernels (`specialised`): per-element kernels spend most of their time turning the flat index into
(b, t, h, p): 4-5 integer div/mod by RUNTIME values, which on a GPU cost tens of cycles each. hiprtc compiles at run
time, so the compiler is told the values: for every `int` parameter whose name is in `consts` an
`if (p != VALUE) __builtin_unreachable();` is inserted at the top of the kernel. The signature does not change; the
optimiser folds the parameter to a constant and div/mod become shifts and multiplies.
"""
import ctypes
import glob
import os
import re

import torch

from ...runtime import DeviceError

_SDK_VARS = ("ROCM_HOME", "ROCM_PATH", "HIP_PATH", "CUDA_HOME")
_state = {"dll": None, "lib": None}
_spec = {}


def setup():
    """Makes hiprtc usable by torch (once); DeviceError off PyTorch ROCm for Windows or without the ROCm SDK package."""
    if _state["dll"] is not None:
        return _state["dll"]
    if os.name != "nt" or not getattr(torch.version, "hip", None):
        raise DeviceError("the radeon backend runs on PyTorch ROCm for Windows (it compiles its kernels with the ROCm "
                          "SDK's hiprtc); this is " + (f"PyTorch {torch.__version__} on {os.name}") +
                          ": use the stock backend (--device cuda without --backend)")
    try:
        import _rocm_sdk_core
    except ImportError:
        raise DeviceError("the radeon backend needs the ROCm SDK Python package (`_rocm_sdk_core`, installed with the "
                          "PyTorch ROCm wheels for Windows): it is missing, use the stock backend")
    root = os.path.dirname(_rocm_sdk_core.__file__)
    dlls = [p for p in sorted(glob.glob(os.path.join(root, "bin", "hiprtc*.dll"))) if "builtins" not in p]
    if not dlls:
        raise DeviceError(f"no hiprtc DLL in {os.path.join(root, 'bin')}: the radeon backend cannot compile its kernels")
    for v in _SDK_VARS:                                           # header lookup only
        os.environ.setdefault(v, root)
    import torch.cuda._utils as _u
    _u._get_hiprtc_library = _hiprtc
    _state["dll"] = dlls[0]
    return dlls[0]


def _hiprtc():
    if _state["lib"] is None:
        dll = setup()
        os.add_dll_directory(os.path.dirname(dll))
        lib = ctypes.CDLL(dll)
        for nv, hip in (("GetErrorString", "GetErrorString"), ("CreateProgram", "CreateProgram"),
                        ("DestroyProgram", "DestroyProgram"), ("CompileProgram", "CompileProgram"),
                        ("GetPTXSize", "GetCodeSize"), ("GetPTX", "GetCode"),
                        ("GetProgramLogSize", "GetProgramLogSize"), ("GetProgramLog", "GetProgramLog"),
                        ("AddNameExpression", "AddNameExpression"), ("GetLoweredName", "GetLoweredName")):
            setattr(lib, "nvrtc" + nv, getattr(lib, "hiprtc" + hip))
        _state["lib"] = lib
    return _state["lib"]


def kernel(src, name, **kw):
    setup()
    return torch.cuda._compile_kernel(src, name, **kw)


def specialised(src, name, consts):
    key = (id(src), name, tuple(sorted(consts.items())))
    if key not in _spec:
        m = re.search(r"void\s+" + name + r"\s*\(([^)]*)\)\s*\{", src)
        params = re.findall(r"\bint\s+(\w+)", m.group(1))
        assume = "".join(f" if ({p} != {int(consts[p])}) __builtin_unreachable();" for p in params if p in consts)
        _spec[key] = kernel(src[:m.end()] + assume + src[m.end():], name)
    return _spec[key]


def demo():
    """Compiles and runs one tiny kernel on the GPU (`python -m pbtrain.gpu.radeon.hipk`, on the GPU box)."""
    k = kernel("extern \"C\" __global__ void addk(float* a, float* o, int n){"
               "int i=blockDim.x*blockIdx.x+threadIdx.x; if(i<n) o[i]=a[i]*2+1;}", "addk")
    a = torch.arange(8., device="cuda")
    o = torch.empty_like(a)
    k(grid=(1, 1, 1), block=(8, 1, 1), args=[a, o, 8])
    print(o)


if __name__ == "__main__":
    demo()
