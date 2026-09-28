"""Radeon backend (PyTorch ROCm on Windows, measured on an RX 6800 XT / gfx1030): fused HIP kernels compiled at run
time with hiprtc.

`install()` replaces, at run time, only the hot paths training uses (GPU, float32, conv width 4, no incoming conv
state, windows of at most 512 bytes); anything else falls through to the reference code:
    ssm_v2 scan       conv1d + SiLU fused (fast_conv), the whole SSD scan as one autograd Function (fast_ssd)
    ssm_v2 forward    the z | xBC split with a one-pass backward, gate + grouped RMSNorm fused (fast_norm)
    ternary linears   F.linear forward; deterministic GEMM formulations in the backward (fast_linear)
    tag head          the same deterministic linear
Parameter names and shapes do not change: checkpoints, export and the engine see exactly the same model. Same maths,
~1e-7 apart per op, NOT bit-identical to the CPU reference; two runs of the same job gave the same bits on the RX 6800
XT (training/tests/test_devices.py, PBTRAIN_GPU_TESTS=1). The reduction block sizes of the kernels are fixed constants
(they decide the summation order, so they are not tunable). Measured on `nano`: 74 -> 32 ms/step for one process;
4 processes 22.6 ms/step aggregate.

Rule: this module REPLACES `SSMv2Block.forward` (causal). Anything added inside the block must be mirrored here, and
training/tests/test_gpu_backend.py must pass on the GPU box (it compares the loss and every gradient, element by
element, with the reference, for a causal and a hydra model).
"""
import sys

import torch

_INSTALLED = False
TESTED_ARCH = "gfx1030"


def install():
    global _INSTALLED
    if _INSTALLED:
        return
    from .hipk import setup
    setup()                                                    # a clear error off PyTorch ROCm for Windows
    if torch.cuda.is_available():
        arch = getattr(torch.cuda.get_device_properties(0), "gcnArchName", "") or "unknown"
        if not arch.startswith(TESTED_ARCH):
            print(f"pbtrain: WARNING the radeon backend was validated on {TESTED_ARCH}; this GPU is {arch}: run "
                  f"PBTRAIN_GPU_TESTS=1 python -m pytest training/tests/test_gpu_backend.py before trusting it",
                  file=sys.stderr)
    from ...model import blocks as B
    from ...model import heads as Hd
    from ...model.layers import ternary as Tn
    from ...model.layers.act import softplus
    from ...model.layers.ssd import ssd_chunked
    from . import fast_norm, fast_ssd
    from .fast_conv import conv_silu
    from .fast_linear import linear

    orig_scan = B.SSMv2Block._scan
    orig_forward = B.SSMv2Block.forward
    orig_tl_forward = Tn.TernaryLinear.forward
    orig_tag_forward = Hd.TagHead.forward

    def _scan(self, xBC, dt, state, mask, reverse=False, include_D=True):
        conv_state = None if state is None else state[1]
        if (self.conv is None or self.conv.width != 4 or conv_state is not None
                or not xBC.is_cuda or xBC.dtype != torch.float32 or xBC.shape[1] < 3):
            return orig_scan(self, xBC, dt, state, mask, reverse, include_D)
        cfg = self.cfg
        b, T, _ = xBC.shape
        di, H, P, N, G = cfg.d_inner, cfg.n_heads, cfg.d_head, cfg.d_state, cfg.n_groups
        ssm_state = None if state is None else state[0]
        if reverse:
            xBC = B.flip_valid(xBC, mask)
            dt = B.flip_valid(dt, mask)
        conv_state = xBC[:, -3:].transpose(1, 2)               # what CausalConv1d would return
        xBC = conv_silu(xBC, self.conv.conv)
        A = -torch.exp(self.A_log.float())
        D = self.D.float() if include_D else None
        if fast_ssd.supported(xBC, dt, mask, ssm_state, cfg.chunk_size):
            Y, ssm_state = fast_ssd.ssd(xBC, dt, A, D, H, P, G, N, cfg.chunk_size)
            x = None if cfg.direction == "causal" else xBC[..., :di].reshape(b, T, H, P)
            return Y, x, (ssm_state, conv_state)
        x = xBC[..., :di].reshape(b, T, H, P)
        Bm = xBC[..., di:di + G * N].reshape(b, T, G, N)
        Cm = xBC[..., di + G * N:].reshape(b, T, G, N)
        Y, ssm_state = ssd_chunked(x, dt, A, Bm, Cm, D, chunk_size=cfg.chunk_size, initial_state=ssm_state, mask=mask)
        return Y.reshape(b, T, di), x, (ssm_state, conv_state)

    def _project(self, u_norm):
        proj = self.in_proj(u_norm)
        z, xBC = fast_norm.Split.apply(proj, self.cfg.d_inner)
        dt = softplus(linear(u_norm, self.dt_proj.weight).float() + self.dt_bias.float())
        return z, xBC, dt

    def _forward(self, u, state=None, mask=None):
        if self.cfg.direction != "causal" or not fast_norm.ok(u):
            return orig_forward(self, u, state, mask)
        z, xBC, dt = self._project(self.norm(u))
        Y, _, state = self._scan(xBC, dt, state, mask)
        return u + self.out_proj(fast_norm.gated_rmsnorm(Y, z, self.gated_norm)), state

    def _tl_forward(self, x):
        if not self.quantize or not x.is_cuda:
            return orig_tl_forward(self, x)
        y = linear(x, self.effective_weight())                 # code x scale (x gain), thr honoured
        if self.lora_A is not None:
            y = y + self.lora_delta(x, linear)
        return y

    def _tag_forward(self, h, mask=None):
        if h.is_cuda and h.dtype == torch.float32 and h.dim() == 3 and h.is_contiguous():
            return linear(h, self.proj.weight) + self.proj.bias
        return orig_tag_forward(self, h, mask)

    B.SSMv2Block._scan = _scan
    B.SSMv2Block._project = _project
    B.SSMv2Block.forward = _forward
    Tn.TernaryLinear.forward = _tl_forward
    Hd.TagHead.forward = _tag_forward
    _INSTALLED = True
