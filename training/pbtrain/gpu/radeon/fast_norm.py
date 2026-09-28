"""Grouped RMSNorm, optionally gated -- out = rmsnorm_g(gate ? Y*silu(z) : x) * w -- with per-element
HIP kernels and torch reductions. Same maths as `RMSNorm` and as `GatedRMSNorm(Y * F.silu(z))`.

A first attempt (one thread per ROW, looping over the 256 channels) was 9x SLOWER than eager.
This one follows the rule that works on this GPU: one thread per output element, reductions
by torch. Eager makes ~6 passes over the tensor forward and ~15 backward; this makes 3 and 6.
Not bit-identical to eager (reduction order).
"""
import torch
from .hipk import kernel

_SRC = r"""
#define IDX int i = blockDim.x * blockIdx.x + threadIdx.x; if (i >= n) return;
__device__ static inline float silu(float z) { return z / (1.0f + expf(-z)); }
__device__ static inline float dsilu(float z) { float s = 1.0f / (1.0f + expf(-z)); return s * (1.0f + z * (1.0f - s)); }

// u = Y * silu(z); z lives inside a wider buffer (row stride zs)
extern "C" __global__ void gate_fwd(const float* Y, const float* z, float* u, int D, int zs, int n) {
    IDX; u[i] = Y[i] * silu(z[(i / D) * zs + i % D]);
}
// out = u * r[row, group] * w[j]
extern "C" __global__ void norm_fwd(const float* u, const float* r, const float* w, float* out, int D, int G, int n) {
    IDX; int j = i % D; out[i] = u[i] * r[(i / D) * G + j / (D / G)] * w[j];
}
// t = dout*w*u (its group sum gives the projection term), v = dout*u*r (its row sum gives dw)
extern "C" __global__ void norm_bw1(const float* dout, const float* u, const float* r, const float* w,
                                    float* t, float* v, int D, int G, int n) {
    IDX; int j = i % D; float du = dout[i] * u[i]; t[i] = du * w[j]; v[i] = du * r[(i / D) * G + j / (D / G)];
}
// du = r*(dout*w - u*r*m), m = r*sum_group(t)/n_g ;  gated: dY = du*silu(z), dz = du*Y*silu'(z)
extern "C" __global__ void norm_bw2(const float* dout, const float* u, const float* r, const float* w, const float* m,
                                    const float* Y, const float* z, float* dY, float* dz, int D, int G, int zs, int n, int gate) {
    IDX; int j = i % D; int rg = (i / D) * G + j / (D / G); float rr = r[rg];
    float du = rr * (dout[i] * w[j] - u[i] * rr * m[rg]);
    if (gate) { float zz = z[(i / D) * zs + j]; dY[i] = du * silu(zz); dz[i] = du * Y[i] * dsilu(zz); }
    else dY[i] = du;
}
"""
_K = {}


def _run(name, n, args, blk=256):
    if name not in _K:
        _K[name] = kernel(_SRC, name)
    _K[name](grid=((n + blk - 1) // blk, 1, 1), block=(blk, 1, 1), args=args)


def _rows(x):
    """Dense rows possibly inside a wider buffer -> (tensor, row stride)."""
    T, D = x.shape[-2], x.shape[-1]
    if x.stride(-1) == 1 and x.stride(0) == T * x.stride(1):
        return x, x.stride(1)
    return x.contiguous(), D


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, z, w, G, eps):
        gate = z is not None
        x = x.contiguous()
        B, T, D = x.shape
        n = x.numel()
        if gate:
            z, zs = _rows(z)
            u = torch.empty_like(x)
            _run("gate_fwd", n, [x, z, u, D, zs, n])
        else:
            z, zs, u = x, D, x
        ng = D // G
        r = torch.rsqrt(torch.linalg.vector_norm(u.view(B, T, G, ng), dim=-1).square() / ng + eps).contiguous()
        out = torch.empty_like(x)
        w = w.contiguous()
        _run("norm_fwd", n, [u, r, w, out, D, G, n])
        ctx.save_for_backward(x, z, u, r, w)
        ctx.meta = (G, zs, gate)
        return out

    @staticmethod
    def backward(ctx, dout):
        x, z, u, r, w = ctx.saved_tensors
        G, zs, gate = ctx.meta
        B, T, D = x.shape
        n, ng = x.numel(), D // G
        dout = dout.contiguous()
        t, v = torch.empty_like(x), torch.empty_like(x)
        _run("norm_bw1", n, [dout, u, r, w, t, v, D, G, n])
        m = (t.view(B, T, G, ng).sum(-1) * r / ng).contiguous()
        dw = v.sum((0, 1))
        dY = t                                                    # `t` is dead from here on: reuse its memory
        dz = v if gate else None
        _run("norm_bw2", n, [dout, u, r, w, m, x, z, dY, v, D, G, zs, n, int(gate)])
        return dY, dz, dw, None, None


def ok(x):
    return x.is_cuda and x.dtype == torch.float32 and x.dim() == 3


def rmsnorm(x, mod):
    return _Norm.apply(x, None, mod.weight, 1, mod.eps)


def gated_rmsnorm(Y, z, mod):
    """== mod(Y * F.silu(z)) with `mod` a GatedRMSNorm."""
    return _Norm.apply(Y, z, mod.weight, mod.n_groups, mod.eps)


class Split(torch.autograd.Function):
    """proj -> (proj[..., :k], proj[..., k:]) as views. Autograd's own backward for two slices is
    two zero-filled full-size buffers, two copies and an add; here it is ONE concatenation."""

    @staticmethod
    def forward(ctx, proj, k):
        ctx.k, ctx.shape = k, proj.shape
        return proj[..., :k], proj[..., k:]

    @staticmethod
    def backward(ctx, ga, gb):
        k, shape = ctx.k, ctx.shape
        if ga is None:
            ga = gb.new_zeros(*shape[:-1], k)
        if gb is None:
            gb = ga.new_zeros(*shape[:-1], shape[-1] - k)
        return torch.cat([ga, gb], -1), None
