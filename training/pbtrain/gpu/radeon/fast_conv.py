"""Fused causal depthwise conv1d (width 4) + SiLU, forward and backward, as HIP kernels.

Same maths as `CausalConv1d` followed by `F.silu` with a zero initial state. It is NOT
bit-identical to MIOpen (different summation order, ~1e-7), so it is opt-in.
"""
import torch

from .hipk import kernel

_L = 16      # time steps per thread of the weight-gradient reduction: it fixes the summation order, so it is a constant

_SRC = r"""
extern "C" __global__ void conv_fwd(const float* x, const float* w, const float* bias, float* y,
                                    int T, int C, int xs, long n, int act) {
    long i = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int c = i % C; long bt = i / C; int t = bt % T;
    float s = bias[c];
    for (int k = 0; k < 4; ++k) { int tt = t - 3 + k; if (tt >= 0) s += w[c * 4 + k] * x[(bt - 3 + k) * xs + c]; }
    y[i] = act ? s / (1.0f + expf(-s)) : s;
}
// dpre = dy * silu'(pre), with pre recomputed from x.
extern "C" __global__ void conv_dpre(const float* x, const float* w, const float* bias, const float* dy,
                                     float* dpre, int T, int C, int xs, long n, int act) {
    long i = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int c = i % C; long bt = i / C; int t = bt % T;
    float s = bias[c];
    for (int k = 0; k < 4; ++k) { int tt = t - 3 + k; if (tt >= 0) s += w[c * 4 + k] * x[(bt - 3 + k) * xs + c]; }
    float g = 1.0f;
    if (act) { float sg = 1.0f / (1.0f + expf(-s)); g = sg * (1.0f + s * (1.0f - sg)); }
    dpre[i] = dy[i] * g;
}
extern "C" __global__ void conv_dx(const float* dpre, const float* w, float* dx, int T, int C, long n) {
    long i = (long)blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int c = i % C; long bt = i / C; int t = bt % T;
    float s = 0.0f;
    for (int k = 0; k < 4; ++k) { int tt = t + 3 - k; if (tt < T) s += w[c * 4 + k] * dpre[(bt + 3 - k) * C + c]; }
    dx[i] = s;
}
// dw, dbias: two-level reduction. thread = (batch row, block of L steps, channel) -> 5 partial sums; torch adds
// the partials. THE rule on this GPU: threads run in lockstep, so at every loop iteration ADJACENT THREADS MUST
// READ ADJACENT MEMORY. With the channel as the fastest index, 256 neighbouring threads read 256 contiguous
// floats per step. Indexed the other way round (neighbours 3 KB apart) this same kernel is 4-8x slower.
extern "C" __global__ void conv_dw(const float* x, const float* dpre, float* out,
                                   int B, int T, int C, int xs, int L, int n) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= n) return;
    int nb = (T + L - 1) / L; int c = i % C; int r = i / C; int blk = r % nb; int b = r / nb;   // c FASTEST: neighbours read neighbours
    float a0 = 0, a1 = 0, a2 = 0, a3 = 0, ab = 0;
    int t1 = (blk + 1) * L < T ? (blk + 1) * L : T;
    for (int t = blk * L; t < t1; ++t) {
        long bt = (long)b * T + t; float d = dpre[bt * C + c];
        ab += d; a3 += d * x[bt * xs + c];
        if (t >= 1) a2 += d * x[(bt - 1) * xs + c];
        if (t >= 2) a1 += d * x[(bt - 2) * xs + c];
        if (t >= 3) a0 += d * x[(bt - 3) * xs + c];
    }
    float* o = out + (long)i * 5;
    o[0] = a0; o[1] = a1; o[2] = a2; o[3] = a3; o[4] = ab;
}
"""
_K = {}


def _k(name):
    if name not in _K:
        _K[name] = kernel(_SRC, name)
    return _K[name]


def _grid(n, blk=256):
    return ((n + blk - 1) // blk, 1, 1), (blk, 1, 1)


def _rows(x):
    """(B,T,C) view whose rows are dense but may sit inside a wider buffer -> row stride."""
    B, T, C = x.shape
    if x.stride(2) == 1 and x.stride(0) == T * x.stride(1):
        return x, x.stride(1)
    x = x.contiguous()
    return x, C


class ConvSiLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, act):
        B, T, C = x.shape
        x, xs = _rows(x)
        w = weight.reshape(C, 4).contiguous()
        y = torch.empty(B, T, C, device=x.device, dtype=x.dtype)
        n = B * T * C; g, b = _grid(n)
        _k("conv_fwd")(grid=g, block=b, args=[x, w, bias, y, T, C, xs, n, int(act)])
        ctx.save_for_backward(x, w, bias); ctx.act, ctx.xs = int(act), xs
        return y

    @staticmethod
    def backward(ctx, dy):
        x, w, bias = ctx.saved_tensors
        B, T, C = dy.shape
        dy = dy.contiguous(); n = B * T * C; g, b = _grid(n)
        dpre = torch.empty_like(dy); dx = torch.empty_like(dy)
        nb = (T + _L - 1) // _L
        part = torch.empty(B * nb, C, 5, device=dy.device, dtype=torch.float32)
        _k("conv_dpre")(grid=g, block=b, args=[x, w, bias, dy, dpre, T, C, ctx.xs, n, ctx.act])
        _k("conv_dx")(grid=g, block=b, args=[dpre, w, dx, T, C, n])
        gc, bc = _grid(C * B * nb)
        _k("conv_dw")(grid=gc, block=bc, args=[x, dpre, part, B, T, C, ctx.xs, _L, C * B * nb])
        red = part.sum(0, dtype=torch.float64).to(dy.dtype)
        return dx, red[:, :4].reshape(C, 1, 4), red[:, 4].contiguous(), None


def conv_silu(x, conv, act=True):
    """x (B,T,C) float32 on GPU; `conv` is the nn.Conv1d of a CausalConv1d."""
    return ConvSiLU.apply(x, conv.weight, conv.bias, act)
