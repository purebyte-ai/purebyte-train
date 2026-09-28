"""Mamba-2 SSD chunked scan as ONE autograd Function: own memory layout, fused HIP kernels for
everything pointwise, hand-derived backward. Same algorithm as `pbtrain.model.layers.ssd.ssd_chunked`
(diagonal block + per-chunk states + inter-chunk recurrence + off-diagonal), same exponent
floor (-30), same clamp semantics in the backward. Not bit-identical (summation order).

What it removes with respect to the eager version, per block and per step:
  - the permute->contiguous copies einsum makes around every bmm (tensors live as
    (b, h, chunk, Q, .) from the start, so every bmm sees contiguous batches);
  - tril / masked_fill / where / cumsum / exp / mul chains over (b,h,nc,Q,Q): one kernel;
  - repeat_interleave of B and C to every head: C.B^T is computed per GROUP (4x fewer) and
    the kernels index the group of each head;
  - slicing xBC into x, B, C and the slice_backward + add that autograd pays for it:
    the Function takes xBC whole and writes ONE gradient buffer.

Kernel rule learnt the hard way on this GPU: ONE THREAD PER OUTPUT ELEMENT. A thread that
loops over 64 elements exposes GDDR6 latency and runs 5-10x slower than the same work spread
over 64 threads. Reductions are left to torch, which does them well.

Only the path training uses: no mask, no initial state, T % chunk == 0, T <= 512, float32 on GPU.
"""
import torch
import torch.nn.functional as F

from ...model.layers.ssd import _segsum
from .hipk import specialised

MAX_T = 512      # validated window length (1,024-byte windows crashed): longer windows take the eager scan
_DOT_L = 16      # entries per thread of the row-dot reductions: it fixes the summation order, so it is a constant

_SRC = r"""
#define FL (-30.0f)
#define V 1   // elements per thread in the copy kernels. MEASURED: V=8 is 2-3x SLOWER (unpack_y 0.21 -> 0.59 ms): on this
               // GPU any loop inside a thread costs more than spreading the same work over more threads.
#define IDX int i = blockDim.x * blockIdx.x + threadIdx.x; if (i >= n) return;

// X[b,h,t,p] = src[b,t,h,p] * (use_dt ? dt[b,t,h] : 1)
extern "C" __global__ void pack_x(const float* src, const float* dt, float* X, int T, int H, int P, int Cw, int n, int use_dt) {
    IDX; int e = i * V; int p = e % P; int r = e / P; int t = r % T; int bh = r / T; int h = bh % H; int b = bh / H;
    int bt = b * T + t; float s = use_dt ? dt[bt * H + h] : 1.0f; const float* sp = src + bt * Cw + h * P + p; float* o = X + e;
    for (int v = 0; v < V; ++v) o[v] = sp[v] * s;
}
// out[b,g,t,k] = xbc[b,t, off + g*N + k]
extern "C" __global__ void pack_bc(const float* xbc, float* out, int T, int G, int N, int Cw, int off, int n) {
    IDX; int k = i % N; int r = i / N; int t = r % T; int bg = r / T; int g = bg % G; int b = bg / G;
    out[i] = xbc[(b * T + t) * Cw + off + g * N + k];
}
// per chunk (tiny): cumulative sum of a = dt*A in DOUBLE, decay to the end (dec) and from the start (e) of
// the chunk, and whether each exponent is above the floor (the clamp passes gradient only there).  thread = (b,h,c)
extern "C" __global__ void mk_dec(const float* dt, const float* A, double* acs, float* dec, float* e, float* uB, float* uC,
                                  int T, int H, int Q, int n) {
    IDX; int nc = T / Q; int c = i % nc; int h = (i / nc) % H; int b = i / (nc * H);
    double run = 0.0;
    for (int l = 0; l < Q; ++l) { run += (double)(dt[(b * T + c * Q + l) * H + h] * A[h]); acs[i * Q + l] = run; }
    for (int l = 0; l < Q; ++l) {
        float sB = (float)(run - acs[i * Q + l]), sC = (float)acs[i * Q + l];
        uB[i * Q + l] = sB >= FL ? 1.0f : 0.0f; uC[i * Q + l] = sC >= FL ? 1.0f : 0.0f;
        dec[i * Q + l] = expf(sB > FL ? sB : FL); e[i * Q + l] = expf(sC > FL ? sC : FL);
    }
}
// M[ci,l,s] = Gm[group][l,s] * exp(max(acs[l]-acs[s], FL)) for s <= l, else 0
extern "C" __global__ void mk_M(const float* Gm, const double* acs, float* M, int H, int G, int nc, int Q, int n) {
    IDX; int s = i % Q; int r = i / Q; int l = r % Q; int ci = r / Q; int c = ci % nc; int bh = ci / nc; int h = bh % H; int b = bh / H;
    if (s > l) { M[i] = 0.0f; return; }
    float seg = (float)(acs[ci * Q + l] - acs[ci * Q + s]);
    M[i] = Gm[(((b * G + h / (H / G)) * nc + c) * Q + l) * Q + s] * expf(seg > FL ? seg : FL);
}
// out[ci,l,k] = src[group][l,k] * w[ci,l]
extern "C" __global__ void mk_BC(const float* src, const float* w, float* out, int H, int G, int nc, int Q, int N, int n) {
    IDX; int k = i % N; int r = i / N; int l = r % Q; int ci = r / Q; int c = ci % nc; int bh = ci / nc; int h = bh % H; int b = bh / H;
    out[i] = src[(((b * G + h / (H / G)) * nc + c) * Q + l) * N + k] * w[r];
}
// Y[b,t,h,p] = Yd (+ D[h]*x), Yd already holding diagonal + off-diagonal
extern "C" __global__ void unpack_y(const float* Yd, const float* xbc, const float* D, float* Y,
                                    int T, int H, int P, int Cw, int n, int use_D) {
    IDX; int e = i * V; int p = e % P; int r = e / P; int h = r % H; int bt = r / H; int t = bt % T; int b = bt / T;
    int k = ((b * H + h) * T + t) * P + p; float d = use_D ? D[h] : 0.0f; const float* xp = xbc + bt * Cw + h * P + p;
    for (int v = 0; v < V; ++v) Y[e + v] = Yd[k + v] + d * xp[v];
}
// ---- backward ----
// E[ci,l,s] = dM*M where s <= l and the exponent is above the floor, else 0.  dAcs = rowsum(E) - colsum(E)
extern "C" __global__ void bw_E(const float* dM, const float* M, const double* acs, float* E, int Q, int n) {
    IDX; int s = i % Q; int r = i / Q; int l = r % Q; int ci = r / Q;
    E[i] = (s <= l && (float)(acs[ci * Q + l] - acs[ci * Q + s]) >= FL) ? dM[i] * M[i] : 0.0f;
}
// dGm[b,g,c,l,s] = sum_{h in group} dM[h][l,s] * exp(max(seg_h(l,s), FL)), 0 above the diagonal
extern "C" __global__ void bw_Gm(const float* dM, const double* acs, float* dGm, int H, int G, int nc, int Q, int n) {
    IDX; int s = i % Q; int r = i / Q; int l = r % Q; int gi = r / Q; int c = gi % nc; int bg = gi / nc; int g = bg % G; int b = bg / G;
    if (s > l) { dGm[i] = 0.0f; return; }
    int R = H / G; float acc = 0.0f;
    for (int hh = 0; hh < R; ++hh) {
        int ci = (b * H + g * R + hh) * nc + c; float seg = (float)(acs[ci * Q + l] - acs[ci * Q + s]);
        acc += dM[(ci * Q + l) * Q + s] * expf(seg > FL ? seg : FL);
    }
    dGm[i] = acc;
}
// dxbc[B or C part][b,t,g,k] = dg[b,g,t,k] + sum_{h in group} d[h][t,k] * w[h][t]
extern "C" __global__ void bw_BC(const float* dg, const float* d, const float* w, float* dxbc,
                                 int T, int H, int G, int N, int Cw, int off, int n) {
    IDX; int k = i % N; int r = i / N; int t = r % T; int bg = r / T; int g = bg % G; int b = bg / G;
    int R = H / G; float acc = dg[i];
    for (int hh = 0; hh < R; ++hh) { int ri = (b * H + g * R + hh) * T + t; acc += d[ri * N + k] * w[ri]; }
    dxbc[(b * T + t) * Cw + off + g * N + k] = acc;
}
// dxbc[x part][b,t,h,p] = dX[b,h,t,p]*dt[b,t,h] (+ D[h]*gY)
extern "C" __global__ void bw_dx(const float* dX, const float* gY, const float* dt, const float* D, float* dxbc,
                                 int T, int H, int P, int Cw, int n, int use_D) {
    IDX; int e = i * V; int p = e % P; int r = e / P; int h = r % H; int bt = r / H; int t = bt % T; int b = bt / T;
    const float* dx = dX + ((b * H + h) * T + t) * P + p; float* o = dxbc + bt * Cw + h * P + p; float s = dt[r], d = use_D ? D[h] : 0.0f;
    for (int v = 0; v < V; ++v) o[v] = dx[v] * s + d * gY[e + v];
}
// part[row, blk] = sum_{k in block of L} d[row,k] * src[row' ,k], row' = the GROUP's row if grouped else the same row.
// thread = (row, block), block FASTEST (neighbours read neighbours); torch adds the N/L blocks.
extern "C" __global__ void bw_dot(const float* d, const float* src, float* part, int H, int G, int nc, int Q, int N, int L, int n, int grouped) {
    IDX; int nb = N / L; int blk = i % nb; int row = i / nb; int srow = row;
    if (grouped) { int l = row % Q; int ci = row / Q; int c = ci % nc; int bh = ci / nc; int h = bh % H; int b = bh / H; srow = ((b * G + h / (H / G)) * nc + c) * Q + l; }
    const float* dp = d + (long)row * N + blk * L; const float* sp = src + (long)srow * N + blk * L;
    float a = 0.0f; for (int k = 0; k < L; ++k) a += dp[k] * sp[k];
    part[i] = a;
}
// dW[bh,z,c] = <dprev[bh,z,:], Scat[bh,c,:]> over the N*P state entries, only for 1 <= c <= z (W is lower
// triangular and column 0 is the zero initial state). Two-level reduction: thread = (bh, z, c, block of L entries),
// the block index FASTEST so neighbouring threads read neighbouring memory; torch adds the blocks.
// rocBLAS has a ~0.4 ms floor on this product (inner dimension 4096); this is ~4x cheaper.
extern "C" __global__ void bw_dW(const float* dprev, const float* Scat, float* part, int nc, int NP, int L, int n) {
    IDX; int nb = NP / L; int blk = i % nb; int r = i / nb; int c = r % (nc + 1); int q = r / (nc + 1); int z = q % nc; int bh = q / nc;
    if (c == 0 || c > z) { part[i] = 0.0f; return; }
    const float* d = dprev + ((long)bh * nc + z) * NP + blk * L; const float* sc = Scat + ((long)bh * (nc + 1) + c) * NP + blk * L;
    float a = 0.0f; for (int k = 0; k < L; ++k) a += d[k] * sc[k];
    part[i] = a;
}
"""
def _run(name, n, args, blk=256, c=None):
    specialised(_SRC, name, c or {})(grid=((n + blk - 1) // blk, 1, 1), block=(blk, 1, 1), args=args)


class _SSD(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xbc, dt, A, D, H, P, G, N, Q):
        b, T, Cw = xbc.shape
        nc, di, dev = T // Q, H * P, xbc.device
        useD = D is not None
        Dz = D if useD else A
        new = lambda *s: torch.empty(*s, device=dev, dtype=torch.float32)
        nX, nB, nch = b * H * T * P, b * G * T * N, b * H * nc
        K = dict(T=T, H=H, P=P, G=G, N=N, Q=Q, Cw=Cw, nc=nc)
        Xr, Bp, Cp = new(nch, Q, P), new(b * G * nc, Q, N), new(b * G * nc, Q, N)
        _run("pack_x", nX, [xbc, dt, Xr, T, H, P, Cw, nX, 0], c=K)                    # x in chunk layout (kept for the backward)
        _run("pack_bc", nB, [xbc, Bp, T, G, N, Cw, di, nB], c=K)
        _run("pack_bc", nB, [xbc, Cp, T, G, N, Cw, di + G * N, nB], c=K)
        X = Xr * dt.transpose(1, 2).reshape(nch, Q, 1)
        acs = torch.empty(nch, Q, device=dev, dtype=torch.float64)
        dec, e, uB, uC = new(nch, Q), new(nch, Q), new(nch, Q), new(nch, Q)
        _run("mk_dec", nch, [dt, A, acs, dec, e, uB, uC, T, H, Q, nch], blk=64, c=K)
        Gm = torch.bmm(Cp, Bp.transpose(1, 2))                                   # (bGnc,Q,Q): per GROUP
        M = new(nch, Q, Q)
        _run("mk_M", nch * Q * Q, [Gm, acs, M, H, G, nc, Q, nch * Q * Q], c=K)
        Yd = torch.bmm(M, X)
        Bd, Cd = new(nch, Q, N), new(nch, Q, N)
        _run("mk_BC", nch * Q * N, [Bp, dec, Bd, H, G, nc, Q, N, nch * Q * N], c=K)
        _run("mk_BC", nch * Q * N, [Cp, e, Cd, H, G, nc, Q, N, nch * Q * N], c=K)
        S = torch.bmm(Bd.transpose(1, 2), X)                                     # (.,N,P)
        # inter-chunk recurrence: tiny tensors, the very same code as the eager version
        tot = acs.view(b, H, nc, Q)[..., -1].float()
        W = torch.exp(_segsum(F.pad(tot, (1, 0)))).view(b * H, nc + 1, nc + 1)
        Scat = torch.cat([S.new_zeros(b * H, 1, N * P), S.view(b * H, nc, N * P)], 1)
        prev = torch.bmm(W[:, :nc], Scat)                                        # state ENTERING each chunk
        final = torch.bmm(W[:, nc:], Scat)
        Yd.baddbmm_(Cd, prev.view(nch, N, P))                                    # + off-diagonal, in place
        Y = new(b, T, di)
        _run("unpack_y", nX, [Yd, xbc, Dz, Y, T, H, P, Cw, nX, int(useD)], c=K)
        ctx.save_for_backward(xbc, dt, A, Dz, Xr, X, Bp, Cp, acs, dec, e, uB, uC, M, Bd, Cd, prev, Scat, W, tot)
        ctx.dims = (b, T, Cw, H, P, G, N, Q, useD)
        ctx.set_materialize_grads(False)
        return Y, final.view(b, H, N, P).transpose(-1, -2)

    @staticmethod
    def backward(ctx, gY, gfinal):
        xbc, dt, A, Dz, Xr, X, Bp, Cp, acs, dec, e, uB, uC, M, Bd, Cd, prev, Scat, W, tot = ctx.saved_tensors
        b, T, Cw, H, P, G, N, Q, useD = ctx.dims
        nc, di, dev = T // Q, H * P, xbc.device
        nX, nB, nch = b * H * T * P, b * G * T * N, b * H * nc
        K = dict(T=T, H=H, P=P, G=G, N=N, Q=Q, Cw=Cw, nc=nc)
        new = lambda *s: torch.empty(*s, device=dev, dtype=torch.float32)
        gY = torch.zeros(b, T, di, device=dev) if gY is None else gY.contiguous()
        gy = new(nch, Q, P)
        _run("pack_x", nX, [gY, dt, gy, T, H, P, di, nX, 0], c=dict(K, Cw=di))
        dCd = torch.bmm(gy, prev.view(nch, N, P).transpose(1, 2))                # (.,Q,N)
        dprev = torch.bmm(Cd.transpose(1, 2), gy).view(b * H, nc, N * P)
        dM = torch.bmm(gy, X.transpose(1, 2))                                    # (.,Q,Q)
        dX = torch.bmm(M.transpose(1, 2), gy)                                    # (.,Q,P)
        # recurrence backward
        dS = torch.bmm(W[:, :nc, 1:].transpose(1, 2), dprev)
        dW = torch.zeros_like(W)
        NP, L = N * P, 16
        if gfinal is None and NP % L == 0:
            part = new(b * H, nc, nc + 1, NP // L)
            _run("bw_dW", part.numel(), [dprev, Scat, part, nc, NP, L, part.numel()], c=dict(nc=nc, NP=NP, L=L))
            dW[:, :nc] = part.sum(-1)
        else:
            dW[:, :nc] = torch.bmm(dprev, Scat.transpose(1, 2))
        if gfinal is not None:
            gf = gfinal.transpose(-1, -2).reshape(b * H, 1, N * P)
            dS = dS + torch.bmm(W[:, nc:, 1:].transpose(1, 2), gf)
            dW[:, nc:] = torch.bmm(gf, Scat.transpose(1, 2))
        with torch.enable_grad():
            tot_ = tot.detach().requires_grad_()
            W_ = torch.exp(_segsum(F.pad(tot_, (1, 0)))).view(b * H, nc + 1, nc + 1)
            (dtot,) = torch.autograd.grad(W_, tot_, dW)
        dS = dS.view(nch, N, P)
        dBd = torch.bmm(X, dS.transpose(1, 2))                                   # (.,Q,N)
        dX.baddbmm_(Bd, dS)
        # decays: dAcs collects every path into the cumulative sums
        E = new(nch, Q, Q)
        _run("bw_E", nch * Q * Q, [dM, M, acs, E, Q, nch * Q * Q], c=K)
        def rowdot(d, src, width, grouped):
            L = _DOT_L
            part = new(nch * Q, width // L)
            _run("bw_dot", part.numel(), [d, src, part, H, G, nc, Q, width, L, part.numel(), grouped], c=dict(K, N=width, L=L))
            return part.sum(1).view(nch, Q)
        dotB, dotC = rowdot(dBd, Bp, N, 1), rowdot(dCd, Cp, N, 1)
        v = dotB * dec * uB
        dAcs = E.sum(2) - E.sum(1) - v + dotC * e * uC
        dAcs[:, -1] += v.sum(1) + dtot.reshape(nch)
        da = dAcs.flip(1).cumsum(1).flip(1)                                      # a = dt*A ; acs = cumsum(a)
        # B and C: through Gm (per group) and through Bd / Cd (summed over the heads of the group)
        dGm = new(b * G * nc, Q, Q)
        _run("bw_Gm", b * G * nc * Q * Q, [dM, acs, dGm, H, G, nc, Q, b * G * nc * Q * Q], c=K)
        dCg = torch.bmm(dGm, Bp)
        dBg = torch.bmm(dGm.transpose(1, 2), Cp)
        dxbc = new(b, T, Cw)
        _run("bw_BC", nB, [dBg, dBd, dec, dxbc, T, H, G, N, Cw, di, nB], c=K)
        _run("bw_BC", nB, [dCg, dCd, e, dxbc, T, H, G, N, Cw, di + G * N, nB], c=K)
        _run("bw_dx", nX, [dX, gY, dt, Dz, dxbc, T, H, P, Cw, nX, int(useD)], c=K)
        # dt, A, D
        dtp = dt.transpose(1, 2).reshape(nch, Q)
        ddtp = da * A.view(1, H, 1, 1).expand(b, H, nc, Q).reshape(nch, Q) + rowdot(dX, Xr, P, 0)
        ddt = ddtp.view(b, H, T).transpose(1, 2).contiguous()
        dA = (da * dtp).view(b, H, -1).sum((0, 2))
        dD = rowdot(gy, Xr, P, 0).view(b, H, -1).sum((0, 2)) if useD else None
        return dxbc, ddt, dA, dD, None, None, None, None, None


def supported(xbc, dt, mask, initial_state, Q):
    return (mask is None and initial_state is None and xbc.is_cuda and xbc.dtype == torch.float32
            and dt.dtype == torch.float32 and xbc.shape[1] % Q == 0 and xbc.shape[1] <= MAX_T
            and xbc.numel() * 4 < 2 ** 31)   # the kernels index with int32; the largest index is ~numel


def ssd(xbc, dt, A, D, H, P, G, N, Q):
    """xbc (b,T,H*P+2*G*N) = [x | B | C] after conv+SiLU, dt (b,T,H), A (H,), D (H,) or None
    -> Y (b,T,H*P), final_state (b,H,P,N)."""
    return _SSD.apply(xbc.contiguous(), dt.contiguous(), A.contiguous(),
                      None if D is None else D.contiguous(), H, P, G, N, Q)
