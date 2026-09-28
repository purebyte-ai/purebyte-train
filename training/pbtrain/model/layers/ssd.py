"""Mamba-2 SSD scan, chunked: the ONLY implementation (ssm_v2 step 5, spec/FORMAT.md §9.1; the engine runs the
equivalent recurrence position by position).

The recurrent mode is not separate code: it is this same function with T=1 and `initial_state` (a hand-written `_step`
once made two code paths diverge). Rewriting the contractions as two-operand einsums is
numerically identical (7e-15) and not faster. The backward memory scales with b*h*T*chunk: `chunk_size` is the lever
if the batch must grow.
"""
import torch
import torch.nn.functional as F

# Floor of the accumulated exponents: exp(-30) ~ 1e-13 is zero in practice and avoids denormals (without it the
# cumulative sum inside a chunk reached -416 and the CPU and MPS backwards diverged, gradient norm 5.9 vs 1494, measured
# on the Apple M3 Pro the model code was developed on). It is applied BEFORE the causal mask so that -inf stays an
# exact zero.
_EXP_FLOOR = -30.0


def _segsum(x: torch.Tensor) -> torch.Tensor:
    """(..., T) -> (..., T, T) with the cumulative sum of the segment (j, i], -inf if j > i."""
    T = x.size(-1)
    xx = x[..., None].expand(*x.shape, T)
    tri = torch.ones(T, T, dtype=torch.bool, device=x.device)
    xx = xx.masked_fill(~torch.tril(tri, -1), 0).cumsum(dim=-2).clamp(min=_EXP_FLOOR)
    return xx.masked_fill(~torch.tril(tri, 0), float("-inf"))


def ssd_chunked(x, dt, A, B, C, D=None, *, chunk_size, initial_state=None, mask=None):
    """
    x  (b,T,H,P)   dt (b,T,H)   A (H,)   B,C (b,T,G,N)   D (H,) or None
    initial_state (b,H,P,N) or None
    mask (b,T) bool: True = valid position. Padding does not advance the state.
    -> y (b,T,H,P), final_state (b,H,P,N)
    """
    b, T, H, P = x.shape
    G, N = B.shape[2], B.shape[3]
    Q = chunk_size

    if mask is not None:                      # padding: dt=0 -> identity transition
        dt = dt * mask[..., None].to(dt.dtype)

    pad = (-T) % Q
    if pad:                                   # padded with dt=0, never with data
        x = F.pad(x, (0, 0, 0, 0, 0, pad))
        dt = F.pad(dt, (0, 0, 0, pad))
        B = F.pad(B, (0, 0, 0, 0, 0, pad))
        C = F.pad(C, (0, 0, 0, 0, 0, pad))
    Tp = T + pad

    if G != H:                                # B/C shared per group
        rep = H // G
        B = B.repeat_interleave(rep, dim=2)
        C = C.repeat_interleave(rep, dim=2)

    nc = Tp // Q
    # The input enters scaled by dt: h' = exp(A*dt)*h + dt*B*x. Without this factor the scan is self-consistent but does
    # NOT implement the recurrence, and dt=0 on the padding no longer cancels the injection.
    xc = (x * dt.unsqueeze(-1)).view(b, nc, Q, H, P)
    Bc = B.view(b, nc, Q, H, N)
    Cc = C.view(b, nc, Q, H, N)
    Ac = (dt * A).view(b, nc, Q, H).permute(0, 3, 1, 2)          # (b,H,nc,Q)
    Acs = Ac.cumsum(-1)

    # 1. diagonal block (inside each chunk)
    L = torch.exp(_segsum(Ac))                                    # (b,H,nc,Q,Q)
    CB = torch.einsum("bclhn,bcshn->bhcls", Cc, Bc)
    Y_diag = torch.einsum("bhcls,bcshp->bclhp", CB * L, xc)

    # 2. state contributed by each chunk
    dec_s = torch.exp((Acs[..., -1:] - Acs).clamp(min=_EXP_FLOOR))  # (b,H,nc,Q)
    Bd = Bc * dec_s.permute(0, 2, 3, 1).unsqueeze(-1)
    states = torch.einsum("bclhn,bclhp->bchpn", Bd, xc)           # (b,nc,H,P,N)

    # 3. recurrence between chunks, starting from the initial state
    if initial_state is None:
        initial_state = states.new_zeros(b, H, P, N)
    states = torch.cat([initial_state.unsqueeze(1), states], dim=1)
    dec_c = torch.exp(_segsum(F.pad(Acs[..., -1], (1, 0))))       # (b,H,nc+1,nc+1)
    new = torch.einsum("bhzc,bchpn->bzhpn", dec_c, states)
    states, final_state = new[:, :-1], new[:, -1]

    # 4. contribution of the state to the output
    Cd = Cc * torch.exp(Acs.clamp(min=_EXP_FLOOR)).permute(0, 2, 3, 1).unsqueeze(-1)
    Y_off = torch.einsum("bclhn,bchpn->bclhp", Cd, states)

    y = (Y_diag + Y_off).reshape(b, Tp, H, P)[:, :T]
    if D is not None:
        y = y + D.view(1, 1, H, 1) * x[:, :T]
    return y, final_state
