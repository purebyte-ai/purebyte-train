"""Blocks of the residual stream. Public registry names: `ssm_v2`, `attention` (spec/FORMAT.md §9).

Every block has the same contract: `forward(u, state=None, mask=None) -> (u + f(u), new_state)`, with `mask` (B,T)
bool and the padding at the END of each window. Parameter names are part of the contract: checkpoints load by
them, and the exporter maps each name to its spec tensor (`export/pbformat.py`).

Bit-identity: parameters are created in a FIXED order with fixed initializers, so the same seed gives the same weights
(training/tests/test_golden_training.py locks it). Do not reorder attributes.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers.act import softplus
from .layers.conv import CausalConv1d
from .layers.norm import GatedRMSNorm, RMSNorm
from .layers.ssd import ssd_chunked
from .layers.ternary import TernaryLinear


def flip_valid(t: torch.Tensor, mask=None) -> torch.Tensor:
    """Reverses ONLY the valid positions of each sequence (padding stays at the end and never enters the state)."""
    if mask is None:
        return torch.flip(t, dims=[1])
    B, T = mask.shape
    lens = mask.sum(1, keepdim=True)                                  # (B,1)
    ar = torch.arange(T, device=t.device).unsqueeze(0)                # (1,T)
    idx = torch.where(ar < lens, lens - 1 - ar, ar).clamp(0, T - 1)   # (B,T)
    view = (B, T) + (1,) * (t.dim() - 2)
    return torch.gather(t, 1, idx.view(view).expand_as(t))


def shift_right(y: torch.Tensor) -> torch.Tensor:
    """y[t] <- y[t-1], zero at t=0: each direction contributes only the STRICTLY preceding (or following) context."""
    return F.pad(y, (0, 0) * (y.dim() - 2) + (1, 0))[:, :-1]


class SSMv2Block(nn.Module):
    """`ssm_v2`: Mamba-2 style selective state space block (spec/FORMAT.md §9.1).

    norm -> in_proj (ternary) -> z | xBC ; dt = softplus(dt_proj(xn) + dt_bias) ; conv1d + SiLU ; SSD scan (+ D x) ;
    y * silu(z) ; grouped RMSNorm ; out_proj (ternary) ; residual add. Directions: `causal`, `bimamba` (two full passes,
    shared weights), `hydra` (projections once, two scans, shifted, D x once).
    """
    TYPE = "ssm_v2"

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        d, di, H, N, G = cfg.d_model, cfg.d_inner, cfg.n_heads, cfg.d_state, cfg.n_groups
        self.norm = RMSNorm(d)
        self.in_proj = TernaryLinear(d, cfg.proj_out, cfg.ternary_group, cfg.ternary, thr=cfg.tern_thr)
        self.dt_proj = nn.Linear(d, H, bias=False)                 # float
        # dt_proj MUST be born small: with nn.Linear's default init its contribution dominates dt_bias and the designed
        # horizons [0.001, 0.1] became [0.0002, 0.41] (max half-life 693 -> 4049 bytes).
        nn.init.normal_(self.dt_proj.weight, std=0.1 * d ** -0.5)
        self.conv = CausalConv1d(di + 2 * G * N, cfg.conv_width) if cfg.conv_width else None
        self.gated_norm = GatedRMSNorm(di, G)
        self.out_proj = TernaryLinear(di, d, cfg.ternary_group, cfg.ternary, thr=cfg.tern_thr)
        # scaled out_proj instead of a learnable residual scale
        nn.init.normal_(self.out_proj.weight, std=(di ** -0.5) / math.sqrt(2 * cfg.n_layers))
        # A and dt log-spaced and PAIRED ascending: A*dt covers A_min*dt_min .. A_max*dt_max (0.4 to 693 bytes of
        # half-life at step 0). The canonical random-dt init leaves every head local at byte level.
        A = torch.exp(torch.linspace(math.log(cfg.A_min), math.log(cfg.A_max), H))
        dt = torch.exp(torch.linspace(math.log(cfg.dt_min), math.log(cfg.dt_max), H))
        dt = dt.clamp(min=cfg.dt_floor)
        self.A_log = nn.Parameter(torch.log(A))
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))   # softplus^-1
        self.D = nn.Parameter(torch.ones(H))

    def _project(self, u_norm):
        """in_proj + dt_proj: computed ONCE for both directions in hydra (95 % of the block's cost)."""
        proj = self.in_proj(u_norm)
        z, xBC = proj[..., :self.cfg.d_inner], proj[..., self.cfg.d_inner:]
        dt = softplus(self.dt_proj(u_norm).float() + self.dt_bias.float())
        return z, xBC, dt

    def _scan(self, xBC, dt, state, mask, reverse=False, include_D=True):
        """conv1d + SiLU + scan over one direction -> (Y (b,T,di), x, state)."""
        cfg = self.cfg
        b, T, _ = xBC.shape
        di, H, P, N, G = cfg.d_inner, cfg.n_heads, cfg.d_head, cfg.d_state, cfg.n_groups
        ssm_state, conv_state = (None, None) if state is None else state
        if reverse:
            xBC = flip_valid(xBC, mask)
            dt = flip_valid(dt, mask)
        if self.conv is not None:
            xBC, conv_state = self.conv(xBC, conv_state)
        xBC = F.silu(xBC)
        x = xBC[..., :di].reshape(b, T, H, P)
        Bm = xBC[..., di:di + G * N].reshape(b, T, G, N)
        Cm = xBC[..., di + G * N:].reshape(b, T, G, N)
        A = -torch.exp(self.A_log.float())
        Y, ssm_state = ssd_chunked(x.float(), dt, A, Bm.float(), Cm.float(),
                                   self.D.float() if include_D else None,
                                   chunk_size=cfg.chunk_size, initial_state=ssm_state, mask=mask)
        return Y.reshape(b, T, di).to(xBC.dtype), x, (ssm_state, conv_state)

    def forward(self, u, state=None, mask=None):
        un = self.norm(u)
        if self.cfg.direction == "causal":
            z, xBC, dt = self._project(un)
            Y, _, state = self._scan(xBC, dt, state, mask)
        elif self.cfg.direction == "bimamba":
            z, xBC, dt = self._project(un)
            Yf, _, state = self._scan(xBC, dt, state, mask)
            z2, xBC2, dt2 = self._project(un)
            Yb, _, _ = self._scan(xBC2, dt2, None, mask, reverse=True)
            Y = Yf + flip_valid(Yb, mask)
        else:                                                     # hydra: y = shift(SS(x)) + flip(shift(SS(flip x))) + D x
            z, xBC, dt = self._project(un)
            Yf, xf, state = self._scan(xBC, dt, state, mask, include_D=False)
            Yb, _, _ = self._scan(xBC, dt, None, mask, reverse=True, include_D=False)
            Yb = flip_valid(shift_right(Yb), mask)
            Y = shift_right(Yf) + Yb + (self.D.float().repeat_interleave(self.cfg.d_head)
                                        * xf.reshape(*xf.shape[:2], -1).float()).to(Yf.dtype)
        y = self.gated_norm(Y * F.silu(z))
        return u + self.out_proj(y), state


class AttentionBlock(nn.Module):
    """`attention`: causal softmax self-attention, RoPE (rotate-half), heads of 64, ternary projections
    (spec/FORMAT.md §9.2). Whole windows only: no incremental state."""
    TYPE = "attention"
    HEAD_DIM = 64

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.d_model
        self.cfg, self.H = cfg, d // self.HEAD_DIM
        self.norm = RMSNorm(d)
        self.qkv = TernaryLinear(d, 3 * d, cfg.ternary_group, cfg.ternary, thr=cfg.tern_thr)
        self.out = TernaryLinear(d, d, cfg.ternary_group, cfg.ternary, thr=cfg.tern_thr)
        nn.init.normal_(self.out.weight, std=(d ** -0.5) / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def rope(x):
        T, h = x.shape[2], x.shape[3] // 2
        f = torch.exp(torch.arange(h, device=x.device, dtype=torch.float32) * (-math.log(10000.0) / h))
        a = torch.arange(T, device=x.device, dtype=torch.float32)[:, None] * f[None]
        c, s_ = a.cos(), a.sin()
        x1, x2 = x[..., :h], x[..., h:]
        return torch.cat([x1 * c - x2 * s_, x1 * s_ + x2 * c], -1)

    def forward(self, u, state=None, mask=None):
        if state is not None:
            raise NotImplementedError("the attention block has no incremental state: whole windows only")
        B, T, d = u.shape
        q, k, v = self.qkv(self.norm(u)).view(B, T, 3, self.H, self.HEAD_DIM).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(self.rope(q), self.rope(k), v, is_causal=True)
        return u + self.out(y.transpose(1, 2).reshape(B, T, d)), None
