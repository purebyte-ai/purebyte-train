"""Ternary projections: the single quantization (spec/FORMAT.md §7, `pb_tern2`).

Threshold and scale are the SAME quantity: the per-group absmean, not learnable, recomputed on every forward. The
straight-through estimator is the identity. `thr` multiplies the zeroing threshold (1.0 = `round`, ~31 % zeros on a
Gaussian); the exported scale never changes, so any `thr` exports unchanged.

Adapters (frozen body): `gain` multiplies every group scale (born 1; exported fused as scale x gain, the engine does
not change); `lora` is the Python-only control (born B = 0). Both are registered EMPTY: a layer without them has no
extra parameter, key or RNG draw, so it is bit-identical to the layer before adapters existed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

_EPS = 1e-8


def ternary_absmean(w: torch.Tensor, group: int, thr: float = 1.0):
    """w (out, in) -> (w_q, scale). w_q takes values in {-s, 0, +s} per group of `group` inputs.

    `in % group != 0` raises: a silent per-row fallback once mixed two granularities inside one model (the old
    `medium` preset, 576 % 128 = 64) and confounded model size with quantization.
    """
    out, inp = w.shape
    if group > 0 and inp % group:
        raise ValueError(
            f"a tensor with input size {inp} cannot be grouped {group} by {group}. Choose a `d_model` that is a "
            f"multiple of the group or a group that divides it: otherwise some layers would quantize per group and "
            f"others per row.")
    g = group if group > 0 else inp
    wg = w.view(out, inp // g, g)
    scale = wg.abs().mean(dim=-1, keepdim=True).clamp_min(_EPS).detach()
    if thr == 1.0:
        wq = (wg / scale).round().clamp_(-1, 1) * scale
    else:
        wq = torch.where(wg.abs() < (0.5 * thr) * scale, torch.zeros_like(wg), torch.sign(wg)) * scale
    return wq.view(out, inp), scale.view(out, inp // g)


class _STEIdentity(torch.autograd.Function):
    @staticmethod
    def forward(ctx, w, group, thr=1.0):
        return ternary_absmean(w, group, thr)[0]

    @staticmethod
    def backward(ctx, grad):
        return grad, None, None


class TernaryLinear(nn.Module):
    """Bias-free linear layer with ternary weights. `quantize=False` gives the float control."""

    def __init__(self, in_f: int, out_f: int, group: int = 128, quantize: bool = True, thr: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_f, in_f))
        nn.init.normal_(self.weight, std=in_f ** -0.5)
        self.group, self.quantize = group, quantize
        self.thr_mult = float(thr)
        self.register_parameter("gain", None)
        self.register_parameter("lora_A", None)
        self.register_parameter("lora_B", None)

    def enable_gain(self):
        """Adapter `scales`: one multiplicative gain per ternary group scale, born at 1 (= today's layer)."""
        if not self.quantize:
            raise ValueError("the `scales` adapter corrects the ternary group scales: a float layer has none")
        if self.gain is None:
            out, inp = self.weight.shape
            self.gain = nn.Parameter(torch.ones(out, inp // (self.group if self.group > 0 else inp),
                                                device=self.weight.device, dtype=self.weight.dtype))
        return self.gain

    def enable_lora(self, r: int, generator=None):
        """Adapter `lora` (the CONTROL, Python only): y = W x + B (A x); A ~ N(0, 1/in) from `generator` (never the
        global RNG), B = 0, so it is born = today's layer. It has no GGUF layout: the exporter refuses it."""
        if int(r) < 1:
            raise ValueError(f"lora_rank must be >= 1, got {r}")
        if self.lora_A is None:
            out, inp = self.weight.shape
            gen = generator if generator is not None else torch.Generator().manual_seed(0)
            a = torch.randn(int(r), inp, generator=gen) * inp ** -0.5
            self.lora_A = nn.Parameter(a.to(device=self.weight.device, dtype=self.weight.dtype))
            self.lora_B = nn.Parameter(torch.zeros(out, int(r), device=self.weight.device, dtype=self.weight.dtype))
        return self.lora_A, self.lora_B

    def adapt(self, w):
        """`w` (code x scale) times the gain of its group; `w` itself without a gain. The GPU backend calls this too."""
        g = self.gain
        if g is None:
            return w
        o, i = w.shape
        return (w.reshape(o, g.shape[1], i // g.shape[1]) * g.unsqueeze(-1)).reshape(o, i)

    def lora_delta(self, x, linear=F.linear):
        """B (A x). `linear` lets the GPU backend pass its deterministic GEMM without rewriting the maths."""
        return linear(linear(x, self.lora_A), self.lora_B)

    def _load_from_state_dict(self, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                              error_msgs):
        # a checkpoint of an adapted model carries `gain` / `lora_A` / `lora_B`: create them before the stock loader
        if prefix + "gain" in state_dict and self.gain is None:
            self.enable_gain()
        if prefix + "lora_A" in state_dict and self.lora_A is None:
            self.enable_lora(int(state_dict[prefix + "lora_A"].shape[0]))
        super()._load_from_state_dict(state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys,
                                      error_msgs)

    def effective_weight(self):
        """The weight the forward multiplies by: code x scale (x gain), or the float weight."""
        if not self.quantize:
            return self.weight
        return self.adapt(_STEIdentity.apply(self.weight, self.group, self.thr_mult))

    def forward(self, x):
        y = F.linear(x, self.effective_weight())
        if self.lora_A is not None:
            y = y + self.lora_delta(x)
        return y

    @torch.no_grad()
    def health(self) -> dict:
        """Trend metrics of the quantization (no targets): zero ratio, dead rows / groups, |w|/scale bands."""
        wq, scale = ternary_absmean(self.weight, self.group, self.thr_mult)
        g = self.weight.shape[1] // scale.shape[1]
        z = (wq == 0)
        ratio = self.weight.abs() / scale.repeat_interleave(g, dim=1)
        return dict(zero_ratio=z.float().mean().item(), dead_rows=z.all(dim=1).float().mean().item(),
                    dead_groups=z.view(z.shape[0], -1, g).all(dim=-1).float().mean().item(),
                    ratio_lt_half=(ratio < 0.5).float().mean().item(), ratio_gt_one=(ratio > 1.0).float().mean().item())
