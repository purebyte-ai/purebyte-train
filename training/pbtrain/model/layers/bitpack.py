"""2-bit ternary packing (`pb_tern2`, spec/FORMAT.md §7): `00 = 0`, `01 = +1`, `10 = -1`, `11` invalid; four codes per
byte, code j in bits 2*(j mod 4); the padding codes of a row's last byte are `00`.

There is ONE packer, with a round-trip test against the unpacker the engine mirrors (a second, incompatible encoding,
-1 -> 00, 0 -> 01, +1 -> 10, once existed next to it: never add another).
"""
import numpy as np
import torch


def pack2(t: torch.Tensor) -> np.ndarray:
    """t (out, in) with values in {-1,0,+1} -> uint8 (out, ceil(in/4))."""
    a = t.detach().cpu().numpy()
    if not np.isin(a, (-1, 0, 1)).all():
        raise ValueError("pack2 expects values in {-1,0,+1}")
    out, inp = a.shape
    pad = (-inp) % 4
    if pad:
        a = np.pad(a, ((0, 0), (0, pad)))
    codes = np.select([a == 0, a == 1, a == -1], [0, 1, 2]).astype(np.uint8)
    codes = codes.reshape(out, -1, 4)
    return (codes[:, :, 0] | (codes[:, :, 1] << 2) | (codes[:, :, 2] << 4) | (codes[:, :, 3] << 6)).astype(np.uint8)


def unpack2(p: np.ndarray, in_dim: int) -> np.ndarray:
    """Inverse of pack2 (what the engine does)."""
    out = p.shape[0]
    c = np.stack([(p >> s) & 0b11 for s in (0, 2, 4, 6)], axis=-1).reshape(out, -1)
    v = np.select([c == 0, c == 1, c == 2], [0, 1, -1], default=0).astype(np.int8)
    return v[:, :in_dim]


def pack_ternary_linear(weight: torch.Tensor, group: int, thr: float = 1.0):
    """Quantizes and packs a matrix -> (uint8 codes, float32 group scales). `thr` must be the layer's own, so the pack
    carries the SAME zeros as training."""
    from .ternary import ternary_absmean
    wq, scale = ternary_absmean(weight, group, thr)
    g = weight.shape[1] // scale.shape[1]
    signs = torch.round(wq / scale.repeat_interleave(g, dim=1)).clamp(-1, 1)
    return pack2(signs), scale.detach().cpu().numpy().astype(np.float32)
