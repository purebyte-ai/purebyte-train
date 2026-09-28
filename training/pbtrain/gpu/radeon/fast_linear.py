"""Bias-free linear whose BACKWARD uses the GEMM formulations rocBLAS runs fast on gfx1030 -- and ONLY
formulations measured deterministic run to run (checked with a GEMM checksum benchmark).

Measured, 16x512 rows, fp32 (ms):            autograd     here
    256->1280   dX  g @ W                      0.62        0.43   (W.t() @ g.t()).t()
                dW  g.t() @ x                  0.55        0.33   bmm per sample, then sum over the batch
    512->256    dX                             0.25        0.25   (autograd's own: the transposed one is slower here)
                dW                             0.33        0.16
rocBLAS has a ~0.3 ms floor whenever the inner dimension is the 8192 rows; the per-sample bmm never
shows it that dimension. REJECTED: dW = (x.t() @ g).t() is as fast but NOT deterministic (different
bits on every run for 256->1280). The choice of formulation is a STATIC rule on the shape, never an
autotune: a timing-based choice could differ between processes and seeds would stop being
reproducible. Forward is F.linear itself. Same maths, different summation order than autograd.
"""
import torch


class _Linear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w):
        ctx.save_for_backward(x, w)
        return torch.nn.functional.linear(x, w)

    @staticmethod
    def backward(ctx, g):
        x, w = ctx.saved_tensors
        O, I = w.shape
        g = g.contiguous()
        dx = dw = None
        if ctx.needs_input_grad[0]:
            if O > 2 * I:
                dx = (w.t() @ g.view(-1, O).t()).t().contiguous().view(x.shape)
            else:
                dx = (g.view(-1, O) @ w).view(x.shape)
        if ctx.needs_input_grad[1]:
            dw = torch.bmm(g.transpose(1, 2), x).sum(0)
        return dx, dw


def linear(x, w):
    if x.is_cuda and x.dtype == torch.float32 and w.dtype == torch.float32 and x.dim() == 3 and x.is_contiguous():
        return _Linear.apply(x, w)
    return torch.nn.functional.linear(x, w)
