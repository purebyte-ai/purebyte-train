"""softplus with our own implementation (used for dt in the ssm_v2 block).

Measured while the model code was developed on an Apple M3 Pro (MPS, PyTorch 2.12.1; training/docs/gpu.md says what
has and has not been run on Apple silicon): the backward of `F.softplus` returned wrong gradients for a non-contiguous
incoming gradient (relative error 211 % against sigmoid(x)); dt is reordered before the scan, so its gradient is never
contiguous. log1p(exp(z)) is identical in the forward over -30..60 and correct on every device. Kept for bit-identity.
"""
import torch


def softplus(z: torch.Tensor, threshold: float = 20.0) -> torch.Tensor:
    return torch.where(z > threshold, z, torch.log1p(torch.exp(z.clamp(max=threshold))))
