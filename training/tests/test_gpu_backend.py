"""The Radeon backend against the reference code on the GPU: same weights, same batch -> the loss and EVERY gradient,
compared ELEMENT BY ELEMENT (|radeon - reference| <= ATOL x max|reference| + RTOL x |reference| for each tensor), for a
causal model (the patched forward) and a hydra model (the patched projections and scans, run through the reference
forward). GPU box only: set PBTRAIN_GPU_TESTS=1 (and hold the GPU: one GPU job at a time).

Each variant runs in its own process (the backend patches classes at import time and cannot be undone).
"""
import os
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)

_CHILD = r'''
import sys, json, torch, torch.nn.functional as F
sys.path.insert(0, sys.argv[1])
fast = sys.argv[2] == "1"
if fast:
    from pbtrain.gpu import enable; enable("radeon")
from pbtrain.model import ByteModel, ModelConfig, TagHead, ChoiceHead
torch.manual_seed(0)
cfg = ModelConfig.preset("nano", direction=sys.argv[4])
m, th, ch = ByteModel(cfg).cuda(), TagHead(cfg.d_model, 5).cuda(), ChoiceHead(cfg.d_model, 2, pooling="max").cuda()
g = torch.Generator().manual_seed(1)
ids = torch.randint(0, 256, (8, 512), generator=g).cuda()
yl = torch.randint(0, 5, (8 * 512,), generator=g).cuda(); yd = torch.randint(0, 2, (8,), generator=g).cuda()
h, _ = m(ids)
loss = F.cross_entropy(th(h).float().reshape(-1, 5), yl) + 0.5 * F.cross_entropy(ch(h).float(), yd)
loss.backward()
out = {"loss": torch.tensor(loss.item())}
for n, p in list(m.named_parameters()) + [("tag." + k, v) for k, v in th.named_parameters()]:
    out[n] = p.grad.detach().double().cpu()
torch.save(out, sys.argv[3])
'''

# per tensor: |radeon - reference| <= ATOL * max|reference| + RTOL * |reference| (float32 kernels, other summation order)
ATOL, RTOL = 1e-4, 1e-3


@pytest.mark.skipif(os.environ.get("PBTRAIN_GPU_TESTS") != "1", reason="GPU box only (PBTRAIN_GPU_TESTS=1)")
@pytest.mark.parametrize("direction", ["causal", "hydra"])
def test_radeon_backend_matches_the_reference(tmp_path, direction):
    import torch
    from pbtrain.runtime import gpu_lock
    res = {}
    with gpu_lock("test_gpu_backend"):
        for fast in (False, True):
            p = str(tmp_path / f"{direction}-{int(fast)}.pt")
            subprocess.run([sys.executable, "-c", _CHILD, TRAINING, str(int(fast)), p, direction], check=True,
                           env=dict(os.environ, HIP_VISIBLE_DEVICES="0", CUDA_VISIBLE_DEVICES="0"))
            res[fast] = torch.load(p, weights_only=True)
    ref, got = res[False], res[True]
    assert set(ref) == set(got)
    assert abs(float(ref["loss"]) - float(got["loss"])) < 1e-5
    worst = {}
    for k in ref:
        if k == "loss":
            continue
        a, b = ref[k], got[k]
        assert a.shape == b.shape, k
        tol = ATOL * float(a.abs().max()) + RTOL * a.abs()
        worst[k] = float(((b - a).abs() / tol.clamp_min(1e-30)).max())
        assert bool(((b - a).abs() <= tol).all()), f"{direction} {k}: max difference {float((b - a).abs().max())}"
    print(f"{direction}: the largest difference is {max(worst.values()):.3g} of its tolerance "
          f"({max(worst, key=worst.get)})")
