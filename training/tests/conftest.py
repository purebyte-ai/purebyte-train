"""Test environment of the training stack: CPU only, one thread, idle priority, no network. CUDA and ROCm devices are
hidden from the session; Apple's MPS cannot be hidden by an environment variable, and no test places a tensor on a GPU
unless PBTRAIN_GPU_TESTS=1 (then only in child processes).

Run from the repository root:  python -m pytest training/tests -q
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TRAINING = os.path.dirname(HERE)
REPO = os.path.dirname(TRAINING)
for p in (HERE, TRAINING, os.path.join(REPO, "specialists")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pbtrain import runtime  # noqa: E402

runtime.setup(threads=1, lowprio=True, gpu=False)

import torch  # noqa: E402

torch.set_num_threads(1)
