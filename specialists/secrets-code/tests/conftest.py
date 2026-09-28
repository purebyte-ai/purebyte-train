"""Test environment of the secrets-code specialist: CPU only, two threads, idle priority, no network, no GPU.

    python -m pytest specialists/secrets-code/tests -q          (from the repository root)

The generator's own locks (its data digests, the factory smoke run) are in training/tests; here: what a user runs
first on a fresh checkout (data/build_smoke_data.py and the smoke recipes), the refusal of made-up stand-ins outside a
smoke run, and how the catalogue rules the generator validates with depend on the Python version.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SPECIALIST = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SPECIALIST))
for p in (os.path.join(REPO, "training"), SPECIALIST):
    if p not in sys.path:
        sys.path.insert(0, p)

from pbtrain import runtime  # noqa: E402

runtime.setup(threads=2, lowprio=True, gpu=False)

import torch  # noqa: E402

torch.set_num_threads(2)
