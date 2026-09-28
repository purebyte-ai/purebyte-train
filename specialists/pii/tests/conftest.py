"""Test environment of the pii specialist: CPU only, two threads, idle priority, no network, no GPU, and a generated
$PUREBYTE_DATA (look-alike lists, corpus, public-format documents and credential formats; see pii_lookalike.py).

    python -m pytest specialists/pii/tests -q          (from the repository root)
"""
import os
import sys
import types

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
SPECIALIST = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SPECIALIST))
for p in (HERE, os.path.join(REPO, "training")):
    if p not in sys.path:
        sys.path.insert(0, p)

from pbtrain import runtime  # noqa: E402

runtime.setup(threads=2, lowprio=True, gpu=False)

import torch  # noqa: E402

torch.set_num_threads(2)

import pii_lookalike  # noqa: E402


@pytest.fixture(scope="session")
def pii_env(tmp_path_factory):
    """PUREBYTE_DATA / PUREBYTE_WORK in a temporary folder with generated resources; the specialist loaded."""
    root = tmp_path_factory.mktemp("piidata")
    data, work = str(root / "data"), str(root / "work")
    firsts, surnames = pii_lookalike.write_all(data)
    old = {k: os.environ.get(k) for k in ("PUREBYTE_DATA", "PUREBYTE_WORK")}
    os.environ["PUREBYTE_DATA"], os.environ["PUREBYTE_WORK"] = data, work
    from pbtrain.verticals import load_specialist
    spec = load_specialist("pii")
    yield types.SimpleNamespace(data=data, work=work, firsts=firsts, surnames=surnames, specialist=spec,
                                plugin=sys.modules["pbtrain_specialist_pii"],
                                corpus=os.path.join(data, "corpora", "smoke-context.bin"))
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
