"""The secrets generator is locked: on a generated corpus with made-up catalogs (smoke_data), the data blocks of the
secrets-code and secrets-bin recipes must produce the SAME arrays, bit for bit, as the generator the released models
were trained with (golden/secrets_data_fingerprints.json, recorded from it). Offline; NumPy's random streams are the
only platform dependency (the numpy version is recorded)."""
import hashlib
import json
import os

import numpy as np
import pytest

import smoke_data

GOLDEN = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden", "secrets_data_fingerprints.json")

# data blocks of the recipes (the generator switches only; sizes are small)
BLOCKS = {
    "code": dict(version="v4", extra_lines=5, web_negatives=True, shared_hosts=True, connection_config=0.08,
                 short_passwords=0.2, ordinary_short_values="separate", real_neighbors=True, reinforced_forms=0.05,
                 non_assignment_weight=35, nested_weight=8),
    "code_real_data_families": dict(version="v4", extra_lines=5, web_negatives=True, shared_hosts=True,
                                    connection_config=0.08, short_passwords=0.2, ordinary_short_values="separate",
                                    real_neighbors=True, reinforced_forms=0.05, non_assignment_weight=35,
                                    nested_weight=8, password_named_short=0.06, auth_header=0.04,
                                    url_short_password=0.03, unnamed_values=0.03, dotnet_resources=0.02),
    "binary": dict(version="v4", nul_separator=True, bare_value=True, long_hex=True, web_negatives=True,
                   extra_lines=3, real_neighbors="no_pem"),
}
SIZES = dict(n_train=300, n_eval=60, seq_len=512, split_policy=dict(unit="corpus_range", seed=0))


def arrays_sha(t):
    h = hashlib.sha256()
    for part in (t.train, t.select, t.test):
        for a in part:
            h.update(np.ascontiguousarray(a).astype(np.int64).tobytes())
    return h.hexdigest()


def build_public(block, data_dir):
    from pbtrain.data.corpus import ByteCorpus
    from pbtrain.factory.spec import Watched, fracs
    from pbtrain.verticals import get, load_specialist
    load_specialist("secrets-code")
    d = dict(SIZES, **BLOCKS[block])
    old = os.environ.get("PUREBYTE_DATA")
    os.environ["PUREBYTE_DATA"] = data_dir            # the vertical reads the catalogs of $PUREBYTE_DATA
    try:
        return get("secrets_synthetic").build(ByteCorpus(os.path.join(data_dir, "corpora", "smoke-context.bin")),
                                              Watched(d), fracs(d))
    finally:
        if old is None:
            os.environ.pop("PUREBYTE_DATA", None)
        else:
            os.environ["PUREBYTE_DATA"] = old


@pytest.fixture(scope="module")
def data_dir(tmp_path_factory):
    d = str(tmp_path_factory.mktemp("secrets_golden"))
    smoke_data.code_corpus(os.path.join(d, "corpora", "smoke-context.bin"))
    smoke_data.catalogs(os.path.join(d, "catalogs"))
    return d


@pytest.mark.parametrize("block", list(BLOCKS))
def test_generator_matches_the_golden(block, data_dir):
    with open(GOLDEN, encoding="utf-8") as f:
        g = json.load(f)
    if g["meta"]["numpy"].split(".")[0] != np.__version__.split(".")[0]:
        pytest.skip(f"golden recorded with numpy {g['meta']['numpy']}")
    assert arrays_sha(build_public(block, data_dir)) == g["cases"][block], f"{block}: the generator changed"


def test_generator_is_deterministic(data_dir):
    assert arrays_sha(build_public("binary", data_dir)) == arrays_sha(build_public("binary", data_dir))
