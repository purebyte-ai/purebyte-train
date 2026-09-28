"""Helpers shared by the pii tests: the labels, small data blocks of the two verticals, and a build as the factory
does it."""
import copy
import os

SPECIALIST = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RECIPES = os.path.join(SPECIALIST, "recipes")
LABELS = ["EMAIL", "PHONE", "IBAN", "CREDIT_CARD", "IP", "DNI_NIE", "PASSPORT", "DATE_OF_BIRTH", "URL_CREDENTIALS",
          "SECRET", "PERSON_NAME", "ADDRESS"]
MASK_LABELS = ["PII"]
NO_SCENE = dict(code=0.0, data=0.0, prose=0.0, email=0.0, log=0.0, public=0.0)


def data_block(**over):
    """A small `data:` block of the typed vertical (128-byte windows, every scene on)."""
    d = dict(format="pii_spans", path="${PUREBYTE_DATA}/corpora/smoke-context.bin",
             split_policy=dict(unit="corpus_range", seed=0), n_train=64, n_eval=24, seq_len=128,
             control="shuffled_spans", empty_frac=0.18, lists="train",
             mix=dict(code=0.3, data=0.2, prose=0.1, email=0.1, log=0.1, public=0.2),
             public_train=dict(dir="${PUREBYTE_DATA}/pii/public_train"))
    d.update(over)
    return d


def mask_block(**over):
    """A small `data:` block of the label-agnostic vertical as the release recipe writes it: public documents only."""
    d = dict(format="pii_mask", path="${PUREBYTE_DATA}/corpora/smoke-context.bin",
             split_policy=dict(unit="corpus_range", seed=0), n_train=64, n_eval=24, seq_len=128,
             control="shuffled_spans", empty_frac=0.0, lists="train", lookalikes="text",
             mix=dict(NO_SCENE, public=1.0), public_train=dict(dir="${PUREBYTE_DATA}/pii/public_train"))
    d.update(over)
    return d


def forms_block(lookalikes="ignore", **over):
    """The structured-forms variant of the release recipe on small shapes: public documents plus generated records,
    logs and code (their share raised so that a small build holds all of them)."""
    return mask_block(mix=dict(NO_SCENE, code=0.2, data=0.3, log=0.2, public=0.3), lookalikes=lookalikes, **over)


def build(env, block):
    """(vertical, partitions, the watched data block) as the factory's data stage builds them."""
    from pbtrain.data.corpus import ByteCorpus
    from pbtrain.factory.spec import Watched, fracs
    from pbtrain.verticals import get
    w = Watched(copy.deepcopy(block))
    v = get(block["format"])
    return v, v.build(ByteCorpus(env.corpus), w, fracs(w)), w


def digest(t):
    """sha256 of every array of a build (train, select, test; X, Y, S in that order)."""
    import hashlib

    import numpy as np
    h = hashlib.sha256()
    for part in (t.train, t.select, t.test):
        for a in part:
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()
