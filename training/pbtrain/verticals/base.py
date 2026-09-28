"""The vertical contract: the ONLY thing that changes from one specialist to another.

The core (model, trainer, factory, exporter) does not know what a task is about. It consumes windows of bytes and, for
every byte, whether (and as which entity type) it belongs to a span. A VERTICAL manufactures those windows and knows its
domain: the classic rival to beat, the guards on its own data, what goes into the manifest, how its model is declared
to the runtime. It is a plugin: a module in `specialists/<name>/plugin.py` that subclasses `Vertical` and decorates it
with `@register`. The factory finds it by the `data.format` of the task.

    @register
    class MyVertical(Vertical):
        data_format = "my_format"          # the task's `data.format`
        needs_corpus = True                 # False when every byte is generated
        overlap_guard = "chunks"            # "chunks" when windows are cut from real bytes, else "windows"
        entities = ("secret",)              # entity types of the tag head (k = len(entities))

        def build(self, corpus, cfgd, fracs):          -> Partitions (train, select, test), each (X, Y, S)
        def guards(self, cfgd, t):                      the vertical's own checks on what it built (raise ValueError)
        def manifest(self, cfgd, t):                    what HAPPENED (effective options, counts) -> done.json
        def span_rival(self, select, test):            the classic rival, tuned on select, scored on test
        def export_meta(self, spec):                   profile / window policy the GGUF declares, by default

What the exported GGUF declares (profile, labels, window policy, n-gram precision, license) comes from the specialist's
`specialist.yaml` (`export:`); `export_meta` supplies the vertical's defaults for the profile and the window, used by
`factory.stages.export_checkpoint` only where specialist.yaml says nothing.

Arrays: X uint8 [n, T] window bytes; S uint8 [n, T] span mask (0 outside; 1 for one entity type, 1 + e for k > 1;
255 = ignore); Y uint8 [n] = the window holds a span (IGNORE bytes do not count).

`overlap_guard` chooses the between-split check the `data` stage ENFORCES (pbtrain/data/leakage.py): "windows" (whole
windows, a 0.1 % tolerance) or "chunks" (120-byte chunks of the first 300 windows, zero tolerance).
"""
from dataclasses import dataclass

import numpy as np

from ..model.decode import IGNORE, window_class

REGISTRY = {}


@dataclass
class Partitions:
    train: tuple
    select: tuple
    test: tuple

    def check(self, seq_len, n_entities=1):
        """The core contract on shapes, dtypes, mask values and Y == «holds a span»."""
        kmax = 1 if n_entities == 1 else n_entities
        for name in ("train", "select", "test"):
            X, Y, S = getattr(self, name)
            if X.dtype != np.uint8 or X.ndim != 2 or X.shape[1] != seq_len:
                raise ValueError(f"{name}: X must be uint8 [n, {seq_len}], it is {X.dtype} {X.shape}")
            if S.shape != X.shape or Y.shape != (len(X),):
                raise ValueError(f"{name}: inconsistent shapes X{X.shape} Y{Y.shape} S{S.shape}")
            bad = (S > kmax) & (S != IGNORE)
            if bad.any():
                raise ValueError(f"{name}: span mask values {sorted(set(np.unique(S[bad]).tolist()))} with "
                                 f"{n_entities} entity type(s)")
            if not np.array_equal(np.asarray(Y).astype(bool), window_class(S).astype(bool)):
                raise ValueError(f"{name}: Y does not match «the window holds a span»")
        return self


class Vertical:
    data_format = None
    needs_corpus = True
    overlap_guard = "windows"
    entities = ("span",)
    checks_contract = True        # a vertical whose Y means something else sets False and checks its own contract

    @property
    def n_entities(self):
        return len(self.entities)

    def build(self, corpus, cfgd, fracs):
        raise NotImplementedError

    def guards(self, cfgd, t):
        """Raise ValueError when what was built is not what the recipe asked for."""

    def manifest(self, cfgd, t=None):
        return {}

    def span_rival(self, select, test):
        """The classic rival of this task on span-F1, tuned on `select`, reported on `test`: a dict with at least
        span_f1, precision, recall and `name`. None = there is no rival (the gate then cannot pass: deliberate)."""
        return None

    def export_meta(self, spec):
        """The vertical's defaults for what the exported GGUF declares: {"profile": ..., "window": {"size", "stride",
        "min"}}. specialist.yaml's `export:` overrides each of them."""
        return {}


def register(cls):
    if not cls.data_format:
        raise ValueError(f"{cls.__name__}: empty data_format")
    old = REGISTRY.get(cls.data_format)
    if old is not None and (old.__module__, old.__qualname__) != (cls.__module__, cls.__qualname__):
        raise ValueError(f"duplicate vertical format {cls.data_format!r}: {old.__module__} and {cls.__module__}")
    REGISTRY[cls.data_format] = cls
    return cls


def get(data_format):
    cls = REGISTRY.get(data_format)
    if cls is None:
        raise KeyError(f"no vertical registered for data.format {data_format!r}; registered: {sorted(REGISTRY)}. "
                       f"Is the task's `specialist:` right?")
    return cls()
