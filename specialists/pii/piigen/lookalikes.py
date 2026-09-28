"""Look-alikes as IGNORE: the generated windows of the label-agnostic `pii_mask` vertical, when a recipe mixes the
generator's scenes (records, logs, code) into the public documents and asks for `data.lookalikes: ignore`.

The generator plants look-alikes as negatives: values that have the shape of personal data but play another role
(order ids, IMEIs, company tax ids, creation dates, postcodes, lone cities, countries, usernames, company names, URLs).
Under the typed policy of `pii_spans` they are plain text (O). The label-agnostic policy follows the public datasets,
which call most of those categories personal data. Teaching them as O would contradict the public documents; teaching
them as PII would contradict the typed policy the generator was written for. So they are IGNORE (label -100): taught
neither way.

`GeneratorLookalikes` is `generator.Generator` with the SAME random draws in the same order: for a given seed its windows
are byte for byte the plain generator's, every labelled byte is the same, and only bytes that are O in a GENERATED
window may become IGNORE. Windows of the `public` scene are never touched. `tests/test_pii_mask.py` checks all of it.

How the look-alikes are found, per generated document:
  record fields   the values of the look-alike kinds of `forms.FIELDS` (KINDS below; `status` only when it is a gender,
                  M or F), recorded while the document is written;
  inline          INLINE patterns anywhere in the document outside every labelled span: ISO dates and timestamps,
                  access-log and syslog dates, request ids, the user of an access-log or sshd line, URLs of any scheme
                  (the credential inside a DSN stays labelled) -- in the real code of the `code` scene too.
Left O (personal data in no source): amounts, hashes, versions, hosts, ports, other statuses, user agents, paths,
log levels.

The recording works by swapping `forms.Doc` and the look-alike makers of `forms.FIELDS` for the duration of ONE
generated document (`_marking`, always restored), so nothing else in the process sees the swap and the existing
modules are unchanged.
"""
import re
from contextlib import contextmanager

import numpy as np

from . import forms as FM
from . import generator as GN
from . import values as VL
from .generator import IGNORE_T, LANG_ES
from .types import IGNORE, TYPES, paint, window_class

KINDS = ("id", "sku", "imei", "vat", "bic", "mac", "created", "expiry", "zip", "city", "country", "username", "company",
         "url", "age", "status")
GENDER = ("M", "F")
INLINE = tuple(re.compile(p) for p in (
    rb"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}:\d{2}(?:\.\d{3})?(?:Z|[+-]\d{2}:\d{2})?)?",   # ISO dates and timestamps
    rb"\d{2}/[A-Z][a-z]{2}/\d{4}:\d{2}:\d{2}:\d{2} [+-]\d{4}",                           # access-log dates
    rb"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec) [ \d]\d \d{2}:\d{2}:\d{2}",   # syslog dates
    rb"(?<=request_id=)[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",      # request ids
    rb"(?<= for )[A-Za-z0-9_.][A-Za-z0-9_.\-]*(?= from )",                               # sshd: the user
    rb"(?<= - )[A-Za-z0-9_.][A-Za-z0-9_.\-]*(?= \[\d{2}/)",                              # access log: the user (not "-")
    rb"\b[A-Za-z][A-Za-z0-9+.\-]*://[^\s\"'<>`)\]}]+",                                  # URLs of any scheme (DSNs too)
))
POLICIES = ("text", "ignore")


class _LVal(FM.Val):
    """A look-alike value: plain text under the typed policy, IGNORE here. `core`: its characters inside the (quoted)
    text."""
    __slots__ = ("core",)

    def __init__(self, text, core=None):
        super().__init__(text, None, None)
        self.core = core or (0, len(text))

    def quoted(self, q):
        return _LVal(q + self.text + q, (self.core[0] + len(q), self.core[1] + len(q)))


class _LDoc(FM.Doc):
    """forms.Doc that also records the byte ranges of the look-alike values written into it. (Inside `_marking`,
    `FM.Doc` IS this class: the parent is reached with super(), never through FM.Doc.)"""
    made = []

    def __init__(self):
        super().__init__()
        self.look = []
        _LDoc.made.append(self)

    def value(self, v):
        if isinstance(v, _LVal) and v.text:
            a = self.n + len(v.text[:v.core[0]].encode("utf-8"))
            self.look.append((a, a + len(v.text[v.core[0]:v.core[1]].encode("utf-8"))))
        return super().value(v)


def _wrap(kind, mk):
    def f(r, L, g):                     # the same maker, the same draws: only the returned value is marked
        v = mk(r, L, g)
        if v.typ is not None or (kind == "status" and v.text not in GENDER):
            return v
        return _LVal(v.text)
    return f


@contextmanager
def _marking():
    """forms.Doc -> _LDoc and the look-alike makers of forms.FIELDS wrapped, for ONE generated document; always
    restored."""
    saved_doc, saved = FM.Doc, {k: FM.FIELDS[k] for k in KINDS}
    del _LDoc.made[:]
    FM.Doc = _LDoc
    for k in KINDS:
        es, en, mk = saved[k]
        FM.FIELDS[k] = (es, en, _wrap(k, mk))
    try:
        yield _LDoc.made
    finally:
        FM.Doc = saved_doc
        FM.FIELDS.update(saved)
        del _LDoc.made[:]


def crop_at(data, spans, seq_len, rng, anchor=None):
    """`generator._crop`, line for line, that also returns where the window starts (the look-alike ranges are clipped
    to it). Kept equal to `_crop` by `tests/test_pii_mask.py`."""
    n = len(data)
    if n <= seq_len:
        data = data + b"\n" * (seq_len - n)
        return data, [(a, min(b, seq_len), t) for a, b, t in spans if a < seq_len], 0
    lo_s, hi_s = 0, n - seq_len
    if anchor is not None:
        a0, b0 = anchor
        if b0 - a0 <= seq_len - 16:
            lo_s, hi_s = max(0, b0 - seq_len + 8), min(hi_s, max(0, a0 - 8))
        else:
            lo_s, hi_s = a0, min(hi_s, b0 - seq_len)
        lo_s, hi_s = min(lo_s, hi_s), max(lo_s, hi_s)
    start = int(rng.integers(lo_s, hi_s + 1))
    if VL.chance(rng, 0.7):
        nl = data.rfind(b"\n", max(0, start - 200), start)
        if nl >= 0 and nl + 1 >= lo_s:
            start = nl + 1
    for a, b, t in spans:
        if a < start < b:
            start = b
    start = min(start, n - seq_len)
    for a, b, t in spans:
        if a < start < b:
            start = a if a >= 0 else start
    start = max(start, 0)
    win = data[start:start + seq_len]
    out = [(max(a, start) - start, min(b, start + seq_len) - start, t) for a, b, t in spans
           if b > start and a < start + seq_len]
    return win, [s for s in out if s[1] > s[0]], start


class GeneratorLookalikes(GN.Generator):
    """generator.Generator whose generated documents also carry IGNORE ranges for the look-alikes (module docstring).
    Label-agnostic only (`collapse=True`): under the typed policy the look-alikes are the negatives and stay O."""

    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        if not self.collapse:
            raise ValueError("look-alikes as IGNORE belong to the label-agnostic policy (collapse)")
        self._look = []
        self.stats.update(lookalike_ignored_bytes=0, lookalike_ignored_windows=0)

    def _doc_scene(self, rng, scene, lang, positive, target=None):
        with _marking() as made:
            data, spans = super()._doc_scene(rng, scene, lang, positive, target)
            self._look = [x for d in made for x in d.look]
        return data, spans

    def _code_scene(self, rng, lang, positive):
        with _marking() as made:
            data, spans, (c0, c1) = super()._code_scene(rng, lang, positive)
            self._look = [(a + c0, b + c0) for d in made for a, b in d.look]
        return data, spans, (c0, c1)

    def _ignore(self, data, spans):
        """[(a, b)] of one generated document: its look-alike values and the inline patterns, minus every labelled
        byte."""
        n = len(data)
        m = np.zeros(n + 1, dtype=bool)
        for a, b in self._look:
            m[a:b] = True
        for rx in INLINE:
            for x in rx.finditer(data):
                m[x.start():x.end()] = True
        for a, b, _ in spans:
            m[max(0, a):min(n, b)] = False
        m[n] = False
        e = np.flatnonzero(np.diff(np.concatenate([[False], m]).astype(np.int8)))
        return [(int(e[i]), int(e[i + 1])) for i in range(0, len(e) - 1, 2)]

    def window(self, rng):
        """generator.Generator.window, the same draws in the same order; the IGNORE ranges are painted after the
        labelled spans and only over bytes that are O, so X and every labelled byte are Generator's."""
        scene = self.scenes[min(int(np.searchsorted(self.cum, rng.random(), side="right")), len(self.scenes) - 1)]
        lang = "es" if VL.chance(rng, LANG_ES) else "en"
        positive = not VL.chance(rng, self.empty_frac)
        anchor, self._look = None, []
        if scene == "code":
            data, spans, anchor = self._code_scene(rng, lang, positive)
        elif scene == "public":
            data, spans = self.public.sample(rng, self.seq_len + int(rng.integers(64, 600)))
        else:
            data, spans = self._doc_scene(rng, scene, lang, positive)
        spans = [(a, b, 0 if t != IGNORE_T else t) for a, b, t in spans]           # collapse: one type, PII
        ign = [] if scene == "public" else self._ignore(data, spans)
        win, spans, start = crop_at(data, spans, self.seq_len, rng, anchor)
        S = np.zeros(self.seq_len, dtype=np.uint8)
        spans = GN._separate(spans)
        paint(S, spans)
        got = 0
        for a, b in ign:
            a, b = max(a - start, 0), min(b - start, self.seq_len)
            if a < b:
                seg = S[a:b]
                free = seg == 0
                got += int(free.sum())
                seg[free] = IGNORE
        self.stats["lookalike_ignored_bytes"] += got
        self.stats["lookalike_ignored_windows"] += int(got > 0)
        self.stats["scenes"][scene] += 1
        if not window_class(S):
            self.stats["empty_windows"] += 1
        for _, _, t in spans:
            if t == IGNORE_T:
                self.stats["ignored_spans"] += 1
            else:
                self.stats["types"]["PII" if self.collapse else TYPES[t]] += 1
        return np.frombuffer(win, dtype=np.uint8), S
