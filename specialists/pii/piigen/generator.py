"""Windows of bytes with typed spans: REAL contexts + planted synthetic personal data + look-alikes.

Scenes, mixed by the task's `mix`:
  code    a REAL window of the code corpus (a byte range disjoint per split). Its own personal data is RE-PLANTED first:
          every e-mail, IP, URL credential and (with context) phone number the rules find, and every person in an
          author context (copyright lines, `@author`, sign-off trailers, package metadata) whose first word is in the
          given-name lists, is replaced by a synthetic value of the same type AND labelled -- a real slot with a fake
          value. What neither matches (other names, an address the rules miss) stays as it is, unlabelled.
          A window holding something that looks like a live credential is skipped. Then a block of 1-3 snippets is
          inserted at a line boundary, in the language of the surrounding code.
  data    records (customer, employee, order, payment, account, server, traveller, patient, device, supplier) rendered
          in many formats.
  prose   paragraphs of real sentences (person slots filled with synthetic names) and sentences carrying data.
  email   a whole e-mail: headers, greeting, body, signature block.
  log     access logs, application logs with data in the message, JSON logs, sshd.
  public  labelled documents of the TRAIN splits of public PII datasets (`PublicDocs`); labels outside the taxonomy
          become IGNORE.
A fraction `empty_frac` of the windows is built with NOTHING to redact (look-alikes, empty or masked fields, roles
instead of people): most real windows hold no personal data, and a generator where every window has some teaches the
model to always find something.

Every window is exactly `seq_len` bytes; a crop never starts inside a span (a span may be cut at the END, as at
inference) nor before the first byte (a negative random start, possible when a planted block is longer than the
window minus 16 bytes, is clamped to 0). Deterministic for the same corpus, lists, documents, catalogues and seed.
"""
import re
import numpy as np
from . import forms as FM
from . import rules as RU
from . import values as VL
from .names import get_lists
from .types import IDX, IGNORE, TYPES, paint, window_class
DEFAULT_MIX = {'code': 0.35, 'data': 0.25, 'prose': 0.13, 'email': 0.1, 'log': 0.17}
LANG_ES = 0.5
_AUTHOR = re.compile(b'((?:Copyright\\s*(?:\\(c\\)|\\xc2\\xa9|\\(C\\))?\\s*(?:\\d{4}(?:\\s*[-,]\\s*\\d{4})*,?\\s+)|@author\\s+|\\bAuthors?:\\s*|\\bAutor(?:es)?:\\s*|\\bMaintainer:\\s*|Signed-off-by:\\s*|Co-authored-by:\\s*|Reviewed-by:\\s*|__author__\\s*=\\s*[\\"\']|\\"author\\"\\s*:\\s*\\"))([A-Z][A-Za-z\\x80-\\xff\'\\-]+(?:[ ][A-Z][A-Za-z\\x80-\\xff\'.\\-]+){1,3})')

_DANGER = re.compile(b'gh[pousr]_[A-Za-z0-9]{36}|github_pat_[A-Za-z0-9_]{50,}|glpat-[A-Za-z0-9_\\-]{20}|(?:AKIA|ASIA)[0-9A-Z]{16}|xox[baprs]-[0-9A-Za-z\\-]{10,}|sk_live_[0-9A-Za-z]{20,}|AIza[0-9A-Za-z\\-_]{35}|sk-ant-api\\d{2}-|sk-(?:proj-)?[A-Za-z0-9\\-_]{40,}|hf_[A-Za-z0-9]{34}|npm_[A-Za-z0-9]{36}|SG\\.[A-Za-z0-9\\-_]{20,}\\.|-----BEGIN [A-Z ]*PRIVATE KEY-----|[a-z][a-z0-9+.\\-]*://[^\\s:/@]+:[^\\s:/@]{8,}@')

def _danger():
    return _DANGER

class Background:

    def __init__(self, data, lo, hi):
        (self.data, self.lo, self.hi) = (data, int(lo), int(hi))
        if self.hi - self.lo < 1 << 16:
            raise ValueError(f'corpus range [{lo}, {hi}) too small')

    def window(self, rng, n):
        off = int(rng.integers(self.lo, self.hi - n - 1))
        return bytes(self.data[off:off + n])

def _first_names():
    L = get_lists('all')
    if not hasattr(L, '_firsts'):
        s = set()
        for pool in (L.first_es_m, L.first_es_f, L.first_en_m, L.first_en_f):
            for r in pool.items:
                for w in r[0].split():
                    s.add(w.lower())
        L._firsts = s
    return L._firsts

def scrub(bg, rng, L, lang):
    if _danger().search(bg):
        return None
    reps = []
    for (a, b, t, _) in RU.detect(bg, ['EMAIL', 'IP', 'URL_CREDENTIALS', 'PHONE']):
        typ = TYPES[t]
        if typ == 'EMAIL':
            new = VL.email(rng, L, lang).encode()
        elif typ == 'IP':
            new = VL.ip(rng).encode()
        elif typ == 'PHONE':
            new = VL.phone(rng, lang).encode()
        else:
            new = f'{VL.pick(rng, VL._USERS)}:{VL.secret_password(rng)}'.encode()
        reps.append((a, b, new, t))
    firsts = _first_names()
    for m in _AUTHOR.finditer(bg):
        nm = m.group(2).decode('utf-8', 'replace')
        if nm.split()[0].lower() in firsts and (not any((a < m.end(2) and m.start(2) < b for (a, b, _, _) in reps))):
            reps.append((m.start(2), m.end(2), VL.person(rng, L, VL.pick(rng, ['es', 'en']))[0].encode('utf-8'), IDX['PERSON_NAME']))
    reps.sort()
    (out, spans, last) = ([], [], 0)
    for (a, b, new, t) in reps:
        if a < last:
            continue
        out.append(bg[last:a])
        pos = sum((len(x) for x in out))
        out.append(new)
        spans.append((pos, pos + len(new), t))
        last = b
    out.append(bg[last:])
    return (b''.join(out), spans)

def _crop(data, spans, seq_len, rng, anchor=None):
    n = len(data)
    if n <= seq_len:
        data = data + b'\n' * (seq_len - n)
        return (data, [(a, min(b, seq_len), t) for (a, b, t) in spans if a < seq_len])
    (lo_s, hi_s) = (0, n - seq_len)
    if anchor is not None:
        (a0, b0) = anchor
        if b0 - a0 <= seq_len - 16:
            (lo_s, hi_s) = (max(0, b0 - seq_len + 8), min(hi_s, max(0, a0 - 8)))
        else:
            (lo_s, hi_s) = (a0, min(hi_s, b0 - seq_len))
        (lo_s, hi_s) = (min(lo_s, hi_s), max(lo_s, hi_s))
    start = int(rng.integers(lo_s, hi_s + 1))
    if VL.chance(rng, 0.7):
        nl = data.rfind(b'\n', max(0, start - 200), start)
        if nl >= 0 and nl + 1 >= lo_s:
            start = nl + 1
    for (a, b, t) in spans:
        if a < start < b:
            start = b
    start = min(start, n - seq_len)
    for (a, b, t) in spans:
        if a < start < b:
            start = a if a >= 0 else start
    start = max(start, 0)
    win = data[start:start + seq_len]
    out = [(max(a, start) - start, min(b, start + seq_len) - start, t) for (a, b, t) in spans if b > start and a < start + seq_len]
    return (win, [s for s in out if s[1] > s[0]])
IGNORE_T = IGNORE - 1

class PublicDocs:

    def __init__(self, directory, sources=None, collapse=False):
        import json
        import os
        from .public_map import map_doc, to_bytes, type_index
        (self.docs, self.ids) = ([], [])
        names = sorted((f[:-6] for f in os.listdir(directory) if f.endswith('.jsonl')))
        use = [n for n in names if sources is None or n in sources]
        if not use:
            raise ValueError(f'no public training documents in {directory} (for {sources}): build them with the data scripts (see specialists/pii/README.md)')
        ex_p = os.path.join(directory, 'EXCLUDED.json')
        excluded = {e['id'] for e in json.load(open(ex_p))} if os.path.exists(ex_p) else set()
        self.excluded = len(excluded)
        for n in use:
            with open(os.path.join(directory, n + '.jsonl'), encoding='utf-8') as f:
                for ln in f:
                    d = json.loads(ln)
                    if d['id'] in excluded:
                        continue
                    self.ids.append(d['id'])
                    (sp, ig) = map_doc(d['text'], [tuple(x) for x in d['labels']], d['source'], collapse=collapse, train=True)
                    spans = [(a, b, type_index(t, collapse)) for (a, b, t) in to_bytes(d['text'], sp)] + [(a, b, IGNORE_T) for (a, b, _) in to_bytes(d['text'], ig)]
                    self.docs.append((d['text'].encode('utf-8'), spans))
        self.sources = use

    def sample(self, rng, need):
        (out, spans, n) = ([], [], 0)
        while n < need:
            (data, sp) = self.docs[int(rng.integers(0, len(self.docs)))]
            if out:
                out.append(b'\n\n')
                n += 2
            spans += [(a + n, b + n, t) for (a, b, t) in sp]
            out.append(data)
            n += len(data)
        return (b''.join(out), spans)

class Generator:

    def __init__(self, bg, seq_len=512, mix=None, empty_frac=0.18, lists='train', held_out_forms=False, public=None, collapse=False):
        (self.bg, self.seq_len) = (bg, int(seq_len))
        self.mix = dict(DEFAULT_MIX, **mix or {})
        self.mix = {k: v for (k, v) in self.mix.items() if v > 0}
        if self.mix.get('public', 0) > 0 and public is None:
            raise ValueError('`mix.public` > 0 needs the public training documents (`data.public_train`)')
        tot = sum(self.mix.values())
        self.scenes = list(self.mix)
        self.cum = np.cumsum([self.mix[s] / tot for s in self.scenes])
        self.empty_frac = float(empty_frac)
        self.L = get_lists(lists)
        self.held_out = bool(held_out_forms)
        (self.public, self.collapse) = (public, bool(collapse))
        self.stats = {'scenes': {s: 0 for s in self.scenes}, 'types': {t: 0 for t in (('PII',) if collapse else TYPES)}, 'empty_windows': 0, 'rescrubbed_spans': 0, 'skipped_danger': 0, 'ignored_spans': 0}

    def _doc_scene(self, rng, scene, lang, positive, target=None):
        (L, d) = (self.L, FM.Doc())
        target = target or self.seq_len + int(rng.integers(64, 400))
        guard = 0
        while d.n < target and guard < 200:
            guard += 1
            if scene == 'data':
                recs = [FM.record(rng, L, lang, all_negative=not positive) for _ in range(int(rng.integers(1, 5)))]
                fmt = FM.renderer(rng, self.held_out)
                if fmt in ('java', 'go', 'env', 'ini', 'toml'):
                    recs = recs[:1]
                FM.render(rng, d, recs, fmt)
                d.add('\n')
            elif scene == 'prose':
                FM.paragraph(rng, L, lang, d, positive=positive)
            elif scene == 'email':
                FM.email_message(rng, L, lang, d, positive=positive)
            elif scene == 'log':
                for _ in range(int(rng.integers(2, 8))):
                    FM.log_line(rng, L, lang, d, positive=positive)
        return (d.data(), d.spans)

    def _code_scene(self, rng, lang, positive):
        L = self.L
        for _ in range(12):
            raw = self.bg.window(rng, self.seq_len * 2)
            sc = scrub(raw, rng, L, lang)
            if sc is None:
                self.stats['skipped_danger'] += 1
                continue
            (win, bspans) = sc
            if not positive and bspans:
                continue
            break
        else:
            (win, bspans) = (b'\n' * (self.seq_len * 2), [])
        nl = [i for (i, c) in enumerate(win[:len(win) // 2]) if c == 10]
        cut = nl[int(rng.integers(0, len(nl)))] + 1 if nl else len(win) // 3
        hint = FM.guess_hint(win)
        blk = FM.Doc()
        for _ in range(int(rng.integers(1, 4))):
            FM.code_snippet(rng, L, lang, blk, positive=positive, hint=hint)
            if blk.n > self.seq_len - 96:
                break
        block = blk.data()
        spans = [(a, b, t) for (a, b, t) in bspans if b <= cut] + [(a + cut, b + cut, t) for (a, b, t) in blk.spans] + [(a + len(block), b + len(block), t) for (a, b, t) in bspans if a >= cut]
        data = win[:cut] + block + win[cut:]
        self.stats['rescrubbed_spans'] += len(bspans)
        return (data, spans, (cut, cut + len(block)))

    def window(self, rng):
        scene = self.scenes[min(int(np.searchsorted(self.cum, rng.random(), side='right')), len(self.scenes) - 1)]
        lang = 'es' if VL.chance(rng, LANG_ES) else 'en'
        positive = not VL.chance(rng, self.empty_frac)
        anchor = None
        if scene == 'code':
            (data, spans, anchor) = self._code_scene(rng, lang, positive)
        elif scene == 'public':
            (data, spans) = self.public.sample(rng, self.seq_len + int(rng.integers(64, 600)))
        else:
            (data, spans) = self._doc_scene(rng, scene, lang, positive)
        if self.collapse:
            spans = [(a, b, 0 if t != IGNORE_T else t) for (a, b, t) in spans]
        (win, spans) = _crop(data, spans, self.seq_len, rng, anchor)
        S = np.zeros(self.seq_len, dtype=np.uint8)
        spans = _separate(spans)
        paint(S, spans)
        self.stats['scenes'][scene] += 1
        if not window_class(S):
            self.stats['empty_windows'] += 1
        for (_, _, t) in spans:
            if t == IGNORE_T:
                self.stats['ignored_spans'] += 1
            else:
                self.stats['types']['PII' if self.collapse else TYPES[t]] += 1
        return (np.frombuffer(win, dtype=np.uint8), S)

    def document(self, rng, target):
        scene = self.scenes[min(int(np.searchsorted(self.cum, rng.random(), side='right')), len(self.scenes) - 1)]
        if scene == 'public':
            scene = 'data'
        lang = 'es' if VL.chance(rng, LANG_ES) else 'en'
        positive = not VL.chance(rng, self.empty_frac)
        if scene == 'code':
            for _ in range(12):
                raw = self.bg.window(rng, int(target))
                sc = scrub(raw, rng, self.L, lang)
                if sc is not None and (positive or not sc[1]):
                    break
            else:
                sc = (b'', [])
            (data, spans) = sc
            hint = FM.guess_hint(data)
            for _ in range(int(rng.integers(1, 4))):
                nl = [i for (i, c) in enumerate(data) if c == 10]
                cut = nl[int(rng.integers(0, len(nl)))] + 1 if nl else len(data)
                if any((a < cut < b for (a, b, _) in spans)):
                    continue
                blk = FM.Doc()
                FM.code_snippet(rng, self.L, lang, blk, positive=positive, hint=hint)
                block = blk.data()
                spans = [(a, b, t) for (a, b, t) in spans if b <= cut] + [(a + cut, b + cut, t) for (a, b, t) in blk.spans] + [(a + len(block), b + len(block), t) for (a, b, t) in spans if a >= cut]
                data = data[:cut] + block + data[cut:]
        else:
            (data, spans) = self._doc_scene(rng, scene, lang, positive, int(target) + 200)
            end = len(data)
            for i in range(min(int(target), len(data) - 1), len(data)):
                if data[i] == 10 and (not any((a <= i < b for (a, b, _) in spans))):
                    end = i + 1
                    break
            data = data[:end]
            spans = [(a, min(b, end), t) for (a, b, t) in spans if a < end]
        spans = _separate(spans)
        if self.collapse:
            spans = [(a, b, 0) for (a, b, _) in spans]
        self.stats['scenes'][scene] += 1
        return (data, spans, scene, lang)

    def build(self, n, rng):
        X = np.zeros((n, self.seq_len), dtype=np.uint8)
        S = np.zeros((n, self.seq_len), dtype=np.uint8)
        for i in range(n):
            (X[i], S[i]) = self.window(rng)
        return (X, window_class(S), S)

def _separate(spans):
    spans = sorted(set(spans))
    out = []
    for (a, b, t) in spans:
        if out and a < out[-1][1]:
            (pa, pb, pt) = out[-1]
            if b - a > pb - pa:
                out[-1] = (a, b, t)
            continue
        if out and a == out[-1][1] and (t == out[-1][2]):
            a += 1
            if a >= b:
                continue
        out.append((a, b, t))
    return out
