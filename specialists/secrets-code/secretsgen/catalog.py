"""Catalogues of real credential formats, read from $PUREBYTE_DATA/catalogs (built by data/build_secret_catalogs.py
from pinned upstream commits of gitleaks and trufflehog; never stored in git).

    gitleaks.json    {"rules": [{"id", "regex", "kw"}...], "tps": [...], "fps": [...]}
    trufflehog.json  {"providers": [...], "patterns": [...], "examples": [...]}

A value is sampled from a rule's regex and VALIDATED against the original regex (sampler artefacts are discarded).
Missing catalogues are recorded (`catalogs_loaded()`), and the vertical refuses to build data without them.
"""
import json
import os
import re
import numpy as np
_HERE = os.path.dirname(os.path.abspath(__file__))
_DIR = os.path.join(os.environ.get('PUREBYTE_DATA', os.path.join(os.path.expanduser('~'), '.purebyte', 'data')), 'catalogs')
_CAT = os.path.join(_DIR, 'gitleaks.json')
_MISSING = []

def _shown(path):
    """A path for a message: relative to the working directory when it can be (on Windows `os.path.relpath` raises
    when the two are on different drives, which would crash the import exactly when a catalogue is missing)."""
    try:
        return os.path.relpath(path)
    except ValueError:
        return path

def catalogs_loaded():
    return (not _MISSING, dict(gitleaks_rules=len(RULES), gitleaks_fps=len(FPS_CURATED), trufflehog_providers=len(PROVIDERS), trufflehog_examples=len(EXAMPLES_TH), missing=list(_MISSING)))

def _expand_klass(txt):
    (outside, i) = ([], 0)
    while i < len(txt):
        if txt[i] == '\\' and i + 1 < len(txt):
            c = txt[i + 1]
            outside += {'w': list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_'), 'd': list('0123456789'), 's': [' '], '.': ['.'], '-': ['-'], '/': ['/'], '+': ['+'], '\\': ['\\']}.get(c, [c])
            i += 2
        elif i + 2 < len(txt) and txt[i + 1] == '-':
            outside += [chr(x) for x in range(ord(txt[i]), ord(txt[i + 2]) + 1)]
            i += 3
        else:
            outside.append(txt[i])
            i += 1
    return outside or list('abcdefghijklmnopqrstuvwxyz0123456789')
_CLEAN = [(re.compile('\\(\\?i\\)'), ''), (re.compile('\\(\\?-i\\)'), ''), (re.compile('\\(\\?:\\s*\\)'), ''), (re.compile('\\(\\?=[^)]*\\)'), ''), (re.compile('\\(\\?![^)]*\\)'), ''), (re.compile('\\(\\?<=[^)]*\\)'), ''), (re.compile('\\(\\?<![^)]*\\)'), ''), (re.compile('\\\\b'), ''), (re.compile('\\\\A'), ''), (re.compile('\\\\z'), '')]

def sample(rx, rng, cap=220):
    for (pat, sub) in _CLEAN:
        rx = pat.sub(sub, rx)
    (outside, i) = ([], 0)
    while i < len(rx) and len(''.join(outside)) < cap:
        c = rx[i]
        if c in '^$':
            i += 1
        elif c == '\\' and i + 1 < len(rx):
            n = rx[i + 1]
            if n in 'wds':
                outside.append(str(rng.choice(_expand_klass('\\' + n))))
                i += 2
            else:
                outside.append(n)
                i += 2
        elif c == '[':
            j = i + 1
            while j < len(rx) and (rx[j] != ']' or rx[j - 1] == '\\'):
                j += 1
            alphab = _expand_klass(rx[i + 1:j])
            i = j + 1
            (n, i) = _quantifier(rx, i, rng)
            outside.append(''.join((str(rng.choice(alphab)) for _ in range(min(n, 90)))))
        elif c == '(':
            (j, prof) = (i + 1, 1)
            while j < len(rx) and prof:
                if rx[j] == '\\':
                    j += 2
                    continue
                prof += (rx[j] == '(') - (rx[j] == ')')
                j += 1
            body = rx[i + 1:j - 1]
            if body.startswith('?:'):
                body = body[2:]
            options = _part_alternation(body)
            i = j
            (n, i) = _quantifier(rx, i, rng)
            if n:
                outside.append(sample(str(rng.choice(options)), rng, cap=60))
        elif c == '.':
            i += 1
            (n, i) = _quantifier(rx, i, rng)
            outside.append(''.join((str(rng.choice(list('abcdefghijklmnopqrstuvwxyz0123456789'))) for _ in range(min(n, 40)))))
        else:
            lit = c
            i += 1
            (n, i) = _quantifier(rx, i, rng)
            outside.append(lit * n)
    return ''.join(outside)

def _part_alternation(s):
    (out, prof, act) = ([], 0, '')
    for ch in s:
        if ch == '(':
            prof += 1
        elif ch == ')':
            prof -= 1
        if ch == '|' and prof == 0:
            out.append(act)
            act = ''
        else:
            act += ch
    out.append(act)
    return out or ['']

def _quantifier(rx, i, rng):
    if i >= len(rx):
        return (1, i)
    c = rx[i]
    if c == '{':
        j = rx.find('}', i)
        if j < 0:
            return (1, i)
        body = rx[i + 1:j]
        if ',' in body:
            (a, _, b) = body.partition(',')
            a = int(a) if a.strip().isdigit() else 1
            b = int(b) if b.strip().isdigit() else a + 24
            return (int(rng.integers(a, min(b, a + 60) + 1)), j + 1)
        return (int(body) if body.isdigit() else 1, j + 1)
    if c == '+':
        return (int(rng.integers(8, 44)), i + 1)
    if c == '*':
        return (int(rng.integers(0, 32)), i + 1)
    if c == '?':
        return (int(rng.integers(0, 2)), i + 1)
    return (1, i)

def _load():
    if not os.path.exists(_CAT):
        _MISSING.append(_shown(_CAT))
        return ([], [], [])
    d = json.load(open(_CAT, encoding='utf8'))
    distinctive = []
    for r in d['rules']:
        rx = r.get('regex', '')
        if not rx or '[\\w.-]{0,50}' in rx or '(?:[ \\t\\w.-]' in rx:
            continue
        distinctive.append(r)
    VOWELS = set('aeiou')
    words = sorted({k for r in d['rules'] for k in r.get('kw', []) if 4 <= len(k) <= 20 and k.isalpha() and (len(VOWELS & set(k)) >= 1) and (sum((c in VOWELS for c in k)) >= 2)})
    return (distinctive, d.get('fps', []), words)
(RULES, FPS_CURATED, WORDS) = _load()

# A global inline flag group after the start, as in `abc(?i)def`: Python 3.10 compiles it with a warning, 3.11 and newer
# refuse it. It is refused on every version, so that the generated data do not depend on the Python version.
_LATE_GLOBAL_FLAGS = re.compile(r'.\(\?[aiLmsux]+\)')

def _compile(rx):
    if _LATE_GLOBAL_FLAGS.search(rx):
        return None
    try:
        return re.compile(rx)
    except re.error:
        return None
_COMPILED = {}

def _held_out(r):
    from .credentials import HOLDOUT
    if not HOLDOUT:
        return False
    text = (str(r.get('id', '')) + ' ' + str(r.get('description', ''))).lower()
    return any((t in text for t in HOLDOUT))

def catalog_secret(rng, attempts=8):
    if not RULES:
        return None
    for _ in range(attempts):
        r = RULES[rng.integers(0, len(RULES))]
        if _held_out(r):
            continue
        rx = r['regex']
        if rx not in _COMPILED:
            _COMPILED[rx] = _compile(rx)
        comp = _COMPILED[rx]
        try:
            v = sample(rx, rng)
        except Exception:
            continue
        if not 12 <= len(v) <= 200 or '\n' in v:
            continue
        if any((c in v for c in '?|\\')):
            continue
        if comp is not None and (not comp.search(v)):
            continue
        return v
    return None

def false_positive_curated(rng):
    if not FPS_CURATED:
        return None
    v = FPS_CURATED[rng.integers(0, len(FPS_CURATED))]
    v = v.replace('\\"', '"').replace('\\\\', '\\')
    m = re.search('[:=]=?\\s*["\\\'`]([^"\\\'`]+)', v)
    if m:
        v = m.group(1)
    v = v.strip()
    if ':=' in v or any((c in v for c in '"\'`')):
        return None
    return v if 8 <= len(v) <= 160 and '\n' not in v else None
_CAT_TH = os.path.join(_DIR, 'trufflehog.json')

def _load_th():
    if not os.path.exists(_CAT_TH):
        _MISSING.append(_shown(_CAT_TH))
        return ([], [])
    d = json.load(open(_CAT_TH, encoding='utf8'))
    provs = [p for p in d.get('providers', []) if 3 <= len(p) <= 20 and p.isalnum()]
    ejs = [e for e in d.get('examples', []) if 20 <= len(e) <= 120 and (not e.startswith(('-----', 'http')))]
    return (provs, ejs)
(PROVIDERS, EXAMPLES_TH) = _load_th()
_SUFFIXES_PROV = ['_api_key', '_token', '_secret', '_key', '_password', '_apikey', '_access_token', '_client_secret', '_auth', '_credential', '_api_secret', '_private_key', '_sk', '_pat']

def name_of_provider(rng):
    if not PROVIDERS:
        return None
    p = PROVIDERS[rng.integers(0, len(PROVIDERS))]
    s = _SUFFIXES_PROV[rng.integers(0, len(_SUFFIXES_PROV))]
    n = p + s
    return n.upper() if rng.integers(0, 3) == 0 else n

def example_trufflehog(rng):
    if not EXAMPLES_TH:
        return None
    return EXAMPLES_TH[rng.integers(0, len(EXAMPLES_TH))]

def word_key(rng):
    return str(rng.choice(WORDS)) if WORDS else None
