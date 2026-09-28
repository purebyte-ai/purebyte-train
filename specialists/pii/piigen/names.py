"""The public lists as samplers, with a stable HELD-OUT split.

The lists are read from `lists_dir()`: `$PUREBYTE_DATA/pii/lists/` by default (`LISTS_DIR` overrides it; the plugin
sets it). They are built from public sources by the data scripts (see the specialist's README.md) and never stored in
git. Every file is a TSV with a header line:

    names_es_male.tsv, names_es_female.tsv, names_en_male.tsv, names_en_female.tsv    name, frequency
    surnames_es.tsv, surnames_en.tsv                                                 surname, frequency
    streets_es.tsv          type, name, postcode, municipality, province
    places_us.tsv           name, state
    municipalities_es.tsv   code, name, province
    street_affixes_en.tsv   suffix
    sentences_es.tsv, sentences_en.tsv      kind, text   (kind: clean | holes; a hole `{P}` is a person slot)

`Lists("train")` draws only from the ~90 % of every list whose stable hash is not 0 mod 10; `Lists("test")` only from
the other ~10 % (an evaluation set built with it measures names, streets and sentences that training never saw);
`Lists("all")` draws from everything. Frequent items are drawn more often, flattened (weight = frequency ** 0.5).
"""
import hashlib
import os
import numpy as np
LISTS_DIR = None

def lists_dir():
    if LISTS_DIR:
        return LISTS_DIR
    return os.path.join(os.environ.get('PUREBYTE_DATA', os.path.join(os.path.expanduser('~'), '.purebyte', 'data')), 'pii', 'lists')
_CACHE = {}

def held_out(key):
    return hashlib.sha1(key.lower().encode('utf-8')).digest()[0] % 10 == 0

class _Pool:

    def __init__(self, items, weights):
        self.items = items
        w = np.asarray(weights, dtype=np.float64)
        w = np.sqrt(np.maximum(w, 1e-09))
        self.cum = np.cumsum(w / w.sum())

    def __len__(self):
        return len(self.items)

    def draw(self, rng):
        i = int(np.searchsorted(self.cum, rng.random(), side='right'))
        return self.items[min(i, len(self.items) - 1)]

def _read(name):
    p = os.path.join(lists_dir(), name)
    if not os.path.exists(p):
        raise FileNotFoundError(f'{p} is missing: build the PII lists with the data scripts (see specialists/pii/README.md)')
    with open(p, encoding='utf-8') as f:
        next(f)
        return [ln.rstrip('\n').split('\t') for ln in f if ln.strip()]

class Lists:

    def __init__(self, split='train'):
        if split not in ('train', 'test', 'all'):
            raise ValueError(split)
        self.split = split

        def keep(key):
            return split == 'all' or held_out(key) == (split == 'test')

        def pool(rows, key=0, wcol=1, clean=None):
            rows = [r for r in rows if keep(r[key]) and (clean is None or clean(r))]
            return _Pool(rows, [float(r[wcol]) if len(r) > wcol else 1.0 for r in rows])
        ok_name = lambda r: r[0].replace(' ', '').replace('-', '').replace("'", '').isalpha()
        self.first_es_m = pool(_read('names_es_male.tsv'), clean=ok_name)
        self.first_es_f = pool(_read('names_es_female.tsv'), clean=ok_name)
        self.sur_es = pool(_read('surnames_es.tsv'), clean=ok_name)
        self.first_en_m = pool(_read('names_en_male.tsv'), clean=ok_name)
        self.first_en_f = pool(_read('names_en_female.tsv'), clean=ok_name)
        self.sur_en = pool(_read('surnames_en.tsv'), clean=ok_name)
        streets = [r for r in _read('streets_es.tsv') if '(' not in r[1] and len(r[1]) >= 3]
        self.streets_es = _Pool([r for r in streets if keep(r[1])], [1.0] * sum((1 for r in streets if keep(r[1]))))
        places = _read('places_us.tsv')
        self.places_us = _Pool([r for r in places if keep(r[0])], [1.0] * sum((1 for r in places if keep(r[0]))))
        munis = _read('municipalities_es.tsv')
        self.munis_es = _Pool([r for r in munis if keep(r[1])], [1.0] * sum((1 for r in munis if keep(r[1]))))
        self.suffix_en = [r[0] for r in _read('street_affixes_en.tsv')]
        self._keep = keep
        self._sent = None

    @property
    def sent(self):
        """The sentence lists, read the first time a sentence is drawn: only the prose and e-mail scenes draw one, so
        a recipe without them runs without sentences_{es,en}.tsv (reading them draws no random number)."""
        if self._sent is None:
            sent = {}
            for lang in ('es', 'en'):
                rows = _read(f'sentences_{lang}.tsv')
                for kind in ('clean', 'holes'):
                    items = [r[1] for r in rows if r[0] == kind and self._keep(r[1])]
                    sent[lang, kind] = _Pool(items, [1.0] * len(items))
            self._sent = sent
        return self._sent

    def first(self, rng, lang, gender=None):
        g = gender or ('m' if rng.random() < 0.5 else 'f')
        p = {('es', 'm'): self.first_es_m, ('es', 'f'): self.first_es_f, ('en', 'm'): self.first_en_m, ('en', 'f'): self.first_en_f}[lang, g]
        return p.draw(rng)[0]

    def surname(self, rng, lang):
        return (self.sur_es if lang == 'es' else self.sur_en).draw(rng)[0]

    def street_es(self, rng):
        return tuple(self.streets_es.draw(rng))

    def place_us(self, rng):
        return tuple(self.places_us.draw(rng))

    def municipality_es(self, rng):
        return tuple(self.munis_es.draw(rng))

    def sentence(self, rng, lang, kind):
        return self.sent[lang, kind].draw(rng)

def get_lists(split='train'):
    if split not in _CACHE:
        _CACHE[split] = Lists(split)
    return _CACHE[split]
