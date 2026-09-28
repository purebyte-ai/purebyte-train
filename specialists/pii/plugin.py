"""The verticals of the pii specialist, both built by the `piigen` generator:

    pii_mask    LABEL-AGNOSTIC masking, the released policy: every personal-data span is one type, `PII` (BIOES x 1 =
                5 labels, gated by the window head). Every label the public datasets give is personal data.
    pii_spans   EXPERIMENTAL typed spans: 12 types (49 labels); labels of the public datasets outside the taxonomy are
                ignored (label -100). See recipes/experimental-typed.yaml.

The runtime's `redact` profile copies the input byte for byte and replaces only the spans with markers (spec/OUTPUT.md,
section 9): `[PII_1]` for `pii_mask` models, typed markers such as `[EMAIL_1]` for `pii_spans` models. This plugin
builds the training windows with the generator and provides the classic rival (the rules).

Every key of the task's `data:` block is read HERE (the factory fails on a declared key that nobody read):

    n_train, n_eval, seq_len, split_policy.seed, path (the code corpus; read by the factory)
    mix           scene proportions {code, data, prose, email, log, public}; a scene left out keeps its default
                  (code 0.35, data 0.25, prose 0.13, email 0.10, log 0.17, public 0); 0 switches a scene off
    empty_frac    fraction of GENERATED windows built with nothing to redact (default 0.18; the public documents are
                  taken as they are)
    lists         the half of the public lists the windows draw from: `train` (default) or `all`. `test` is refused:
                  that half is held out for evaluation sets
    public_train  {dir, sources}: labelled documents of public TRAIN splits for the `public` scene (needs mix.public
                  > 0). A document goes to train / select / test by a stable hash of its id (about 90 / 5 / 5 %)
    lookalikes    pii_mask only: `text` (default) or `ignore`. What the generated look-alikes are (order ids, dates,
                  postcodes, cities, company names, URLs: personal data for the public datasets, negatives for the
                  typed policy): plain text, or IGNORE (label -100, taught neither way; piigen/lookalikes.py). `ignore`
                  needs a generated scene in `mix`
    collapse      refused: the label-agnostic policy is its own format, `pii_mask`

Resources, read from $PUREBYTE_DATA (built by the data scripts, never stored in git):
    pii/lists/          the public lists (the files are listed in piigen/names.py)
    pii/public_train*/  <source>.jsonl labelled documents (data/fetch_pii.py, data/build_pii_pool.py)
    catalogs/           the credential-format catalogues of the secrets-code specialist (SECRET values and rules)

S holds, per byte, 0 outside every span, 1 + type index inside one (1 for PII), and 255 (IGNORE, trained with label
-100) on bytes taught neither way; Y = the window holds a span (IGNORE bytes do not count), which the core contract
checks.
"""
import hashlib
import os
import sys

import numpy as np
from pbtrain import paths
from pbtrain.eval.spans import _prf
from pbtrain.model.decode import mask_spans
from pbtrain.verticals import REGISTRY, Partitions, Vertical, register

from piigen import metrics as _metrics
from piigen import names as _names
from piigen import rules as _rules
from piigen.generator import DEFAULT_MIX, Background, Generator, PublicDocs
from piigen.lookalikes import POLICIES as LOOKALIKE_POLICIES
from piigen.lookalikes import GeneratorLookalikes
from piigen.types import IDX, IGNORE, K, TYPES, ignored_of, spans_of
from secretsgen import catalog as _catalog
from secretsgen import standins as _standins

SCENES = ("code", "data", "prose", "email", "log", "public")
LIST_FILES = ("names_es_male.tsv", "names_es_female.tsv", "surnames_es.tsv", "names_en_male.tsv",
              "names_en_female.tsv", "surnames_en.tsv", "streets_es.tsv", "places_us.tsv", "municipalities_es.tsv",
              "street_affixes_en.tsv", "sentences_es.tsv", "sentences_en.tsv")
SENTENCE_FILES = ("sentences_es.tsv", "sentences_en.tsv")     # read only by the prose and e-mail scenes
AGNOSTIC_TYPES = ("PII",)
# the rival's knob: a floor on the rules' own scores (0.3..1.0), chosen on `select`. The values sit between the score
# levels the rules emit, so a floor never lands on a rounding boundary; 0.0 keeps every detection (`rules.detect`).
RULE_SCORE_FLOORS = (0.0, 0.55, 0.65, 0.72, 0.78, 0.82, 0.87, 0.92, 0.97)
_STATE = {}
_PUBLIC = {}


# ------------------------------------------------------------------------------------------------ data block and resources
def _config(cfgd, agnostic=False):
    """The vertical's keys of the `data:` block, validated (a key that would do nothing is refused)."""
    mix = cfgd.get("mix")
    if mix is not None:
        bad = sorted(set(mix) - set(SCENES))
        if bad:
            raise ValueError(f"unknown scene(s) {bad} in `mix`; the scenes are {list(SCENES)}")
        mix = {k: float(mix[k]) for k in list(mix.keys())}
        if any(v < 0 for v in mix.values()):
            raise ValueError(f"`mix` proportions must be >= 0: {mix}")
    merged = dict(DEFAULT_MIX, **(mix or {}))
    if sum(v for v in merged.values() if v > 0) <= 0:
        raise ValueError("`mix` switches every scene off")
    generated = any(v > 0 for k, v in merged.items() if k != "public")
    empty = float(cfgd.get("empty_frac", 0.18))
    if not 0.0 <= empty < 1.0:
        raise ValueError(f"`empty_frac: {empty}` must be in [0, 1)")
    lists = str(cfgd.get("lists", "train"))
    if lists == "test":
        raise ValueError("`lists: test` draws from the held-out half of the lists, which evaluation sets are built from: "
                         "a model trained on it could not be measured on them. Use `train` (or `all`)")
    if lists not in ("train", "all"):
        raise ValueError(f"`lists: {lists}`: train | all")
    if "collapse" in cfgd:
        cfgd["collapse"]
        raise ValueError("`collapse` is not a key of this vertical: the label-agnostic policy (every personal-data label "
                         "is one type, PII) is its own format, `format: pii_mask`; the typed one is `format: pii_spans`")
    lookalikes = "text"
    if "lookalikes" in cfgd:
        lookalikes = str(cfgd["lookalikes"])
        if not agnostic:
            raise ValueError("`lookalikes` belongs to `format: pii_mask`: under the typed policy the look-alikes ARE the "
                             "negatives and stay plain text")
        if lookalikes not in LOOKALIKE_POLICIES:
            raise ValueError(f"`lookalikes: {lookalikes}`: {' | '.join(LOOKALIKE_POLICIES)}")
        if lookalikes == "ignore" and not generated:
            raise ValueError("`lookalikes: ignore` acts on generated windows only and `mix` switches every generated "
                             "scene off: it would do nothing")
    pt = cfgd.get("public_train")
    if pt is not None:
        if "dir" not in pt:
            raise ValueError("`public_train` needs `dir`")
        sources = pt.get("sources")
        pt = dict(dir=paths.expand(pt["dir"]), sources=(sorted(str(s) for s in sources) if sources else None))
    uses_public = merged.get("public", 0.0) > 0
    if uses_public and pt is None:
        raise ValueError("`mix.public` > 0 needs the labelled public documents (`public_train: {dir: ...}`)")
    if pt is not None and not uses_public:
        raise ValueError("`public_train` is declared but `mix.public` is 0: it would do nothing")
    return dict(mix=mix, empty_frac=empty, lists=lists, public_train=pt, lookalikes=lookalikes)


def _sha16(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()[:16]


def _reload_catalogs():
    """The secrets-code plugin owns the catalogue loader (its `reload_catalogs`); `uses: [secrets-code]` loads it."""
    cls = REGISTRY.get("secrets_synthetic")
    fn = getattr(sys.modules.get(cls.__module__), "reload_catalogs", None) if cls is not None else None
    if fn is None:
        raise ValueError("the secrets-code specialist is not loaded (specialist.yaml `uses: [secrets-code]`): its "
                         "credential catalogues feed the SECRET type and the rules")
    return fn()


def sync_resources(sentences=True):
    """Points the generator and the rules at $PUREBYTE_DATA (again, when it changed in this process) and checks that
    the lists and the catalogues are there. `sentences`: whether the build draws sentences (the prose and e-mail
    scenes); without them the two sentence lists may be absent. Returns what was used (digests), for the manifest."""
    data = os.path.abspath(paths.data_dir())
    lists = os.path.join(data, "pii", "lists")
    if _names.LISTS_DIR != lists:
        _names.LISTS_DIR = lists
        _names._CACHE.clear()
        _rules._NAMES = None
    needed = [f for f in LIST_FILES if sentences or f not in SENTENCE_FILES]
    missing = [f for f in needed if not os.path.exists(os.path.join(lists, f))]
    if missing:
        raise ValueError(f"missing PII lists {missing} in {lists}: build them with the data scripts (see "
                         f"specialists/pii/README.md)")
    if os.path.normcase(os.path.abspath(_catalog._DIR)) != os.path.normcase(os.path.join(data, "catalogs")):
        _reload_catalogs()
    if _STATE.get("rules_of") is not _catalog.RULES:          # the rules compile the catalogue once: recompile
        _rules._SECRET_RULES = None
        _STATE["rules_of"] = _catalog.RULES
    ok, detail = _catalog.catalogs_loaded()
    if not ok:
        raise ValueError(f"missing catalogues of credential formats {detail['missing']}: build them with the data "
                         f"scripts of the secrets-code specialist (without them the SECRET type and its rule would "
                         f"silently be much weaker)")
    return dict(lists={f: _sha16(os.path.join(lists, f)) for f in LIST_FILES if os.path.exists(os.path.join(lists, f))},
                catalogs=detail)


def check_standins(corpus, public_dir=None, sentences=True):
    """The made-up stand-ins of data/build_smoke_data.py (lists, catalogues, public documents) among the files this
    build reads: refused unless the corpus is the smoke corpus (secretsgen/standins.py); returned for the manifest
    otherwise. Nothing is ever deleted: the message says which files to replace or remove."""
    data = os.path.abspath(paths.data_dir())
    used = [f"pii/lists/{f}" for f in LIST_FILES if sentences or f not in SENTENCE_FILES]
    used += ["catalogs/gitleaks.json", "catalogs/trufflehog.json"]
    if public_dir and os.path.isdir(public_dir):
        try:
            rel = os.path.relpath(os.path.abspath(public_dir), data).replace("\\", "/")
        except ValueError:                                       # another drive on Windows: not under the data folder
            rel = ".."
        if not rel.startswith(".."):
            used += [f"{rel}/{f}" for f in sorted(os.listdir(public_dir)) if f.endswith(".jsonl")]
    return _standins.check(data, tuple(used), getattr(corpus, "path", None),
                           "PII lists, credential catalogues or public documents",
                           "python data/fetch_pii.py lists (and train), python data/build_secret_catalogs.py; a "
                           "stand-in left among real public documents (pii/public_train/lookalike.jsonl) is removed "
                           "by hand")


def _part_of(uid):
    """train / select / test of a public document: a stable hash of its id (never one document in two splits)."""
    h = hashlib.sha1(uid.encode()).digest()[0] % 20
    return "select" if h == 0 else ("test" if h == 1 else "train")


class _Split(PublicDocs):
    """The public documents of ONE split."""

    def __init__(self, base, part):
        self.sources = base.sources
        self.docs = [d for d, i in zip(base.docs, base.ids) if _part_of(i) == part]


def _load_public(directory, sources, collapse=False):
    """The labelled public documents of a directory, mapped to the vertical's policy (cached per process)."""
    key = (directory, tuple(sources or ()), bool(collapse))
    if key not in _PUBLIC:
        pd = PublicDocs(directory, sources, bool(collapse))
        if len(pd.ids) != len(pd.docs):
            raise ValueError(f"{directory}: document ids and documents are misaligned")
        files = {s: _sha16(os.path.join(directory, s + ".jsonl")) for s in pd.sources}
        _PUBLIC[key] = (pd, dict(dir=directory, files=files, documents=len(pd.docs), excluded=pd.excluded))
    return _PUBLIC[key]


# ------------------------------------------------------------------------------------------------------ the rival
def rule_candidates(row):
    """Every rule's candidate spans in one window, before the overlap resolution of `rules.detect`:
    (text, [(char start, char end, type name, score)]), in the order `rules.detect` collects them."""
    text = bytes(row).decode("utf-8", "surrogateescape")
    return text, [(a, b, t, sc) for t in _rules.RULES for a, b, sc in _rules.RULES[t](text) if b > a]


def rule_spans(text, cands, floor=0.0, collapse=False):
    """`rules.detect` with a floor on the score: the candidates below it are dropped BEFORE the overlap resolution
    (priority, then length). -> {(byte start, byte end, type index)}, type 0 for every span with `collapse`. With
    floor 0.0 it is `rules.detect`."""
    cs = sorted((c for c in cands if c[3] >= floor), key=lambda c: (_rules.PRIORITY[c[2]], -(c[1] - c[0]), c[0]))
    taken, out = [], []
    for a, b, t, _ in cs:
        if any(a < y and x < b for x, y in taken):
            continue
        taken.append((a, b))
        out.append((a, b, t))
    if not out:
        return set()
    cum = _rules._byte_offsets(text)
    return {(int(cum[a]), int(cum[b]), 0 if collapse else IDX[t]) for a, b, t in out}


def gold_spans(row, k=K):
    """The gold spans of one window as the core reads a mask (`mask_spans`), IGNORE bytes excluded: they are not
    spans."""
    return {(int(a), int(b), int(e)) for a, b, e in mask_spans(np.where(row == IGNORE, 0, row), k)}


def _score_rules(cands, gold, floor, collapse=False):
    """Micro span-F1 with EXACT boundaries and types, window by window: how `pbtrain.eval.spans` scores the model."""
    tp = fp = fn = 0
    for (text, c), g in zip(cands, gold):
        p = rule_spans(text, c, floor, collapse)
        tp += len(p & g)
        fp += len(p - g)
        fn += len(g - p)
    return _prf(tp, fp, fn)


# ------------------------------------------------------------------------------------------------------ the verticals
@register
class PiiSpans(Vertical):
    """EXPERIMENTAL: the 12 typed span types of piigen/types.py."""
    data_format = "pii_spans"
    needs_corpus = True
    overlap_guard = "windows"     # generated records may repeat a 120-byte header by chance; real code ranges are disjoint
    entities = TYPES
    agnostic = False

    def build(self, corpus, cfgd, fracs):
        c = _config(cfgd, self.agnostic)
        mix = dict(DEFAULT_MIX, **(c["mix"] or {}))
        sentences = mix.get("prose", 0) > 0 or mix.get("email", 0) > 0
        resources = sync_resources(sentences=sentences)
        made_up = check_standins(corpus, (c["public_train"] or {}).get("dir"), sentences)
        if made_up:                               # the smoke recipes on made-up resources: allowed, and recorded
            resources["smoke_standins"] = made_up
        sp = cfgd.get("split_policy") or {}
        if "seed" not in sp:
            raise ValueError("`split_policy.seed` is required")
        seed = int(sp["seed"])
        sl, n_train, n_eval = int(cfgd["seq_len"]), int(cfgd["n_train"]), int(cfgd["n_eval"])
        lo, hi = corpus.tr_lo, corpus.tr_hi
        span = hi - lo
        cuts = [lo, lo + int(span * fracs[0]), lo + int(span * (fracs[0] + fracs[1])), hi]
        pub = None
        if c["public_train"]:
            pub, resources["public_train"] = _load_public(c["public_train"]["dir"], c["public_train"]["sources"],
                                                          self.agnostic)
        gen_cls = GeneratorLookalikes if c["lookalikes"] == "ignore" else Generator
        parts, stats, docs = [], {}, {}
        for k, (name, n, sd) in enumerate((("train", n_train, seed), ("select", n_eval, seed + 1000),
                                           ("test", n_eval, seed + 2000))):
            split_docs = _Split(pub, name) if pub is not None else None
            if split_docs is not None:
                if not split_docs.docs:
                    raise ValueError(f"no public training document falls in the `{name}` split (too few documents)")
                docs[name] = len(split_docs.docs)
            # the guard is the window READ (2 x seq_len), so no split reads a byte of the next one
            g = gen_cls(Background(corpus.data, cuts[k], cuts[k + 1] - 2 * sl - 1), sl, c["mix"], c["empty_frac"],
                        c["lists"], public=split_docs, collapse=self.agnostic)
            parts.append(g.build(n, np.random.default_rng(sd)))
            stats[name] = g.stats
        self._built = dict(config=c, stats=stats, public_docs_per_split=docs or None, resources=resources)
        return Partitions(*parts)

    def guards(self, cfgd, t):
        """Every type is planted in a large enough train split (typed policy), and the windows with nothing to redact
        are about `empty_frac` (checked only when the generator's own scenes are on)."""
        c = getattr(self, "_built", {}).get("config") or _config(cfgd, self.agnostic)
        mix = dict(DEFAULT_MIX, **(c["mix"] or {}))
        generated = any(v > 0 for k, v in mix.items() if k != "public")
        S = t.train[2]
        if not self.agnostic and generated and len(S) >= 2000:
            seen = {TYPES[v - 1] for v in np.unique(S) if 0 < v <= K}
            missing = [x for x in TYPES if x not in seen]
            if missing:
                raise ValueError(f"types never planted in train: {missing}")
        empty = float((t.train[1] == 0).mean())
        if generated and not 0.02 <= empty <= 0.6:
            raise ValueError(f"windows with nothing to redact are {empty:.2f} of train; expected about `empty_frac`")

    def manifest(self, cfgd, t=None):
        b = getattr(self, "_built", {})
        c = b.get("config") or {}
        types = list(self.entities)
        out = dict(generator="piigen", policy="label-agnostic" if self.agnostic else "typed", types=types,
                   k=len(types), n_labels=1 + 4 * len(types), mix=dict(DEFAULT_MIX, **(c.get("mix") or {})),
                   empty_frac=c.get("empty_frac"), lists=c.get("lists"), lookalikes=c.get("lookalikes"),
                   public_train=c.get("public_train"), public_docs_per_split=b.get("public_docs_per_split"),
                   stats=b.get("stats"), resources=b.get("resources"))
        if t is not None:
            S = t.train[2]
            out["bytes_per_type_train"] = {types[v - 1]: int((S == v).sum()) for v in range(1, len(types) + 1)}
            out["ignored_bytes_train"] = int((S == IGNORE).sum())
            out["empty_windows_train"] = float((t.train[1] == 0).mean())
        return out

    def span_rival(self, select, test):
        """The rules (piigen/rules.py) with a floor on their scores chosen on `select` (lowest floor among the best),
        scored on `test` exactly like the model: micro span-F1 over (start, end, type), exact boundaries (every rule
        span is type PII under the label-agnostic policy)."""
        sync_resources(sentences=False)               # the rules read the name lists, never the sentences
        types, col = tuple(self.entities), self.agnostic
        Xs, _, Ss = select
        Xt, _, St = test
        cs, ct = [rule_candidates(x) for x in Xs], [rule_candidates(x) for x in Xt]
        gs, gt = [gold_spans(r, len(types)) for r in Ss], [gold_spans(r, len(types)) for r in St]
        best_u, best_f = 0.0, -1.0
        for u in RULE_SCORE_FLOORS:
            f = _score_rules(cs, gs, u, col)["span_f1"]
            if f > best_f:
                best_f, best_u = f, u
        r = _score_rules(ct, gt, best_u, col)
        acc = _metrics.new_acc()
        for (text, c), row, x in zip(ct, St, Xt):
            _metrics.score_doc(sorted(rule_spans(text, c, best_u, col)), spans_of(row), ignored_of(row), text=x.tobytes(),
                               acc=acc, k=len(types))
        s = _metrics.summary(acc, types=types)
        return dict(r, name="rules: patterns + checksums + context words + name lists + credential catalogue",
                    min_score=best_u, select_f1=round(best_f, 4), grid=list(RULE_SCORE_FLOORS),
                    test_f1_all_detections=_score_rules(ct, gt, 0.0, col)["span_f1"],
                    per_type_test={x: s[x]["exact_f1"] for x in types}, gold_per_type_test={x: s[x]["gold"] for x in types},
                    redaction_test=s.get("_redaction"))


@register
class PiiMask(PiiSpans):
    """The released policy: label-agnostic masking, one type `PII` (every personal-data label of the sources)."""
    data_format = "pii_mask"
    entities = AGNOSTIC_TYPES
    agnostic = True
