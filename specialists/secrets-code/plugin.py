"""Vertical `secrets_synthetic`: synthetic credential lines injected into windows of REAL context bytes.

Used by secrets-code (context: source code and configuration) and by secrets-bin (context: binaries; with
`nul_separator`, `bare_value`, `long_hex`, `binary_symbols`). Every key of the task's `data:` block is read HERE (the
factory fails on a declared key that nobody read):

    n_train, n_eval, seq_len, split_policy.seed, version (v4), path (the corpus; read by the factory)
    extra_lines          lines per injected block (0 = one line per window)
    real_neighbors       true | no_pem: half of the negative neighbours of a block are REAL corpus lines
    nul_separator        binaries: 70 % of the blocks embedded between NULs (a C string table)
    bare_value           binaries: sometimes the value alone, when it is decidable by itself
    mined_negatives      {path, fraction}: real windows a released model flagged falsely, as train negatives
    and the generator families (DATA_KEYS below): proportions or switches, 0 / false = not drawn at all.
"""
import contextlib
import hashlib
import os
import re

import numpy as np
from pbtrain import paths
from pbtrain.eval.spans import _prf
from pbtrain.model.decode import mask_spans
from pbtrain.verticals import Partitions, Vertical, register

from secretsgen import catalog as _catalog
from secretsgen import credentials as _cred
from secretsgen import standins as _standins
from secretsgen.task import SecretsTask

# data key -> (generator switch, kind, default). kind: bool | prob (0..0.5) | g (0..0.25) | int | ...
DATA_KEYS = {
    "documented_examples": ("DOCUMENTED_EXAMPLES", "bool", False),      # AKIA...EXAMPLE-style values as negatives
    "web_negatives": ("WEB_NEGATIVES", "bool", False),                  # web tokens that are not secrets
    "shared_hosts": ("SHARED_HOSTS", "bool", False),                    # connection strings to shared/public hosts
    "bodyless_pem": ("BODYLESS_PEM", "bool", False),                    # PEM headers without a body (negatives)
    "long_hex": ("LONG_HEX", "bool", False),                            # long hex digests (binaries)
    "binary_symbols": ("BINARY_SYMBOLS", "bool", False),                # symbol-table-like strings (binaries)
    "short_passwords": ("SHORT_PASSWORDS", "prob", 0.0),                # 8-16 char passwords under any name
    "ordinary_short_values": ("ORDINARY_SHORT_VALUES", "ordinary", False),
    "no_equals_sign": ("NO_EQUALS_SIGN", "prob", 0.0),
    "reinforced_forms": ("REINFORCED_FORMS", "prob", 0.0),
    "reinforced_forms_tuple": ("REINFORCED_FORMS_TUPLE", "bool", True),
    "weak_passwords": ("WEAK_PASSWORDS", "prob", 0.0),
    "connection_config": ("CONNECTION_CONFIG", "prob", 0.0),            # proportion of connection-config lines
    "non_assignment_weight": ("NON_ASSIGNMENT_WEIGHT", "int_or_none", None),
    "nested_weight": ("NESTED_WEIGHT", "int", 0),
    "invented_prefixes": ("INVENTED_PREFIXES", "invented", 0.0),        # made-up provider prefixes
    # the real-data recipe families (proportions of lines, 0 = off)
    "password_named_short": ("PASSWORD_NAMED_SHORT", "g", 0.0),         # short passwords under a password name
    "auth_header": ("AUTH_HEADER", "g", 0.0),                           # Authorization: Basic / Bearer
    "url_short_password": ("URL_SHORT_PASSWORD", "g", 0.0),             # scheme://user:short@host
    "unnamed_values": ("UNNAMED_VALUES", "g", 0.0),                     # OTP seeds, tokens passed as arguments
    "dotnet_resources": ("DOTNET_RESOURCES", "g", 0.0),                 # PublicKeyToken, resource strings (negatives)
}


def _parse(key, kind, raw):
    if kind == "bool":
        if raw not in (True, False):
            raise ValueError(f"`{key}: {raw!r}` must be true or false")
        return bool(raw)
    if kind in ("prob", "g"):
        x = float(raw or 0.0)
        top = 0.5 if kind == "prob" else 0.25
        if not 0.0 <= x <= top:
            raise ValueError(f"`{key}: {x}` is a proportion of lines between 0 and {top}")
        return x
    if kind == "int":
        return int(raw or 0)
    if kind == "int_or_none":
        return None if raw is None else int(raw)
    if kind == "ordinary":
        return "separate" if raw == "separate" else bool(raw)
    if kind == "invented":
        x = 0.3 if raw is True else float(raw or 0.0)
        if not 0.0 <= x <= 0.9:
            raise ValueError(f"`{key}: {raw}` is not a proportion between 0 and 0.9")
        return x
    raise AssertionError(kind)


def reload_catalogs(data_dir=None):
    """(Re)loads the catalogues from `<data_dir or $PUREBYTE_DATA>/catalogs` into the generator (they are read once at
    import; a process that changes PUREBYTE_DATA calls this, and `build` does it when the directory changed)."""
    folder = os.path.join(data_dir or paths.data_dir(), "catalogs")
    _catalog._DIR = folder
    _catalog._CAT = os.path.join(folder, "gitleaks.json")
    _catalog._CAT_TH = os.path.join(folder, "trufflehog.json")
    _catalog._MISSING.clear()
    _catalog._COMPILED.clear()
    _catalog.RULES, _catalog.FPS_CURATED, _catalog.WORDS = _catalog._load()
    _catalog.PROVIDERS, _catalog.EXAMPLES_TH = _catalog._load_th()
    return _catalog.catalogs_loaded()


@contextlib.contextmanager
def _switches(values):
    """Sets the generator's module switches for one build and restores them afterwards (no cross-talk)."""
    old = {glob: getattr(_cred, glob) for glob, _, _ in DATA_KEYS.values()}
    try:
        for k, (glob, _, _) in DATA_KEYS.items():
            setattr(_cred, glob, values[k])
        yield
    finally:
        for glob, v in old.items():
            setattr(_cred, glob, v)


# the classic rival: what a competent human would write -- a long quoted literal, not a known placeholder, with high
# entropy. The entropy threshold is tuned on `select`, like the model's operating point.
PLACEHOLDERS = re.compile(
    rb"YOUR_|_HERE|CHANGE|changeme|xxxx|TODO|REDACTED|\*\*\*|not-a-real|replace-with|"
    rb"dummy|example|test|placeholder|INSERT_|^0+$|^a+$|abcdef0123456789", re.I)
LITERAL = re.compile(rb"""["']([^"'\n]{12,90})["']""")


def entropy(b):
    if not b:
        return 0.0
    c = np.bincount(np.frombuffer(b, dtype=np.uint8), minlength=256)
    p = c[c > 0] / len(b)
    return float(-(p * np.log2(p)).sum())


def rule_spans(X, S, threshold):
    tp = fp = fn = 0
    for i in range(len(X)):
        b = X[i].tobytes()
        pred = {(m.start(1), m.end(1)) for m in LITERAL.finditer(b)
                if not PLACEHOLDERS.search(m.group(1)) and entropy(m.group(1)) >= threshold}
        gold = {(a, z) for a, z, _ in mask_spans(S[i], 1)}
        tp += len(pred & gold)
        fp += len(pred - gold)
        fn += len(gold - pred)
    return _prf(tp, fp, fn)


@register
class SyntheticSecrets(Vertical):
    data_format = "secrets_synthetic"
    needs_corpus = True
    overlap_guard = "chunks"              # windows are cut from real bytes: no 120-byte chunk shared train/test
    entities = ("secret",)
    checks_contract = False               # Y = "a positive line was injected"; its span may fall outside the window

    def _values(self, cfgd):
        return {k: _parse(k, kind, cfgd.get(k, default)) for k, (_, kind, default) in DATA_KEYS.items()}

    def build(self, corpus, cfgd, fracs):
        values = self._values(cfgd)
        if os.path.normcase(os.path.abspath(_catalog._DIR)) != os.path.normcase(
                os.path.abspath(os.path.join(paths.data_dir(), "catalogs"))):
            reload_catalogs()
        ok, detail = _catalog.catalogs_loaded()
        if not ok:
            raise ValueError(f"missing catalogues of real credential formats {detail['missing']}: build them with "
                             f"data/build_secret_catalogs.py (without them the generator would fall back to a few "
                             f"hand-written prefixes and train on much worse data)")
        if cfgd.get("hard", True) is not True:
            raise ValueError("`hard: false` selects a generator that is not built; drop the key")
        made_up = _standins.check(paths.data_dir(), ("catalogs/",), getattr(corpus, "path", None),
                                  "the credential-format catalogues", "python data/build_secret_catalogs.py")
        if made_up:                               # the smoke recipes on made-up catalogues: allowed, and recorded
            detail = dict(detail, smoke_standins=made_up)
        mined = cfgd.get("mined_negatives")
        mined = (paths.expand(mined["path"]), float(mined["fraction"])) if mined else None
        if mined and not 0.0 < mined[1] <= 0.5:
            raise ValueError(f"`mined_negatives.fraction: {mined[1]}` must be in (0, 0.5]")
        rn = cfgd.get("real_neighbors", False)
        with _switches(values):
            t = SecretsTask(corpus, n_train=int(cfgd["n_train"]), n_eval=int(cfgd["n_eval"]),
                            seq_len=int(cfgd["seq_len"]), seed=int(cfgd["split_policy"]["seed"]),
                            version=cfgd.get("version", "v4"), fracs=fracs, extra_lines=int(cfgd.get("extra_lines", 0)),
                            nul_separator=bool(cfgd.get("nul_separator", False)),
                            real_neighbors=("no_pem" if rn == "no_pem" else bool(rn)),
                            bare_value=bool(cfgd.get("bare_value", False)), mined_negatives=mined)
        self._built = dict(values=values, catalogs=detail, mined_used=t.mined_used,
                           mined_sha=(hashlib.sha256(open(mined[0], "rb").read()).hexdigest()[:16] if mined else None))
        for part in (t.train, t.select, t.test):
            X, Y, S = part
            if X.dtype != np.uint8 or S.shape != X.shape or Y.shape != (len(X),):
                raise ValueError(f"generator contract broken: X{X.shape} {X.dtype} Y{Y.shape} S{S.shape}")
        return Partitions(t.train, t.select, t.test)

    def guards(self, cfgd, t):
        """With held-out providers (PUREBYTE_HOLDOUT), their prefix followed by something value-like must not appear
        even once in train or select: otherwise the held-out measurement would not be valid."""
        if not _cred.HOLDOUT:
            return
        pat = re.compile(b"|".join(
            b"(?:" + re.escape(lit.encode()) + rb"(?=[A-Za-z0-9\-\.\*<]{8,})(?=[a-z\-]{0,11}[0-9A-Z\*\.<]))"
            for lit in _cred._LITERALS))
        leaks = sum(len(pat.findall(r.tobytes())) for A in (t.train[0], t.select[0]) for r in A)
        if leaks:
            raise ValueError(f"{leaks} occurrences of HELD-OUT providers in the data: the held-out measurement would "
                             f"not be valid")

    def manifest(self, cfgd, t=None):
        b = getattr(self, "_built", {})
        return dict(generator=f"secrets_synthetic {cfgd.get('version', 'v4')}", switches=b.get("values"),
                    catalogs=b.get("catalogs"), holdout=sorted(_cred.HOLDOUT),
                    extra_lines=int(cfgd.get("extra_lines", 0)), real_neighbors=cfgd.get("real_neighbors", False),
                    nul_separator=bool(cfgd.get("nul_separator", False)), bare_value=bool(cfgd.get("bare_value", False)),
                    mined_negatives=(dict(used=b.get("mined_used"), sha=b.get("mined_sha"))
                                     if cfgd.get("mined_negatives") else None))

    def span_rival(self, select, test):
        """The entropy rule over quoted literals; its threshold chosen on select, reported on test."""
        Xs, _, Ss = select
        Xt, _, St = test
        best_u, best_f = 3.2, -1.0
        for u in (2.4, 2.8, 3.2, 3.6, 4.0, 4.4):
            f = rule_spans(Xs, Ss, u)["span_f1"]
            if f > best_f:
                best_f, best_u = f, u
        r = rule_spans(Xt, St, best_u)
        return dict(r, name="entropy rule on quoted literals", entropy_threshold=best_u, select_f1=round(best_f, 4))
