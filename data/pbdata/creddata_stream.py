"""The labelled real-window stream of secrets-code: windows of CredData DEV with every credential labelled per span.

`build_creddata_stream.py` is the command; this module is the recipe. The released secrets-code 1.0.0 was trained
with this stream (`train.real_windows`, `real_mode: spans`); with the default parameters the arrays come out bit for
bit the same (`RELEASED`; the conditions are in data/README.md).

WHAT GOES IN
- ONLY CredData DEV, and inside DEV only the `train` subset (`creddata.dev_subset`: sha1(id) % 100 < 70, 117 of the
  169 DEV repositories). The `select` and `heldout` subsets are where operating points are chosen and errors studied:
  no window comes from them. The openssl repository (cryptographic test vectors) is excluded by name as well.
- THE HARD CHECK (`assert_dev_train`) runs before a meta row is used and before a data file is opened: the repository
  is DEV in the split, not excluded, in the `train` subset, and the file lives under `data/<that repository>/`. The meta
  file of a TEST repository is never opened by this path. `check_file` repeats the check on the written file.
- THE TEST GUARD: a DEV credential whose value is also the value of a CredData TEST credential (a file vendored in a
  DEV and in a TEST repository) is not labelled and its windows are dropped. The TEST values are read by
  `test_value_hashes` alone, which returns keyed hashes (HMAC-SHA256 with a key drawn for the run and never stored).
- LABELS PER SPAN: the value (the meta's ValueStart:ValueEnd); a multi-line PEM block as one span; a URL credential
  widened to its whole URL (the synthetic generator's convention). A T row without a position: the value `line_form`
  finds on the line.
- UUID POLICY: a UUID -- or a credential whose only rules are UUID / Salt / Nonce -- is positive ONLY under an
  unequivocal credential name; otherwise it is ignored, and, with no "ignore" mask in the trainer, every window that
  touches it is dropped, like every window that touches an X unit.
- WINDOWS of `seq_len` bytes, in slots of four: two around a T span, one around an F unit without a T in the window (a
  hard negative), one random background window without a T. In a share `replant` of the T slots the value is
  RE-PLANTED with a generated value of the same form (the secrets generator: short password, Basic, JWT, bearer, OTP,
  NKEY, UUID, hex, provider format, generic) and the SAME hole gets its CONTRAST (a marker, an interpolation, an
  environment lookup, empty): two windows. The real values are repeated with window-offset augmentation only.

The .npz: X uint8 [N, T], S uint8 [N, T] (0/1 span mask), Y uint8 [N] (= S.any(1)), kind uint8 [N] (0 original T,
1 re-planted T, 2 contrast, 3 F, 4 background), repo int16 [N] (index into meta.repos), meta (a JSON string: counts,
repositories, parameters, the split's digest; no clock, no path). Nothing here prints or records a value.
"""
import collections
import contextlib
import hashlib
import hmac
import json
import os
import re
import sys
import warnings

import numpy as np

from . import creddata, paths
from .util import DataError

KINDS = ("original", "replanted", "contrast", "f_negative", "background")
EXCLUDED_REPOS = ("2ba83c6a",)        # openssl/openssl: cryptographic test vectors (a DEV `select` repository)
MAX_BYTES = 4_000_000                 # the text scanner's per-file limit: larger files are not read
RNG_SALT = 0xC4ED
DEFAULTS = dict(n=24000, seq_len=512, seed=0, replant=0.5)

# The stream secrets-code 1.0.0 was trained with, built on Linux with Python 3.14 and NumPy 2.5 from the pinned
# CredData snapshot: the split it was built on, how many of the generator's catalogue rules that Python compiles (the
# generator validates sampled provider values with them), and the sha256 of the raw bytes of every array. The file's
# own sha256 is not pinned: its `meta` member was written by the maintainers' tooling (data/README.md).
RELEASED = dict(
    params=dict(DEFAULTS),
    split_sha256="c2816b18c8046f73d4d97cfe39068ba9b65f231474d027e80960075404b23eca",
    catalog=dict(rules=120, compiled=99, sha256="d80e16f18d27b5ad29326bd6a2cde603a4ade6df9d87c063a99db70e17f54842"),
    arrays={
        "X": dict(dtype="|u1", shape=[24000, 512],
                  sha256="b747e70ab61da3258a510d9afdac0299d95d8c8b442dbcca9609314b4ff1eec0"),
        "S": dict(dtype="|u1", shape=[24000, 512],
                  sha256="0c17dc41afebeed6b4edeb621adf2f93ef34201183ba49ff5a0a4b2e9d5af735"),
        "Y": dict(dtype="|u1", shape=[24000],
                  sha256="2307fcff77978d54889c57d069f9a019ce0f268ff65a2a7488cfbed6396ba5f0"),
        "kind": dict(dtype="|u1", shape=[24000],
                     sha256="fd464ab5cd92b0621bfae1ff8b40728af1cb6be9bef078abc394e8fb58462726"),
        "repo": dict(dtype="<i2", shape=[24000],
                     sha256="01575e50dd545d08ead1e2fe1b600215267203219c0826e6cc6bd48c89a59817"),
    })

_UUID = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
# an UNEQUIVOCAL credential name ...
_CRED_NAME = re.compile(r"api[_-]?key|apikey|token|secret|passw|pwd|auth|licen[cs]e[_-]?key|access[_-]?key|"
                        r"private[_-]?key|credential|bearer|x-api-key|signing[_-]?key", re.I)
# ... and not an identifier's (`id`, `*_id`, `uuid`, `guid`, `_postman_id`, `publisherId`)
_ID_NAME = re.compile(r"(^|[_.\-])(id|uuid|guid)$|Id$|ID$|postman|uuid|guid", re.I)
_URL_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9+.\-]*://[^\s'\"`<>]+")


def credential_named(name):
    """The UUID policy: is `name` an unequivocal credential name (and not an identifier's)?"""
    return bool(name) and bool(_CRED_NAME.search(name)) and not _ID_NAME.search(name)


# ------------------------------------------------------------------------------------------------ the hard check
def assert_dev_train(repo, split, path=None):
    """THE hard check: anything but a repository of CredData DEV's `train` subset raises, before a byte is read."""
    if repo not in split:
        raise DataError(f"repository {repo!r} is not in the CredData split: another CredData snapshot? Refusing")
    if split[repo] != "dev":
        raise DataError(f"HARD CHECK: repository {repo!r} is CredData {split[repo]!r}, not DEV: nothing of TEST may be "
                        f"read")
    if repo in EXCLUDED_REPOS:
        raise DataError(f"HARD CHECK: repository {repo!r} is excluded")
    part = creddata.dev_subset(repo)
    if part != "train":
        raise DataError(f"HARD CHECK: repository {repo!r} is in DEV `{part}`; only DEV `train` feeds the stream")
    if path is not None and not path.startswith(f"data/{repo}/"):
        raise DataError(f"HARD CHECK: {path!r} does not live under data/{repo}/")


def dev_train_rows(meta_dir, split):
    """(meta rows, repositories) of DEV `train`. The meta file of any other repository is never opened (its name is
    its repository id); every row is checked before it is used."""
    rows, repos = [], []
    for name in sorted(os.listdir(meta_dir)):
        if not name.endswith(".csv"):
            continue
        repo = name[:-4]
        if repo not in split:
            raise DataError(f"meta file {name!r}: repository not in the CredData split; refusing to guess")
        if split[repo] != "dev" or repo in EXCLUDED_REPOS or creddata.dev_subset(repo) != "train":
            continue
        rs = creddata.read_meta(meta_dir, keep=lambda r, repo=repo: r == repo)
        for r in rs:
            if r["RepoName"] != repo:
                raise DataError(f"meta file {name!r} carries a row of repository {r['RepoName']!r}")
            assert_dev_train(repo, split, r["FilePath"])
        rows += rs
        repos.append(repo)
    return rows, repos


# ------------------------------------------------------------------------------------------------ the TEST guard
def keyed_hasher():
    """A keyed hash for the TEST guard; its key exists only in this process."""
    key = os.urandom(32)
    return lambda value: hmac.new(key, value.encode("utf-8", "surrogateescape"), hashlib.sha256).digest()


def test_value_hashes(root, split, keyed):
    """Keyed hashes of the values of every CredData TEST credential (T unit): what the DEV stream must not contain.
    The only reader of TEST in this module; it returns digests and counts, never a value."""
    meta_dir = os.path.join(root, "meta")
    unknown = sorted(n[:-4] for n in os.listdir(meta_dir) if n.endswith(".csv") and n[:-4] not in split)
    if unknown:
        raise DataError(f"meta files of repositories that are not in the split: {unknown[:5]}: another snapshot?")
    by_file = collections.defaultdict(list)
    for key, u in creddata.units(creddata.read_meta(meta_dir, keep=lambda r: split[r] == "test")).items():
        if u["label"] == "T":
            by_file[key[0]].append((key, u))
    hashes, missing = set(), []
    for f in sorted(by_file):
        p = creddata.data_path(root, f)
        if not os.path.exists(p):
            missing.append(f)
            continue
        with open(p, "rb") as fh:
            lines = fh.read().decode("utf-8", "replace").split("\n")
        for key, u in by_file[f]:
            hashes.add(keyed(creddata.line_form(lines, key, u["rows"])[1].strip()))
    if missing:
        raise DataError(f"{len(missing)} labelled CredData TEST file(s) are missing (first: {missing[0]}; an "
                        f"antivirus?): the TEST guard would be incomplete. Fetch CredData again")
    return hashes, dict(files=len(by_file), units=sum(len(v) for v in by_file.values()), distinct_values=len(hashes))


# ------------------------------------------------------------------------------------------------ spans of one file
def char_to_byte(data, line_start, n):
    """Column `n` in CHARACTERS of the line starting at byte `line_start` -> byte offset."""
    end = data.find(b"\n", line_start)
    end = len(data) if end < 0 else end
    return line_start + len(data[line_start:end].decode("utf-8", "replace")[:n].encode("utf-8", "replace"))


def _int(row, k):
    v = row.get(k)
    return int(float(v)) if v not in ("", None) else -1


def file_units(data, file_units_, keyed, test_hashes, macro, stats):
    """[(start, end, kind, info)] for one file. kind: T (a labelled span), F (a hard-negative centre), IGN (never
    touched by a window). `info` (T only): the form of the unit, for re-planting (name, quoted hole, categories,
    shape...). `file_units_`: [(meta key, unit)] of this file, in meta order."""
    ini = [0] + [i + 1 for i, c in enumerate(data) if c == 10]
    lines = data.decode("utf-8", "replace").split("\n")
    out = []
    for key, u in file_units_:
        _, ls, le = key
        if not (1 <= ls <= le <= len(ini)):
            continue
        a_l = ini[ls - 1]
        z_l = (ini[le] - 1) if le < len(ini) else len(data)
        if u["label"] == "X":
            out.append((a_l, z_l, "IGN", None))
            stats["X_units"] += 1
            continue
        if u["label"] == "F":
            out.append((a_l, z_l, "F", None))
            continue
        form, value = creddata.line_form(lines, key, u["rows"])
        if test_hashes is not None and keyed(value.strip()) in test_hashes:     # the TEST guard
            out.append((a_l, z_l, "IGN", None))
            stats["T_units_hash_in_TEST"] += 1
            continue
        cats = set(u["cats"])
        uuidish = bool(_UUID.match(value.strip())) or (cats and cats <= {"UUID", "Salt", "Nonce"})
        for r in u["rows"]:
            v0, v1 = _int(r, "ValueStart"), _int(r, "ValueEnd")
            if ls == le and v0 >= 0 and v1 > v0:
                a, z = char_to_byte(data, a_l, v0), char_to_byte(data, a_l, v1)
            elif le > ls:                                      # multi-line (PEM): the block
                a = char_to_byte(data, a_l, v0) if v0 >= 0 else a_l
                z = char_to_byte(data, ini[le - 1], v1) if v1 >= 0 else z_l
            else:                                              # no position: the value `line_form` finds on the line
                i = lines[ls - 1].find(value) if value else -1
                if i < 0:
                    out.append((a_l, z_l, "IGN", None))
                    stats["T_rows_unlocated"] += 1
                    continue
                a = a_l + len(lines[ls - 1][:i].encode("utf-8", "replace"))
                z = a + len(value.encode("utf-8", "replace"))
            z = min(z, len(data))
            if z <= a:
                out.append((a_l, z_l, "IGN", None))
                stats["T_rows_unlocated"] += 1
                continue
            if uuidish:                                        # the UUID policy
                if not credential_named(form["name"]):
                    out.append((a, z, "IGN", None))
                    stats["uuid_ignored_not_credential_name"] += 1
                    continue
                stats["uuid_positive_credential_name"] += 1
            la, lz = a, z                                      # the LABEL span (a URL widens to the whole URL)
            if ls == le and ("URL Credentials" in cats or form["syntax"] == "url_userinfo"):
                seg = data[a_l:z_l].decode("utf-8", "replace")
                for m in _URL_TOKEN.finditer(seg):
                    ma = a_l + len(seg[:m.start()].encode("utf-8", "replace"))
                    mz = a_l + len(seg[:m.end()].rstrip(").,;").encode("utf-8", "replace"))
                    if ma <= a and z <= mz:
                        la, lz = ma, mz
                        stats["url_widened"] += 1
                        break
            q = data[a - 1:a] if a > 0 else b""
            info = dict(cats=sorted(cats), syntax=form["syntax"], name_kind=form["name_kind"], shape=form["shape"],
                        n=form["n"], name=form["name"], quoted=q in (b'"', b"'", b"`"), multiline=le > ls,
                        hole=(a, z), macro=macro(u["cats"]))
            out.append((la, lz, "T", info))
    return out


# ------------------------------------------------------------------------------------------------ re-planting
def replant_value(rng, info, G, macro):
    """A generated value of the SAME form as the real one (never derived from it). `G`: the secrets generator."""
    cats, syn, nk, n, shape = set(info["cats"]), info["syntax"], info["name_kind"], info["n"], info["shape"]
    if "URL Credentials" in cats or syn == "url_userinfo":
        return G._password_g1(rng, 3, 12)
    if "Basic Authorization" in cats:
        return G._basic_b64(rng)
    if "JSON Web Token" in cats or shape == "jwt":
        return G._jwt_real(rng)
    if "Bearer Authorization" in cats or syn == "auth_header":
        return G._bearer_value(rng)
    if "OTP / 2FA Secret" in cats:
        return G._otp_secret(rng)
    if "NKEY Seed" in cats:
        return G._nkey_seed(rng)
    if shape == "uuid":
        return G._uuid(rng)
    if nk == "password" or "Password" in cats:
        return G._password_g1(rng) if n <= 16 else G._password_short(rng) if rng.integers(0, 2) else G._sec_generic(rng)
    if shape == "hex":
        return G._r(rng, G.HEX, max(8, min(int(n), 128)))
    if shape == "provider_prefix" or macro(info["cats"]) == "provider_pattern":
        v = G.catalog_secret(rng) if rng.integers(0, 2) else None
        return v or G._sec_prefix(rng)
    return G._sec_generic(rng)


def contrast_value(rng, info, G):
    """The NEGATIVE in the same hole: a marker, an interpolation / reference, or empty."""
    cats = set(info["cats"])
    N = re.sub(r"\W+", "_", info["name"] or "").strip("_").upper() or "SECRET"
    auth = "Basic Authorization" in cats or "Bearer Authorization" in cats or info["syntax"] == "auth_header"
    k = int(rng.integers(0, 4))
    if k == 0 or (k == 3 and info["quoted"]):
        return str(rng.choice(G._BEARER_FILL if auth else G._MARKERS_G1))
    if k == 1:
        return str(rng.choice([f"${{{N}}}", f"${N}", f"%({N.lower()})s", f"{{{{ {N.lower()} }}}}", f"${{env:{N}}}"]))
    if k == 2:
        return "" if info["quoted"] and not auth else str(rng.choice(G._MARKERS_G1))
    return str(rng.choice([f"os.getenv(\"{N}\")", f"process.env.{N}", f"ENV[\"{N}\"]", "cfg.password",
                           "settings.SECRET"]))


# ------------------------------------------------------------------------------------------------ windows
def _window(data, spans, w0, L):
    """(X row, S row) of [w0, w0 + L) of `data`, labelling every T span that overlaps (clipped). Padding: spaces."""
    x = np.full(L, 32, np.uint8)
    s = np.zeros(L, np.uint8)
    seg = np.frombuffer(data[w0:w0 + L], np.uint8)
    x[:len(seg)] = seg
    for a, z, kd, _ in spans:
        if kd == "T" and a < w0 + L and w0 < z:
            s[max(a, w0) - w0:min(z, w0 + L) - w0] = 1
    return x, s


def _touches(spans, w0, L, kinds):
    return any(kd in kinds and a < w0 + L and w0 < z for a, z, kd, _ in spans)


def _start_around(rng, a, z, L, nb):
    w0 = a - int(rng.integers(0, max(1, L - min(z - a, L - 1))))
    return max(0, min(w0, max(0, nb - L)))


def _replace(data, spans, idx, hole, new, label_new):
    """`data` with `hole` replaced by `new`; the spans moved. The span `idx` becomes the new value's label span when
    `label_new`, or disappears (the contrast). Another span overlapping the hole is never labelled again."""
    ha, hz = hole
    d = len(new) - (hz - ha)
    out_data = data[:ha] + new + data[hz:]
    out = []
    for i, (a, z, kd, info) in enumerate(spans):
        if i == idx:
            if label_new:
                out.append((a, z + d, kd, info))               # the label span contains the hole (URL: the URL)
            continue
        if z <= ha:
            out.append((a, z, kd, info))
        elif a >= hz:
            out.append((a + d, z + d, kd, info))
        else:
            out.append((a, z + d, "IGN", None))
    return out_data, out


def sample_windows(files, fspans, frepo, n, L, seed, replant, G, macro, stats):
    """The windows (see the module docstring). Returns (X, S, kind, repo, counts); the order of the random draws is
    part of the recipe: changing it changes every window after the change."""
    Tl = [(fi, si) for fi, sp in enumerate(fspans) for si, x in enumerate(sp) if x[2] == "T"]
    Fl = [(fi, si) for fi, sp in enumerate(fspans) for si, x in enumerate(sp) if x[2] == "F"]
    if not Tl:
        raise DataError("no labelled credential span in CredData DEV train: nothing to build")
    rng = np.random.default_rng([seed, RNG_SALT])
    X, S, K, R = [], [], [], []
    slot = attempts = 0
    positives, replanted = collections.Counter(), collections.Counter()
    while len(X) < n and attempts < n * 40:
        attempts += 1
        mode = slot % 4
        if mode < 2:
            fi, si = Tl[int(rng.integers(0, len(Tl)))]
            b, sp = files[fi], fspans[fi]
            a, z, _, info = sp[si]
            w0 = _start_around(rng, a, z, L, len(b))
            if _touches(sp, w0, L, ("IGN",)):
                stats["discarded_touches_X_or_ignored"] += 1
                continue
            ha, hz = info["hole"]
            can = (replant > 0 and not info["multiline"] and hz - ha <= 200 and a <= ha and hz <= z
                   and rng.random() < replant)
            if can and len(X) + 2 <= n:
                nv = replant_value(rng, info, G, macro).encode("utf-8", "replace")
                cv = contrast_value(rng, info, G).encode("utf-8", "replace")
                b1, sp1 = _replace(b, sp, si, (ha, hz), nv, True)
                b2, sp2 = _replace(b, sp, si, (ha, hz), cv, False)
                a1, z1 = sp1[si][0], sp1[si][1]
                w1 = w0 if z1 <= w0 + L else max(0, min(z1 - L + 8, a1))
                if _touches(sp1, w1, L, ("IGN",)) or _touches(sp2, w1, L, ("IGN",)):
                    stats["discarded_touches_X_or_ignored"] += 1
                    continue
                for bb, ss, kd in ((b1, sp1, 1), (b2, sp2, 2)):
                    x, s = _window(bb, ss, w1, L)
                    X.append(x)
                    S.append(s)
                    K.append(kd)
                    R.append(frepo[fi])
                replanted[info["macro"]] += 1
            else:
                x, s = _window(b, sp, w0, L)
                X.append(x)
                S.append(s)
                K.append(0)
                R.append(frepo[fi])
            positives[info["macro"]] += 1
        elif mode == 2:
            if not Fl:
                slot += 1
                continue
            fi, si = Fl[int(rng.integers(0, len(Fl)))]
            b, sp = files[fi], fspans[fi]
            a, z, _, _ = sp[si]
            w0 = _start_around(rng, a, z, L, len(b))
            if _touches(sp, w0, L, ("IGN", "T")):
                stats["discarded_f_window_with_T_or_X"] += 1
                continue
            x, s = _window(b, sp, w0, L)
            X.append(x)
            S.append(s)
            K.append(3)
            R.append(frepo[fi])
        else:
            fi = int(rng.integers(0, len(files)))
            b, sp = files[fi], fspans[fi]
            w0 = int(rng.integers(0, max(1, len(b) - L)))
            if _touches(sp, w0, L, ("IGN", "T")):
                stats["discarded_background_with_T_or_X"] += 1
                continue
            x, s = _window(b, sp, w0, L)
            X.append(x)
            S.append(s)
            K.append(4)
            R.append(frepo[fi])
        slot += 1
    X = np.stack(X)
    S = np.stack(S)
    K = np.array(K, np.uint8)
    R = np.array(R, np.int16)
    perm = rng.permutation(len(X))
    counts = dict(T_spans=len(Tl), F_units=len(Fl), attempts=attempts,
                  positives_by_macro=dict(sorted(positives.items())),
                  replanted_by_macro=dict(sorted(replanted.items())))
    return X[perm], S[perm], K[perm], R[perm], counts


# ------------------------------------------------------------------------------------------------ the generator
@contextlib.contextmanager
def generator_defaults(G):
    """The generator switch the re-planting reads (`INVENTED_PREFIXES`, via `_sec_prefix`) at its default for the
    build, restored afterwards; a held-out provider set is refused (the real windows may hold those providers)."""
    if getattr(G, "HOLDOUT", None):
        raise DataError("PUREBYTE_HOLDOUT is set: the stream is built with the generator's defaults and real windows "
                        "may hold any provider. Unset it and run this again")
    old = G.INVENTED_PREFIXES
    G.INVENTED_PREFIXES = 0.0
    try:
        yield G
    finally:
        G.INVENTED_PREFIXES = old


def load_generator(data_dir=None):
    """(credentials, catalog): the secrets generator (specialists/secrets-code/secretsgen) with the credential-format
    catalogues of `<data_dir or $PUREBYTE_DATA>/catalogs` loaded. Raises when they are missing: without them the
    re-planted provider values would come from a fallback, silently different."""
    src = os.path.join(paths.REPO, "specialists", "secrets-code")
    if src not in sys.path:
        sys.path.insert(0, src)
    from secretsgen import catalog as C
    from secretsgen import credentials as G
    folder = os.path.join(data_dir or paths.data_dir(), "catalogs")
    C._DIR = folder
    C._CAT = os.path.join(folder, "gitleaks.json")
    C._CAT_TH = os.path.join(folder, "trufflehog.json")
    C._MISSING.clear()
    C._COMPILED.clear()
    C.RULES, C.FPS_CURATED, C.WORDS = C._load()
    C.PROVIDERS, C.EXAMPLES_TH = C._load_th()
    ok, detail = C.catalogs_loaded()
    if not ok:
        raise DataError(f"the credential-format catalogues are missing ({', '.join(detail['missing'])}): run "
                        f"`python data/build_secret_catalogs.py` first")
    from secretsgen import standins
    made_up = [rel for rel in standins.in_place(data_dir or paths.data_dir()) if rel.startswith("catalogs/")]
    if made_up:
        raise DataError(f"the credential-format catalogues are the made-up stand-ins of data/build_smoke_data.py "
                        f"({', '.join(made_up)}): run `python data/build_secret_catalogs.py` first")
    return G, C


def catalog_signature(C):
    """Which of the generator's catalogue rules this Python compiles. The generator validates every sampled provider
    value with its rule's regex when the regex compiles, so a Python whose `re` accepts other rules samples other
    values, and every draw after them. With the pinned gitleaks catalogue that happens between 3.13 and 3.14 only:
    3.14 accepts `\\z` (3 of the 120 sampled rules), and the generator's `_compile` refuses a global flag after the
    start of a pattern on every version (3.10 alone would compile it), so 3.10 to 3.13 compile 96 rules and 3.14 99."""
    compiled = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for r in C.RULES:
            compiled.append(C._compile(r["regex"]) is not None)
    blob = json.dumps([[r.get("id"), r["regex"], c] for r, c in zip(C.RULES, compiled)], separators=(",", ":"))
    return dict(rules=len(C.RULES), compiled=sum(compiled), sha256=hashlib.sha256(blob.encode()).hexdigest())


# ------------------------------------------------------------------------------------------------ build and check
def build(root, split, G, macro, test_hashes, keyed, n=DEFAULTS["n"], seq_len=DEFAULTS["seq_len"],
          seed=DEFAULTS["seed"], replant=DEFAULTS["replant"], source=None):
    """({X, S, Y, kind, repo}, meta) from the CredData checkout at `root`. `split`: {repo: dev | test} (the pinned
    split); `test_hashes`: `test_value_hashes(...)` with the same `keyed`; `macro`: the scorer's `macro_category`;
    `source`: provenance written into the meta (dataset URL and commit)."""
    if not 0.0 <= replant <= 1.0:
        raise DataError(f"--replant {replant}: a share between 0 and 1")
    rows, repos = dev_train_rows(os.path.join(root, "meta"), split)
    per_file = collections.defaultdict(list)
    for key, u in creddata.units(rows).items():
        per_file[key[0]].append((key, u))
    stats = collections.Counter()
    files, fspans, frepo, missing = [], [], [], []
    ridx = {r: i for i, r in enumerate(repos)}
    for f in sorted(per_file):
        repo = f.split("/")[1]
        assert_dev_train(repo, split, f)                       # before the file is opened
        p = creddata.data_path(root, f)
        if not os.path.exists(p):
            missing.append(f)
            continue
        if not 0 < os.path.getsize(p) <= MAX_BYTES:
            stats["files_empty_or_over_4MB"] += 1
            continue
        with open(p, "rb") as fh:
            data = fh.read()
        fspans.append(file_units(data, per_file[f], keyed, test_hashes, macro, stats))
        files.append(data)
        frepo.append(ridx[repo])
    if missing:
        raise DataError(f"{len(missing)} labelled CredData DEV file(s) are missing (first: {missing[0]}; an "
                        f"antivirus?): the stream would differ from the released one. Fetch CredData again")
    with generator_defaults(G):
        X, S, K, R, counts = sample_windows(files, fspans, frepo, n, seq_len, seed, replant, G, macro, stats)
    Y = S.any(axis=1).astype(np.uint8)
    meta = dict(
        format="purebyte-real-spans/1", builder="data/build_creddata_stream.py", source=source or {},
        split=dict(seed=20260922, sha256=creddata.split_digest(split)),
        subset="CredData DEV, repositories of its `train` subset only (sha1(id) % 100 < 70)",
        exclude_repos=list(EXCLUDED_REPOS),
        uuid_policy="positive only under an unequivocal credential name; otherwise its windows are dropped",
        test_guard="DEV values equal to a CredData TEST credential are not labelled and their windows are dropped",
        n=n, seq_len=seq_len, seed=seed, replant=replant, repos=repos, repos_used=len({repos[i] for i in R.tolist()}),
        files=len(files), windows=int(len(X)), per_kind={KINDS[k]: int((K == k).sum()) for k in range(len(KINDS))},
        positive_windows=int(Y.sum()), counts=dict(counts, **{k: int(v) for k, v in sorted(stats.items())}))
    return dict(X=X, S=S, Y=Y, kind=K, repo=R), meta


def check_arrays(X, S, Y, K, R, n_repos):
    if not (X.dtype == np.uint8 and X.ndim == 2 and S.dtype == np.uint8 and S.shape == X.shape
            and bool(((S == 0) | (S == 1)).all()) and Y.shape == (len(X),)
            and np.array_equal(Y.astype(bool), S.any(1)) and len(K) == len(R) == len(X)
            and (len(R) == 0 or (int(R.max()) < n_repos and int(R.min()) >= 0))
            and (len(K) == 0 or int(K.max()) < len(KINDS))):
        raise DataError("HARD CHECK: incoherent arrays in the stream")


def check_file(path, split):
    """The hard check on a WRITTEN stream: built on this split, every repository it records is DEV `train`, the arrays
    are coherent. Returns its meta."""
    with np.load(path) as z:
        X, S, Y, K, R = z["X"], z["S"], z["Y"], z["kind"], z["repo"]
        meta = json.loads(str(z["meta"]))
    if (meta.get("split") or {}).get("sha256") != creddata.split_digest(split):
        raise DataError("HARD CHECK: the stream was built on another CredData split")
    for r in meta["repos"]:
        assert_dev_train(r, split)
    check_arrays(X, S, Y, K, R, len(meta["repos"]))
    return meta


def array_digests(arrays):
    """{name: {dtype, shape, sha256 of the raw bytes}}: what `RELEASED` pins (the .npz container is not compared)."""
    return {k: dict(dtype=a.dtype.str, shape=list(a.shape),
                    sha256=hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
            for k, a in arrays.items()}


def compare_with_released(digests, params, split, signature):
    """(identical, reasons): is this the stream secrets-code 1.0.0 was trained with, and if not, the likely why."""
    reasons = []
    if params != RELEASED["params"]:
        reasons.append(f"parameters {params} differ from the released {RELEASED['params']}")
    if creddata.split_digest(split) != RELEASED["split_sha256"]:
        reasons.append("the CredData split differs from the released one")
    if signature.get("sha256") != RELEASED["catalog"]["sha256"]:
        if signature.get("compiled") != RELEASED["catalog"]["compiled"]:
            reasons.append(f"this Python compiles {signature.get('compiled')} of the {signature.get('rules')} "
                           f"catalogue rules, the released stream was built with one that compiles "
                           f"{RELEASED['catalog']['compiled']} (Python 3.14): re-planted provider values differ, and "
                           f"so does every window drawn after the first of them")
        else:
            reasons.append("the credential-format catalogues differ from the ones the released stream was built with")
    same = all(digests.get(k, {}).get("sha256") == v["sha256"] and digests[k]["shape"] == v["shape"]
               and digests[k]["dtype"] == v["dtype"] for k, v in RELEASED["arrays"].items())
    if not same and not reasons:
        reasons.append("the inputs differ (CredData files, catalogues or generator)")
    return same, ([] if same else reasons)
