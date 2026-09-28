"""YAML manifests: loading and schema validation (hand-written, so that the only dependency is PyYAML).

Two kinds of manifest live in git:

    data/sources/*.yaml   kind: source   what a build may read (training material)
    data/exams/*.yaml     kind: exam     evaluation sets, grouped in `sets`; never training material

The schema is documented in data/README.md ("Manifest schema"). Every entry names one upstream artifact, pinned:

    type   required pin                          what it is
    git    pin.commit (40 hex)                   a repository at one commit (read from git objects, never checked out)
    url    pin.sha256 (64 hex)                   a file behind a URL (zip, tarball, jar, raw file...)
    deb    pin.version + pin.sha256              a Debian binary package (.deb)
    pypi   pin.version + pin.sha256              one distribution file (wheel or sdist) from PyPI
    hf     pin.revision (40 hex)                 a Hugging Face dataset at one revision, with the splits it uses
    local  files[].sha256 + size                 files with no public download (evaluation only, `--from-dir`)

A `url` entry may use `pin.verify: members` instead of `pin.sha256` when the container's digest is not published: then
every member it contributes is listed in `files` with its sha256, and that is what is verified.
"""
import os
import re

from .util import DataError, check_hex, sha256_file

SCHEMA = 1
TYPES = ("git", "url", "deb", "pypi", "hf", "local")
ROLES = ("test", "dev")
FINGERPRINT_MODES = ("files", "jsonl-text", "creddata")
ID_RE = re.compile(r"^[a-z0-9][a-z0-9._+-]*$")
SET_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
_SPDX_TOKEN = re.compile(r"^(?:LicenseRef-[A-Za-z0-9.-]+|[A-Za-z0-9][A-Za-z0-9.+-]*)$")
_SPDX_OPS = {"AND", "OR", "WITH"}
_ENTRY_KEYS = {
    "common": {"id", "type", "licence", "description", "notes", "identities", "take", "files", "group", "role_note"},
    "git": {"url", "pin", "repos", "partition"},
    "url": {"url", "pin", "mirrors", "format", "package", "size"},
    "deb": {"url", "pin", "mirrors", "package", "arch", "distro", "filename"},
    "pypi": {"url", "pin", "package", "filename"},
    "hf": {"dataset", "pin", "splits", "derived_from", "sampling"},
    "local": {"provenance"},
}
_TAKE_KEYS = {"include", "exclude", "max_file_bytes", "min_file_bytes", "members", "nested", "max_files", "group", "self",
              "groups", "blob_limit"}
_FILE_KEYS = {"path", "member", "name", "sha256", "size", "family", "note", "nested"}


def spdx_ok(expr):
    """A light check of an SPDX license expression (identifiers, LicenseRef-*, AND / OR / WITH, parentheses)."""
    if not isinstance(expr, str) or not expr.strip():
        return False
    tokens = expr.replace("(", " ( ").replace(")", " ) ").split()
    depth, expect_id = 0, True
    for t in tokens:
        if t == "(":
            if not expect_id:
                return False
            depth += 1
        elif t == ")":
            if expect_id or depth == 0:
                return False
            depth -= 1
        elif t in _SPDX_OPS:
            if expect_id:
                return False
            expect_id = True
        else:
            if not expect_id or not _SPDX_TOKEN.match(t):
                return False
            expect_id = False
    return depth == 0 and not expect_id


def _err(errors, where, msg):
    errors.append(f"{where}: {msg}")


def _check_files(errors, where, files, need_sha=True):
    if not isinstance(files, list) or not files:
        _err(errors, where, "`files` must be a non-empty list")
        return
    seen = set()
    for i, f in enumerate(files):
        w = f"{where}.files[{i}]"
        if not isinstance(f, dict):
            _err(errors, w, "must be a mapping")
            continue
        unknown = set(f) - _FILE_KEYS
        if unknown:
            _err(errors, w, f"unknown keys {sorted(unknown)}")
        if not any(k in f for k in ("path", "member", "name")):
            _err(errors, w, "needs one of path / member / name")
        if need_sha and not check_hex(f.get("sha256"), 64):
            _err(errors, w, "sha256 must be 64 lowercase hex characters")
        if "size" in f and (not isinstance(f["size"], int) or f["size"] < 0):
            _err(errors, w, "size must be a non-negative integer")
        key = f.get("path") or f.get("member") or f.get("name")
        if key in seen:
            _err(errors, w, f"duplicate file `{key}`")
        seen.add(key)


def _check_take(errors, where, take):
    if take is None:
        return
    if not isinstance(take, dict):
        _err(errors, where, "`take` must be a mapping")
        return
    unknown = set(take) - _TAKE_KEYS
    if unknown:
        _err(errors, where, f"unknown `take` keys {sorted(unknown)}")
    for k in ("include", "exclude", "members", "nested"):
        if k in take and not (isinstance(take[k], list) and all(isinstance(x, str) for x in take[k])):
            _err(errors, where, f"take.{k} must be a list of patterns")
    for k in ("max_file_bytes", "min_file_bytes", "max_files", "blob_limit"):
        if k in take and (not isinstance(take[k], int) or take[k] < 0):
            _err(errors, where, f"take.{k} must be a non-negative integer")


def validate_entry(e, where, kind):
    errors = []
    if not isinstance(e, dict):
        return [f"{where}: an entry must be a mapping"]
    eid, etype = e.get("id"), e.get("type")
    where = f"{where}[{eid}]"
    if not isinstance(eid, str) or not ID_RE.match(eid):
        _err(errors, where, f"id `{eid}` must match {ID_RE.pattern}")
    if etype not in TYPES:
        _err(errors, where, f"type `{etype}` must be one of {', '.join(TYPES)}")
        return errors
    unknown = set(e) - _ENTRY_KEYS["common"] - _ENTRY_KEYS[etype]
    if unknown:
        _err(errors, where, f"unknown keys {sorted(unknown)} for type {etype}")
    if not spdx_ok(e.get("licence")):
        _err(errors, where, f"licence `{e.get('licence')}` is not an SPDX expression (use LicenseRef-* when needed)")
    pin = e.get("pin") or {}
    if etype != "local" and not isinstance(e.get("pin"), dict):
        _err(errors, where, "`pin` is required")
    if etype in ("git", "url", "deb", "pypi"):
        url = e.get("url")
        if not isinstance(url, str) or not url.startswith("https://"):
            _err(errors, where, "url must be an https:// URL")
        for m in e.get("mirrors") or []:
            if not isinstance(m, str) or not m.startswith("https://"):
                _err(errors, where, f"mirror `{m}` must be an https:// URL")
    if etype == "git":
        if not check_hex(pin.get("commit"), 40):
            _err(errors, where, "pin.commit must be a full 40-character commit id")
        for i, r in enumerate(e.get("repos") or []):
            if not (isinstance(r, dict) and isinstance(r.get("url"), str) and check_hex(r.get("commit"), 40)):
                _err(errors, f"{where}.repos[{i}]", "needs url and a 40-character commit")
    elif etype == "url":
        if pin.get("verify") == "members":
            _check_files(errors, where, e.get("files"))
        elif not check_hex(pin.get("sha256"), 64):
            _err(errors, where, "pin.sha256 must be 64 lowercase hex characters (or pin.verify: members + files)")
        pkg = e.get("package")
        if pkg is not None and not (isinstance(pkg, dict) and {"ecosystem", "name", "version"} <= set(pkg)):
            _err(errors, where, "package must have ecosystem, name and version")
    elif etype in ("deb", "pypi"):
        if not isinstance(e.get("package"), str) or not e["package"]:
            _err(errors, where, "package is required")
        if not isinstance(pin.get("version"), str) or not pin["version"]:
            _err(errors, where, "pin.version is required")
        if not check_hex(pin.get("sha256"), 64):
            _err(errors, where, "pin.sha256 must be 64 lowercase hex characters")
        if etype == "deb" and not isinstance(e.get("arch"), str):
            _err(errors, where, "arch is required for a deb")
        if etype == "pypi" and not isinstance(e.get("filename"), str):
            _err(errors, where, "filename is required for a pypi entry")
    elif etype == "hf":
        if not isinstance(e.get("dataset"), str) or "/" not in e["dataset"]:
            _err(errors, where, "dataset must be `owner/name`")
        if not check_hex(pin.get("revision"), 40):
            _err(errors, where, "pin.revision must be a full 40-character revision")
        splits = e.get("splits")
        if not (isinstance(splits, list) and splits and all(isinstance(s, str) for s in splits)):
            _err(errors, where, "splits must be a non-empty list of split names")
        if "files" in e:
            _check_files(errors, where, e["files"], need_sha=False)
    elif etype == "local":
        _check_files(errors, where, e.get("files"))
        for f in e.get("files") or []:
            if isinstance(f, dict) and "size" not in f:
                _err(errors, where, f"local file `{f.get('name') or f.get('path')}` needs its size")
    if etype in ("url", "deb", "pypi") and "files" in e and pin.get("verify") != "members":
        _check_files(errors, where, e["files"])
    for s in e.get("identities") or []:
        if not (isinstance(s, str) and ":" in s):
            _err(errors, where, f"identity `{s}` must be `namespace:key`")
    _check_take(errors, where, e.get("take"))
    if kind == "source" and etype == "local":
        _err(errors, where, "a source cannot be `local`: training material must be downloadable by anyone")
    return errors


def validate(doc, where="manifest"):
    """All the problems of one manifest document, as readable strings (empty list: valid)."""
    errors = []
    if not isinstance(doc, dict):
        return [f"{where}: the document must be a mapping"]
    if doc.get("schema") != SCHEMA:
        _err(errors, where, f"schema must be {SCHEMA}")
    kind = doc.get("kind")
    if kind not in ("source", "exam"):
        _err(errors, where, "kind must be `source` or `exam`")
        return errors
    if not isinstance(doc.get("name"), str) or not SET_RE.match(doc["name"]):
        _err(errors, where, f"name must match {SET_RE.pattern}")
    if kind == "source":
        allowed = {"schema", "kind", "name", "description", "output", "builder", "build", "entries", "notes"}
        unknown = set(doc) - allowed
        if unknown:
            _err(errors, where, f"unknown top-level keys {sorted(unknown)}")
        entries = doc.get("entries")
        if not isinstance(entries, list) or not entries:
            _err(errors, where, "entries must be a non-empty list")
            return errors
        ids = [e.get("id") for e in entries if isinstance(e, dict)]
        dup = sorted({i for i in ids if ids.count(i) > 1})
        if dup:
            _err(errors, where, f"duplicate entry ids {dup}")
        for e in entries:
            errors += validate_entry(e, f"{where}.entries", kind)
    else:
        allowed = {"schema", "kind", "name", "description", "sets", "notes"}
        unknown = set(doc) - allowed
        if unknown:
            _err(errors, where, f"unknown top-level keys {sorted(unknown)}")
        sets = doc.get("sets")
        if not isinstance(sets, list) or not sets:
            _err(errors, where, "sets must be a non-empty list")
            return errors
        names = []
        for s in sets:
            if not isinstance(s, dict):
                _err(errors, where, "a set must be a mapping")
                continue
            sname = s.get("name")
            w = f"{where}.sets[{sname}]"
            names.append(sname)
            unknown = set(s) - {"name", "role", "description", "fingerprint", "entries", "usage", "licence_note"}
            if unknown:
                _err(errors, w, f"unknown keys {sorted(unknown)}")
            if not isinstance(sname, str) or not SET_RE.match(sname):
                _err(errors, w, f"set name must match {SET_RE.pattern}")
            if s.get("role") not in ROLES:
                _err(errors, w, f"role must be one of {', '.join(ROLES)}")
            if s.get("fingerprint", "files") not in FINGERPRINT_MODES:
                _err(errors, w, f"fingerprint must be one of {', '.join(FINGERPRINT_MODES)}")
            entries = s.get("entries")
            if not isinstance(entries, list) or not entries:
                _err(errors, w, "entries must be a non-empty list")
                continue
            ids = [e.get("id") for e in entries if isinstance(e, dict)]
            dup = sorted({i for i in ids if ids.count(i) > 1})
            if dup:
                _err(errors, w, f"duplicate entry ids {dup}")
            for e in entries:
                errors += validate_entry(e, f"{w}.entries", kind)
        dup = sorted({n for n in names if names.count(n) > 1})
        if dup:
            _err(errors, where, f"duplicate set names {dup}")
    return errors


class Manifest:
    """One validated manifest file. `sha256` is the digest of its raw bytes: fingerprints and builds record it, so a
    manifest edited after a build is detected."""

    def __init__(self, path, doc, sha256):
        self.path = os.path.abspath(path)
        self.doc = doc
        self.sha256 = sha256
        self.kind = doc["kind"]
        self.name = doc["name"]

    @property
    def entries(self):
        if self.kind == "source":
            return list(self.doc["entries"])
        return [e for s in self.doc["sets"] for e in s["entries"]]

    @property
    def sets(self):
        return list(self.doc.get("sets") or [])

    def entry(self, eid):
        for e in self.entries:
            if e["id"] == eid:
                return e
        raise DataError(f"{self.path}: no entry `{eid}`")

    def __repr__(self):
        return f"Manifest({self.kind}:{self.name}, {os.path.basename(self.path)})"


def load(path):
    import yaml
    with open(path, "rb") as f:
        raw = f.read()
    try:
        doc = yaml.safe_load(raw.decode("utf-8"))
    except yaml.YAMLError as e:
        raise DataError(f"{path}: not valid YAML ({e})") from None
    errors = validate(doc, os.path.basename(path))
    if errors:
        raise DataError(f"{path}: {len(errors)} schema problem(s):\n  " + "\n  ".join(errors[:40]))
    from .util import sha256_bytes
    return Manifest(path, doc, sha256_bytes(raw))


def from_doc(doc, name="<memory>.yaml"):
    """A validated Manifest from a document already in memory (tests, tools). Its sha256 is that of its JSON form."""
    import json
    errors = validate(doc, name)
    if errors:
        raise DataError(f"{name}: {len(errors)} schema problem(s):\n  " + "\n  ".join(errors[:40]))
    from .util import sha256_bytes
    return Manifest(name, doc, sha256_bytes(json.dumps(doc, sort_keys=True).encode("utf-8")))


def load_dir(directory):
    out = []
    if not os.path.isdir(directory):
        return out
    for name in sorted(os.listdir(directory)):
        if name.endswith((".yaml", ".yml")):
            out.append(load(os.path.join(directory, name)))
    return out


def load_sources():
    from .paths import SOURCE_MANIFESTS
    return load_dir(SOURCE_MANIFESTS)


def load_exams():
    from .paths import EXAM_MANIFESTS
    return load_dir(EXAM_MANIFESTS)


def exam_sets(exams=None):
    """[(set dict, manifest)] for every evaluation set declared in data/exams/*.yaml."""
    exams = load_exams() if exams is None else exams
    return [(s, m) for m in exams for s in m.sets]


def file_sha(path):
    return sha256_file(path)
