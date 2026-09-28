"""CredData (Samsung/CredData, Apache-2.0): the pinned checkout, its labelled files, its obfuscation and its split.

Shared by `fetch_creddata.py` (the evaluation sets) and `build_creddata_stream.py` (the labelled DEV stream).

    meta           the official meta files (`meta/<repo>.csv`): rows, labelled line units, the files each repo needs
    partial fetch  the labelled files of one upstream repository WITHOUT cloning it: its commit and trees only
                   (`--filter=blob:none`), the FileID of every path (sha256(path)[:8], as CredData's `download_data.py`
                   computes it), and then exactly the blobs whose FileID and official destination are in the meta
                   (plus the licence files CredData copies). Same files as `download_data.py`, which clones every
                   repository whole (tens of GB), for about 1 GB of downloads
    obfuscation    CredData's own `obfuscate_creds.py`, unmodified, run once over what was fetched. It replaces every
                   labelled credential value (seeded per row: the result is the same on every machine) and rewrites
                   the files with LF line endings; it must run on a POSIX system
    split          DEV / TEST by repository, the public scorer's (`benchmarks/creddata/score.py`), and the three DEV
                   subsets (`dev_subset`)
    line form      the value a labelled line holds and the name before it (`line_form`)

Privacy: nothing here prints, logs or returns a credential value, except `line_form`, whose value the callers only hash
or re-plant. The official script's own warnings quote values: its output is captured and never shown.
"""
import binascii
import collections
import csv
import hashlib
import importlib.util
import json
import os
import re
import shutil
import stat
import subprocess
import sys
import time

from . import fetch, manifest, paths
from .exams import _safe_relpath
from .util import DataError, ensure_dir, sha256_file, write_json

OBFUSCATION_STATE = "obfuscation.json"     # next to the checkout: running | done (the script is not idempotent)
FETCH_STATUS = "fetch-status.json"         # per repository: files wanted / written (a partial fetch resumes from it)
DEV_SUBSETS = (("train", 70), ("select", 85), ("heldout", 100))
TEXT_MODE_WRITES_CRLF = os.name == "nt"    # obfuscate_creds.py writes in text mode: CRLF line endings on Windows


# ------------------------------------------------------------------------------------------------ pins and split
def load_scorer():
    """benchmarks/creddata/score.py as a module: its `CredData` class computes the split, `macro_category` the family
    of a unit."""
    spec = importlib.util.spec_from_file_location("creddata_score", paths.CREDDATA_SCORER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def yaml_partitions():
    """(exam manifest, the CredData entry, {partition: {repo id: repo record}}) from data/exams/creddata.yaml."""
    m = next((m for m in manifest.load_exams() if m.name == "creddata"), None)
    if m is None:
        raise DataError("data/exams/creddata.yaml is missing")
    parts, entry = {}, None
    for s in m.sets:
        for e in s["entries"]:
            if e["id"] == "creddata":
                entry = e
                parts[e["partition"]] = {r["id"]: r for r in e["repos"]}
    return m, entry, parts


def pinned_split(parts):
    """{repo id: 'dev' | 'test'} as creddata.yaml lists it."""
    return {rid: part for part, repos in parts.items() for rid in repos}


def split_digest(split):
    """sha256 of a split {repo: partition}, independent of the file it was read from."""
    return hashlib.sha256(json.dumps(dict(sorted(split.items())), separators=(",", ":")).encode()).hexdigest()


def dev_subset(repo):
    """The part of CredData DEV a repository belongs to: `train` (sha1(id) % 100 < 70: the repositories the labelled
    stream is built from), `select` (70-84) or `heldout` (85-99). Operating points are chosen and errors studied on
    `select` and `heldout`, so no stream window ever comes from them."""
    h = int(hashlib.sha1(repo.encode()).hexdigest(), 16) % 100
    return next(name for name, top in DEV_SUBSETS if h < top)


def creddata_dir():
    return os.path.join(paths.exams_dir(), "creddata")


def checkout_dir():
    return os.path.join(creddata_dir(), "CredData")


# ------------------------------------------------------------------------------------------------ meta
def new_repo_id(snapshot_id):
    """The CredData repository id of a snapshot.json key (`download_data.get_new_repo_id`)."""
    return f"{binascii.crc32(binascii.unhexlify(snapshot_id)):08x}"


def read_meta(meta_dir, keep=None):
    """Rows of the official meta files, file by file in name order. `keep(repo id)` chooses which files are opened at
    all: the others are never read."""
    rows = []
    for name in sorted(os.listdir(meta_dir)):
        if not name.endswith(".csv") or (keep is not None and not keep(name[:-4])):
            continue
        with open(os.path.join(meta_dir, name), encoding="utf-8", newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def units(rows):
    """Labelled LINE units, keyed by the official meta key (FilePath, LineStart, LineEnd), in first-seen order. The
    label is T if any row on the key is T, otherwise F if any is F, otherwise X; `cats` and `rows` are those of the
    rows that carry the unit's label (the scorer's rule, plus the rows, whose value positions the stream needs)."""
    by = collections.defaultdict(list)
    for r in rows:
        by[(r["FilePath"], int(r["LineStart"]), int(r["LineEnd"]))].append(r)
    out = {}
    for key, rs in by.items():
        labels = {r["GroundTruth"] for r in rs}
        label = "T" if "T" in labels else ("F" if "F" in labels else "X")
        cats = sorted({c for r in rs if r["GroundTruth"] == label for c in r["Category"].split(":") if c})
        out[key] = dict(label=label, cats=cats, rows=[r for r in rs if r["GroundTruth"] == label],
                        repo=rs[0]["RepoName"])
    return out


def wanted_files(meta_dir):
    """{repo id: {FileID: FilePath}}: the labelled files every repository must deliver."""
    want = collections.defaultdict(dict)
    for r in read_meta(meta_dir):
        want[r["RepoName"]][r["FileID"]] = r["FilePath"]
    return dict(want)


def data_path(root, meta_path):
    """The file of a meta FilePath (`data/<repo>/...`) under the checkout, refusing anything that leaves it."""
    rel = _safe_relpath(meta_path)
    if rel is None or not meta_path.startswith("data/"):
        raise DataError(f"meta path {meta_path!r} is not a plain relative path under data/: refusing it")
    return os.path.join(root, rel)


# ------------------------------------------------------------------------------------------------ partial fetch
def file_scope(path_without_extension, words):
    """`download_data.get_file_scope`: the directory a file goes to, from the words of word_in_path.json."""
    result = "/"
    low = f"./{path_without_extension.lower()}"
    for word in words:
        if word in low:
            result += word[1:] if word.startswith("/") else word
            if not result.endswith("/"):
                result += "/"
    return "/_/" if result == "/" else result


def is_licence(first_component):
    """`download_data.collect_licenses`: the top-level entries CredData copies next to a repository's files."""
    return (any(k in first_component for k in ("LICEN", "Licen", "licen", "COPYING"))
            and "licensemanager" not in first_component)


def _rmtree(path):
    """shutil.rmtree that also removes the read-only files git leaves on Windows."""
    def retry(func, p, _exc):
        try:
            os.chmod(p, stat.S_IWRITE)
            func(p)
        except OSError:
            pass
    if sys.version_info >= (3, 12):
        shutil.rmtree(path, onexc=retry)
    else:
        shutil.rmtree(path, onerror=retry)


def _git(scratch, *args, input=None, check=True):
    return fetch.git("-c", "pack.threads=1", "-c", "index.threads=1", *args, cwd=scratch, input=input, check=check)


def _git_retry(scratch, args, input=None, attempts=3):
    last = b""
    for attempt in range(attempts):
        r = subprocess.run(["git", "-c", "pack.threads=1", "-c", "index.threads=1", *args], cwd=scratch, input=input,
                           stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=fetch._GIT_ENV)
        if r.returncode == 0:
            return None
        last = r.stderr
        time.sleep(5 * (attempt + 1))
    lines = last.decode("utf-8", "replace").strip().splitlines()
    return (lines[-1] if lines else "unknown error")[:200]


def select_blobs(tree_listing, repo_id, wanted, words):
    """From `git ls-tree -r -z --full-tree` output: {blob id: [destination paths]}, the FileIDs found, the number of
    licence files and the number of paths whose FileID matched but whose destination did not.

    A path is taken when its FileID is in the meta AND its official destination (`data/<repo><scope><FileID><ext>`) is
    the meta's FilePath; symbolic links, submodules, `.git/` paths and `.xml` files are skipped as CredData's
    `download_data.py` skips them. Licence files keep their path under `data/<repo>/` (`docs/mixes/LICENSE` becomes
    `data/<repo>/LICENSE`)."""
    sel, found, licences, mismatch = {}, set(), 0, 0
    for ent in tree_listing.split(b"\0"):
        if not ent:
            continue
        meta_part, path_b = ent.split(b"\t", 1)
        mode, otype, oid = meta_part.split()
        if otype != b"blob" or mode == b"120000":
            continue
        path = path_b.decode("utf-8", "surrogateescape")
        first = path.split("/", 1)[0]
        if is_licence(first) or path == "docs/mixes/LICENSE":
            sel.setdefault(oid.decode(), []).append(f"data/{repo_id}/{path if is_licence(first) else 'LICENSE'}")
            licences += 1
            continue
        if "/.git/" in path or path.endswith(".xml"):
            continue
        fid = hashlib.sha256(path_b).hexdigest()[:8]
        if fid not in wanted:
            continue
        stem, ext = os.path.splitext(path)
        dest = f"data/{repo_id}{file_scope(stem, words)}{fid}{ext.lower()}"
        if dest != wanted[fid]:
            mismatch += 1
            continue
        sel.setdefault(oid.decode(), []).append(dest)
        found.add(fid)
    return sel, found, licences, mismatch


def fetch_repository(root, snapshot_id, url, wanted, words, scratch_root):
    """Writes the labelled files (and licence files) of one upstream repository into `root/data/<repo id>/`.
    Returns a status record {repo, wanted, got, licences, mismatched, error}; never raises for a network failure (the
    caller decides), and always deletes its scratch repository."""
    repo_id = new_repo_id(snapshot_id)
    commit = snapshot_id[:40]
    st = dict(repo=repo_id, wanted=len(wanted), got=0, licences=0, mismatched=0, error="")
    scratch = os.path.join(scratch_root, repo_id)
    if os.path.exists(scratch):
        _rmtree(scratch)
    ensure_dir(scratch)
    try:
        _git(scratch, "init", "-q")
        _git(scratch, "remote", "add", "origin", url)
        err = _git_retry(scratch, ["-c", "protocol.version=2", "fetch", "-q", "--depth", "1", "--filter=blob:none",
                                   "--no-tags", "origin", commit])
        if err:
            st["error"] = f"fetching the commit: {err}"
            return st
        listing = _git(scratch, "ls-tree", "-r", "-z", "--full-tree", commit)
        sel, found, st["licences"], st["mismatched"] = select_blobs(listing, repo_id, wanted, words)
        oids = sorted(sel)
        for lo in range(0, len(oids), 2000):
            chunk = ("\n".join(oids[lo:lo + 2000]) + "\n").encode()
            err = _git_retry(scratch, ["-c", "fetch.negotiationAlgorithm=noop", "fetch", "-q", "origin", "--no-tags",
                                       "--no-write-fetch-head", "--recurse-submodules=no", "--filter=blob:none",
                                       "--stdin"], input=chunk)
            if err:
                st["error"] = f"fetching the files: {err}"
        with fetch.BlobReader(scratch) as reader:
            for oid in oids:
                try:
                    body = reader.read(oid)
                except DataError:
                    continue                                   # counted below as not delivered
                for dest in sel[oid]:
                    rel = _safe_relpath(dest)
                    if rel is None:
                        continue                               # a licence path this system cannot hold
                    full = os.path.join(root, rel)
                    ensure_dir(os.path.dirname(full))
                    with open(full, "wb") as f:
                        f.write(body)
        st["got"] = sum(1 for fid in found if os.path.exists(data_path(root, wanted[fid])))
    except (OSError, DataError) as e:
        st["error"] = f"{type(e).__name__}: {str(e)[:200]}"
    finally:
        _rmtree(scratch)
    return st


def missing_files(root, wanted):
    """{repo id: number of labelled files not on disk}."""
    out = {}
    for repo, files in wanted.items():
        n = sum(1 for p in files.values() if not os.path.exists(data_path(root, p)))
        if n:
            out[repo] = n
    return out


# ------------------------------------------------------------------------------------------------ obfuscation
def obfuscation_state(base):
    p = os.path.join(base, OBFUSCATION_STATE)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        return json.load(f)


SAME_VALUE = "The same value"               # obfuscate_creds.py logs "The same value: <value>, <meta row>"
_PROGRESS = re.compile(r"(Downloaded: \d+/\d+|Start download|Download finished|Download skipped|Finalizing dataset|"
                       r"Obfuscate \d+ files|Obfuscation was done|Done! All files saved)")


def run_captured(cmd, cwd, show=None):
    """Runs one of CredData's own scripts with everything it prints captured, so that no labelled value reaches the
    terminal or a log: `obfuscate_creds.py` logs "The same value: <value>, <meta row>" at WARNING level for every value
    it leaves unchanged, and `download_data.py` calls it at the end. Those lines are counted, never shown; the scripts'
    own progress lines (`_PROGRESS`) are passed to `show`, message only; everything else is kept back. Returns
    (exit status, lines quoting a value, the last lines that cannot hold a value, for an error message)."""
    import collections
    same, tail = 0, collections.deque(maxlen=5)
    p = subprocess.Popen(cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, errors="replace")
    for line in p.stdout:
        line = line.rstrip("\r\n")
        if SAME_VALUE in line:
            same += 1
            continue
        if show is not None and _PROGRESS.search(line):
            show("  CredData: " + line.rsplit(" | ", 1)[-1][:200])
        elif line.strip() and "value" not in line.lower():
            tail.append(line[:200])
    return p.wait(), same, list(tail)


def obfuscate(root, base, python=None):
    """Runs CredData's `obfuscate_creds.py` (unmodified) once over the fetched files. Refuses to run twice: the
    script replaces values in place, so a second run would change the data again."""
    state = obfuscation_state(base)
    if state and state.get("status") == "done":
        return state
    if state and state.get("status") == "running":
        raise DataError("an earlier obfuscation of CredData was interrupted: the files are in an unknown state. Remove "
                        f"{os.path.join(root, 'data')} and {os.path.join(base, FETCH_STATUS)}, then fetch again")
    if TEXT_MODE_WRITES_CRLF:
        raise DataError("CredData's obfuscate_creds.py writes files in text mode, so on Windows their line endings "
                        "would differ from everyone else's: run this step on Linux, macOS or inside WSL")
    python = python or sys.executable
    r = subprocess.run([python, "-c", "import base62"], capture_output=True)
    if r.returncode != 0:
        raise DataError("CredData's obfuscation script needs its own requirement `pybase62`: run "
                        f"`{python} -m pip install pybase62==1.0.0` (or install the checkout's requirements.txt), "
                        "then run this again")
    script = os.path.join(root, "obfuscate_creds.py")
    write_json(os.path.join(base, OBFUSCATION_STATE), dict(status="running", script_sha256=sha256_file(script)))
    # The script warns "The same value: <value>" when an obfuscated value equals the original: those lines are
    # counted, never shown. On failure only its lines that cannot hold a value are shown.
    code, same, tail = run_captured([python, "obfuscate_creds.py", "--meta_dir", "meta", "--data_dir", "data"], root)
    if code != 0:
        raise DataError(f"CredData's obfuscate_creds.py failed (exit {code}): " + " | ".join(tail))
    state = dict(status="done", script_sha256=sha256_file(script), values_left_identical=same)
    write_json(os.path.join(base, OBFUSCATION_STATE), state)
    return state


# ------------------------------------------------------------------------------------------------ line form
_CRED_WORD = [("password", re.compile(r"pass|pwd|passwd|(^|[^a-z])pw([^a-z]|$)", re.I)),
              ("secret", re.compile(r"secret", re.I)),
              ("token", re.compile(r"token|bearer|jwt|session", re.I)),
              ("key", re.compile(r"key", re.I)),
              ("auth_cred", re.compile(r"auth|cred|login|user", re.I))]
_NAME_BEFORE = re.compile(r"""(-{1,2}[A-Za-z][\w-]*|[A-Za-z_$@][\w.$\-]*)["'\]]?\s*(?::=|=>|==|=|:|\s)\s*[\[(]?\s*"""
                          r"""[bruf]?["'`]?\s*$""")
_ASSIGN = re.compile(r"""([A-Za-z_][\w.\-]*)["']?\s*(?::=|=>|=|:)\s*["'`]?([^"'`\s,;]{1,400})""")
_PLACEHOLDER = re.compile(r"(?i)x{3,}|\*{3,}|your|example|changeme|dummy|sample|fake|placeholder|<[^>]*>|\{\{|\$\{|%\(|"
                          r"%s|^\$[A-Za-z_]|^(null|none|true|false|nil|undefined)$|redacted|replace|todo|insert|enter")
_PREFIXES = ("AKIA", "ASIA", "ghp_", "gho_", "ghs_", "ghu_", "github_pat_", "xox", "sk_live", "sk_test", "pk_live",
             "pk_test", "rk_", "AIza", "ya29.", "glpat-", "SG.", "npm_", "sk-", "hooks.slack", "dckr_", "EAA", "shpat_",
             "AC", "SK", "whsec_")


def value_shape(v):
    """The coarse shape of a value (no value is kept): tiny, placeholder/template, reference/call, provider_prefix,
    uuid, jwt, hex, digits, word_like, b64_like, alnum or with_symbols."""
    if len(v) <= 3:
        return "tiny"
    if _PLACEHOLDER.search(v):
        return "placeholder/template"
    if "(" in v or re.fullmatch(r"[A-Za-z_]+(\.[A-Za-z_]+)+", v) or v.startswith(("$", "process.env", "os.")):
        return "reference/call"
    if v.startswith(_PREFIXES) and len(v) >= 16:
        return "provider_prefix"
    if re.fullmatch(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}", v):
        return "uuid"
    if v.startswith("eyJ"):
        return "jwt"
    if re.fullmatch(r"[0-9a-fA-F]+", v) and len(v) >= 8:
        return "hex"
    if re.fullmatch(r"[0-9]+", v):
        return "digits"
    if re.fullmatch(r"[A-Za-z]+[0-9]{0,4}[!.@#]?", v):
        return "word_like"
    if re.fullmatch(r"[A-Za-z0-9+/=_\-]+", v) and len(v) >= 16 and re.search(r"[A-Z]", v) and re.search(r"[a-z]", v) \
            and re.search(r"[0-9]", v):
        return "b64_like"
    if re.fullmatch(r"[A-Za-z0-9]+", v):
        return "alnum"
    return "with_symbols"


def line_form(lines, key, rows):
    """(form, value) of a labelled unit. `lines`: the file decoded as UTF-8 (errors replaced) and split on "\\n";
    `key`: the unit's (FilePath, LineStart, LineEnd); `rows`: its meta rows.

    The value is the meta's ValueStart:ValueEnd on a one-line unit; without a usable position, the value of the
    assignment on the line whose name looks like a credential's (else the first assignment, else the longest token);
    on a multi-line unit, the lines themselves. `form`: the name before the value (its last 32 characters), the kind
    of that name (password, secret, token, key, auth_cred, other_name, no_name), the syntax (pem_or_multiline,
    url_userinfo, auth_header, cli_flag, named_quoted, named_unquoted, bare_value), the value's shape and length."""
    ls, le = key[1], key[2]
    line = lines[ls - 1] if 1 <= ls <= len(lines) else ""
    vs = ve = -1
    for r in rows:
        a, z = r["ValueStart"], r["ValueEnd"]
        if a not in ("", None) and z not in ("", None) and int(float(a)) >= 0 and int(float(z)) > int(float(a)) \
                and ls == le:
            vs, ve = int(float(a)), int(float(z))
            break
    if vs < 0 or ve > len(line):
        best = None
        for m in _ASSIGN.finditer(line[:20000]):              # minified files: one line can be megabytes
            kind = next((n for n, rx in _CRED_WORD if rx.search(m.group(1))), None)
            if best is None or (kind and not best[0]):
                best = (kind, m.start(2), m.end(2))
        if best:
            vs, ve = best[1], best[2]
        else:
            toks = [(m.end() - m.start(), m.start(), m.end()) for m in re.finditer(r"\S+", line[:20000])]
            _, vs, ve = max(toks) if toks else (0, 0, 0)
    value = line[vs:ve] if ls == le else "\n".join(lines[ls - 1:le])
    before = line[max(0, vs - 300):vs]
    m = _NAME_BEFORE.search(before)
    name = m.group(1) if m else ""
    name_kind = next((n for n, rx in _CRED_WORD if rx.search(name)), "other_name") if name else "no_name"
    quoted = bool(re.search(r"[\"'`]\s*$", before))
    if le > ls or "PRIVATE KEY" in line or "BEGIN" in line:
        syntax = "pem_or_multiline"
    elif re.search(r"[a-z][a-z0-9+.\-]*://[^\s/@:]*:[^\s/@]+@", line):
        syntax = "url_userinfo"
    elif re.search(r"(?i)(authorization|bearer|basic)\b", before):
        syntax = "auth_header"
    elif name.startswith("-"):
        syntax = "cli_flag"
    elif name:
        syntax = "named_" + ("quoted" if quoted else "unquoted")
    else:
        syntax = "bare_value"
    stripped = value.strip()
    form = dict(name=name[-32:], name_kind=name_kind, syntax=syntax, shape=value_shape(stripped), n=len(stripped))
    return form, value
