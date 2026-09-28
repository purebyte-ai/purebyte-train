"""Evaluation sets on disk: fetching them (for evaluation) and reading their content (for fingerprints).

    $PUREBYTE_DATA/exams/<set>/<entry id>/...    git entries: the exact committed files (written from git objects)
    $PUREBYTE_DATA/exams/<set>/files/<name>      files taken from archives, downloads or a local copy (--from-dir)
    $PUREBYTE_DATA/exams/<set>/*.jsonl           document sets (PII benchmarks)
    $PUREBYTE_DATA/exams/<set>/manifest.json     what was fetched: pins, file names, sha256, what is missing

A file listed in a manifest with its sha256 is only ever accepted with that sha256, whatever its origin.
"""
import json
import os

from . import archives, fetch, paths
from .util import DataError, ensure_dir, log, sha256_bytes, sha256_file, write_json

WINDOWS_RESERVED = {"con", "prn", "aux", "nul"} | {f"com{i}" for i in range(1, 10)} | {f"lpt{i}" for i in range(1, 10)}


def _safe_relpath(path):
    """A relative path that can be written on this system, or None."""
    parts = path.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return None
    if os.name == "nt":
        for p in parts:
            if any(c in p for c in '<>:"\\|?*') or p.split(".")[0].lower() in WINDOWS_RESERVED or p.endswith((" ", ".")):
                return None
    return os.path.join(*parts)


def file_name(f):
    return f.get("name") or os.path.basename(f.get("member") or f.get("path"))


# --------------------------------------------------------------------------------------------------- local copies
class LocalIndex:
    """Files under a directory, indexed by sha256 on demand (by size first, so only candidates are hashed)."""

    def __init__(self, root):
        self.root = os.path.abspath(root)
        self.by_size = {}
        for dp, _, fns in os.walk(self.root):
            for fn in fns:
                p = os.path.join(dp, fn)
                try:
                    if os.path.isfile(p) and not os.path.islink(p):
                        self.by_size.setdefault(os.path.getsize(p), []).append(p)
                except OSError:
                    continue
        self._sha = {}

    def find(self, sha256, size=None):
        pool = self.by_size.get(size, []) if size is not None else [p for ps in self.by_size.values() for p in ps]
        for p in pool:
            if p not in self._sha:
                self._sha[p] = sha256_file(p)
            if self._sha[p] == sha256:
                return p
        return None


# --------------------------------------------------------------------------------------------------- fetching
def _write_file(dest, data):
    ensure_dir(os.path.dirname(dest))
    with open(dest, "wb") as f:
        f.write(data)


def fetch_git_entry(set_name, e, dry_run=False):
    repo = fetch.fetch_git(e["url"], e["pin"]["commit"], blob_limit=None, dry_run=dry_run)
    if dry_run:
        return dict(id=e["id"], planned=True)
    root = os.path.join(paths.exam_dir(set_name), e["id"])
    files = fetch.ls_tree(repo, e["pin"]["commit"])
    written, skipped, nbytes = 0, [], 0
    with fetch.BlobReader(repo) as r:
        for mode, sha, size, path in files:
            rel = _safe_relpath(path)
            if rel is None:
                skipped.append(path)
                continue
            dest = os.path.join(root, rel)
            if not (os.path.exists(dest) and os.path.getsize(dest) == size):
                _write_file(dest, r.read(sha))
            written += 1
            nbytes += size
    if skipped:
        log(f"    {len(skipped)} file name(s) cannot exist on this system and were not written "
            f"(they are still fingerprinted from git objects)")
    return dict(id=e["id"], url=e["url"], commit=e["pin"]["commit"], tree=fetch.tree_id(repo, e["pin"]["commit"]),
                files=written, bytes=nbytes, not_written=skipped[:50], not_written_count=len(skipped))


def fetch_file_entry(set_name, e, local=None, dry_run=False):
    """The listed members/files of a url / deb / pypi / local entry, verified by sha256, into exams/<set>/files/."""
    out_dir = os.path.join(paths.exam_dir(set_name), "files")
    wanted = e.get("files") or []
    got, missing = [], []
    todo = []
    for f in wanted:
        dest = os.path.join(out_dir, file_name(f))
        if os.path.exists(dest) and os.path.getsize(dest) == f.get("size", os.path.getsize(dest)) \
                and sha256_file(dest) == f["sha256"]:
            got.append(file_name(f))
        else:
            todo.append(f)
    if local is not None:
        for f in list(todo):
            p = local.find(f["sha256"], f.get("size"))
            if p:
                if not dry_run:
                    with open(p, "rb") as src:
                        _write_file(os.path.join(out_dir, file_name(f)), src.read())
                got.append(file_name(f))
                todo.remove(f)
    if todo and e["type"] != "local":
        if dry_run:
            fetch.fetch_file(e, dry_run=True)
        else:
            path = fetch.fetch_file(e)
            for f in todo:
                if f.get("member"):                       # a member of an archive
                    data = archives.read_member(path, f["member"], fmt=archives.entry_format(e))
                else:                                     # the downloaded file itself
                    with open(path, "rb") as fh:
                        data = fh.read()
                if sha256_bytes(data) != f["sha256"]:
                    raise DataError(f"{set_name}:{e['id']}: {f.get('member') or file_name(f)} has sha256 "
                                    f"{sha256_bytes(data)}, the manifest pins {f['sha256']}")
                _write_file(os.path.join(out_dir, file_name(f)), data)
                got.append(file_name(f))
            todo = []
    missing = [file_name(f) for f in todo]
    return dict(id=e["id"], type=e["type"], files=sorted(got), missing=missing)


# --------------------------------------------------------------------------------------------------- reading
def local_only_files(s):
    """`<entry id>:<file name>` of every file of a set that has no public download (`type: local`)."""
    return [f"{e['id']}:{file_name(f)}" for e in s["entries"] if e["type"] == "local" for f in e.get("files") or []]


def iter_set_content(s, missing, from_dir=None, allow_partial=False):
    """Yield (label, bytes) for every piece of content of an evaluation set, read from what `fetch_exams.py` (or
    `fetch_creddata.py`) put on disk and from `from_dir`: nothing is downloaded here. What is not available is appended
    to `missing`. At the end the generator raises when a file that HAS a public download is missing, unless
    `allow_partial`. Files without one (`type: local`) never make it raise: nobody but the holder of a copy can supply
    them, so their absence is only recorded (the fingerprint file says it is incomplete, and every build repeats it)."""
    mode = s.get("fingerprint", "files")
    local = LocalIndex(from_dir) if from_dir else None
    exam_root = paths.exam_dir(s["name"])
    for e in s["entries"]:
        if mode == "creddata":
            yield from _creddata_content(e)
            continue
        if e["type"] == "git":
            if not fetch.is_fetched(e["url"], e["pin"]["commit"]):
                raise DataError(f"set {s['name']}: the repository `{e['id']}` is not fetched (commit "
                                f"{e['pin']['commit'][:12]} is not in the cache): run "
                                f"`python data/fetch_exams.py --set {s['name']}` first")
            repo = fetch.repo_cache(e["url"])
            with fetch.BlobReader(repo) as r:
                for mode_, sha, size, path in fetch.ls_tree(repo, e["pin"]["commit"]):
                    yield f"{e['id']}/{path}", r.read(sha)
            continue
        if mode == "jsonl-text":
            for f in e.get("files") or []:
                p = os.path.join(exam_root, file_name(f))
                if not os.path.exists(p):
                    missing.append(f"{e['id']}:{file_name(f)}")
                    continue
                with open(p, encoding="utf-8") as fh:
                    for i, line in enumerate(fh):
                        line = line.strip()
                        if line:
                            text = json.loads(line).get("text") or ""
                            yield f"{e['id']}:{file_name(f)}:{i}", text.encode("utf-8")
            continue
        for f in e.get("files") or []:
            p = os.path.join(exam_root, "files", file_name(f))
            data = None
            if os.path.exists(p):
                with open(p, "rb") as fh:
                    data = fh.read()
            elif local is not None:
                q = local.find(f["sha256"], f.get("size"))
                if q:
                    with open(q, "rb") as fh:
                        data = fh.read()
            if data is None or sha256_bytes(data) != f["sha256"]:
                missing.append(f"{e['id']}:{file_name(f)}")
                continue
            yield f"{e['id']}:{file_name(f)}", data
    local_only = set(local_only_files(s))
    public = [m for m in missing if m not in local_only]
    if public and not allow_partial:
        raise DataError(f"set {s['name']}: {len(public)} file(s) that have a public download are not on disk: "
                        f"{', '.join(public[:10])}{' ...' if len(public) > 10 else ''}. Fetch them "
                        f"(`python data/fetch_exams.py --set {s['name']}`), point --from-dir at a copy, or pass "
                        f"--allow-partial (the fingerprint file then says which files it lacks, and no build excludes "
                        f"their content)")


def _creddata_content(e):
    """Every file of the CredData repositories of one partition (the obfuscated dataset files and their licences)."""
    root = os.path.join(paths.exams_dir(), "creddata", "CredData", "data")
    if not os.path.isdir(root):
        raise DataError("CredData is not downloaded: run `python data/fetch_creddata.py` first (Linux, macOS or WSL)")
    for r in sorted(e["repos"], key=lambda r: r["id"]):
        base = os.path.join(root, r["id"])
        if not os.path.isdir(base):
            continue                                   # a repository without labelled files
        for dp, dns, fns in os.walk(base):
            dns.sort()
            for fn in sorted(fns):
                p = os.path.join(dp, fn)
                with open(p, "rb") as fh:
                    yield os.path.relpath(p, root).replace("\\", "/"), fh.read()


def write_set_manifest(set_name, record):
    write_json(os.path.join(paths.exam_dir(set_name), "manifest.json"), record)
