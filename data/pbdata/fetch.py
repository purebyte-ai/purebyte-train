"""Downloads and git objects, verified before use. Everything lands under $PUREBYTE_DATA/cache:

    cache/blobs/<aa>/<sha256>          files pinned by sha256 (content-addressed: a second manifest pinning the same
                                       file reuses it; a corrupt file is detected by `--verify` and re-downloaded)
    cache/blobs/members-verified/...   files whose container digest is not published (`pin.verify: members`)
    cache/git/<host>/<owner>/<name>.git  bare repositories holding exactly the pinned commits (depth 1)

Repositories are never checked out for building: files are read from git objects (`ls-tree` + `cat-file`), so the bytes
are the committed bytes on every platform (no line-ending conversion, no invalid file names, no symlink surprises).
"""
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.request

from . import paths
from .identity import norm_repo
from .util import DataError, ensure_dir, human, log, order_key, sha256_file

USER_AGENT = "purebyte-data/1 (+https://github.com/purebyte-ai/purebyte-train)"
_GIT_ENV = dict(os.environ, GIT_TERMINAL_PROMPT="0", GIT_ASKPASS="echo", LC_ALL="C")  # purebyte:allow


# --------------------------------------------------------------------------------------------------- http
def http_get(url, dest, expected_sha256=None, expected_size=None, retries=4, timeout=600):
    """Stream `url` to `dest`, verifying size and sha256 on the fly. The file only appears at `dest` when verified."""
    import hashlib
    ensure_dir(os.path.dirname(os.path.abspath(dest)))
    part = dest + ".part"
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            h, n = hashlib.sha256(), 0
            with urllib.request.urlopen(req, timeout=timeout) as r, open(part, "wb") as out:
                while True:
                    block = r.read(1 << 20)
                    if not block:
                        break
                    out.write(block)
                    h.update(block)
                    n += len(block)
            if expected_size is not None and n != expected_size:
                raise DataError(f"{url}: {n} bytes, expected {expected_size}")
            if expected_sha256 and h.hexdigest() != expected_sha256:
                raise DataError(f"{url}: sha256 {h.hexdigest()} does not match the pin {expected_sha256}. The upstream "
                                f"file changed or the download is corrupt; never use it unverified.")
            os.replace(part, dest)
            return dest
        except DataError:
            if os.path.exists(part):
                os.remove(part)
            raise
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            last = e
            if os.path.exists(part):
                os.remove(part)
            time.sleep(3 + 5 * attempt)
    raise DataError(f"{url}: download failed after {retries} attempts ({last})")


def http_json(url, retries=4, timeout=120):
    import json
    last = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read().decode("utf-8"))
        except (urllib.error.URLError, OSError, TimeoutError, ValueError) as e:
            last = e
            time.sleep(3 + 5 * attempt)
    raise DataError(f"{url}: request failed after {retries} attempts ({last})")


def blob_path(sha256):
    return os.path.join(paths.cache_dir(), "blobs", sha256[:2], sha256)


def unpinned_path(url):
    base = os.path.basename(url.split("?")[0]) or "download"
    return os.path.join(paths.cache_dir(), "blobs", "members-verified", order_key(url)[:16] + "-" + base)


def urls_of(entry):
    return [entry["url"]] + list(entry.get("mirrors") or [])


def fetch_file(entry, dry_run=False, verify=False):
    """The cached, verified path of a url / deb / pypi entry (downloaded when missing)."""
    pin = entry.get("pin") or {}
    sha = pin.get("sha256")
    dest = blob_path(sha) if sha else unpinned_path(entry["url"])
    size = pin.get("size", entry.get("size"))
    if os.path.exists(dest):
        if verify and sha and sha256_file(dest) != sha:
            log(f"  {entry['id']}: cached file is corrupt, downloading again")
            os.remove(dest)
        else:
            return dest
    if dry_run:
        log(f"  would download {entry['url']}" + (f" ({human(size)})" if size else "") + f" -> {paths.rel(dest)}")
        return None
    errors = []
    for url in urls_of(entry):
        try:
            log(f"  downloading {url}")
            return http_get(url, dest, expected_sha256=sha, expected_size=size)
        except DataError as e:
            errors.append(str(e))
    raise DataError(f"{entry['id']}: no URL gave the pinned file:\n  " + "\n  ".join(errors))


# --------------------------------------------------------------------------------------------------- git
def git(*args, cwd=None, input=None, check=True):
    r = subprocess.run(["git", *args], cwd=cwd, input=input, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env=_GIT_ENV)
    if check and r.returncode != 0:
        raise DataError(f"git {' '.join(args[:3])}...: {r.stderr.decode('utf-8', 'replace').strip()[:500]}")
    return r.stdout


def git_available():
    return shutil.which("git") is not None


def repo_cache(url):
    return os.path.join(paths.cache_dir(), "git", *norm_repo(url).split("/")) + ".git"


def has_commit(repo, commit):
    if not os.path.isdir(repo):
        return False
    r = subprocess.run(["git", "cat-file", "-t", commit], cwd=repo, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                       env=_GIT_ENV)
    return r.returncode == 0 and r.stdout.strip() == b"commit"


def is_fetched(url, commit, blob_limit=None):
    """True when the cache already holds `commit` of `url` with every file up to `blob_limit` (None: every file), so
    that reading it downloads nothing."""
    repo = repo_cache(url)
    marker = os.path.join(repo, f"pb-fetched-{commit}")
    if not (has_commit(repo, commit) and os.path.exists(marker)):
        return False
    with open(marker, encoding="utf-8") as f:
        have = f.read().strip()
    return have == "all" or (blob_limit is not None and have.isdigit() and int(have) >= blob_limit)


def fetch_git(url, commit, blob_limit=None, dry_run=False):
    """A bare repository in the cache holding `commit` (depth 1). `blob_limit` (bytes) leaves larger files out
    (a partial clone): builds never read them, so they are never downloaded."""
    repo = repo_cache(url)
    if is_fetched(url, commit, blob_limit):
        return repo
    if dry_run:
        log(f"  would fetch {url} @ {commit[:12]}" + (f" (files <= {human(blob_limit)})" if blob_limit else ""))
        return None
    if not git_available():
        raise DataError("git is required (https://git-scm.com)")
    if not os.path.isdir(repo):
        ensure_dir(os.path.dirname(repo))
        git("init", "--bare", "-q", repo)
        git("remote", "add", "origin", url, cwd=repo)
    if blob_limit:
        for k, v in (("core.repositoryformatversion", "1"), ("extensions.partialClone", "origin"),
                     ("remote.origin.promisor", "true"),
                     ("remote.origin.partialclonefilter", f"blob:limit={blob_limit}")):
            git("config", k, v, cwd=repo)
    args = ["-c", "protocol.version=2", "fetch", "-q", "--depth", "1", "--no-tags"]
    if blob_limit:
        args.append(f"--filter=blob:limit={blob_limit}")
    log(f"  fetching {url} @ {commit[:12]}")
    git(*args, "origin", commit, cwd=repo)
    if not has_commit(repo, commit):
        raise DataError(f"{url}: commit {commit} not found after fetching (was it rewritten upstream?)")
    with open(os.path.join(repo, f"pb-fetched-{commit}"), "w", encoding="utf-8") as f:
        f.write(str(blob_limit) if blob_limit else "all")
    return repo


def ls_tree(repo, commit):
    """[(mode, sha, size, path)] of the regular files of `commit`, sorted by path. Sizes come from the tree, so
    files left out of a partial clone are still listed (and simply never read)."""
    out = git("ls-tree", "-r", "-l", "-z", "--full-tree", commit, cwd=repo)
    files = []
    for rec in out.split(b"\0"):
        if not rec:
            continue
        meta, _, path = rec.partition(b"\t")
        mode, otype, sha, size = meta.split()
        if otype != b"blob" or mode not in (b"100644", b"100755"):
            continue                                          # submodules, symlinks
        files.append((mode.decode(), sha.decode(), int(size), path.decode("utf-8", "replace")))
    files.sort(key=lambda x: x[3])
    return files


def tree_id(repo, commit):
    return git("rev-parse", f"{commit}^{{tree}}", cwd=repo).decode().strip()


class BlobReader:
    """A persistent `git cat-file --batch` over one repository: blob bytes by object id."""

    def __init__(self, repo):
        self.repo = repo
        self.proc = subprocess.Popen(["git", "cat-file", "--batch"], cwd=repo, stdin=subprocess.PIPE,
                                     stdout=subprocess.PIPE, env=_GIT_ENV)

    def read(self, sha):
        self.proc.stdin.write(sha.encode() + b"\n")
        self.proc.stdin.flush()
        header = self.proc.stdout.readline()
        parts = header.split()
        if len(parts) < 3 or parts[1] == b"missing":
            raise DataError(f"{self.repo}: object {sha} is missing")
        size = int(parts[2])
        data = self.proc.stdout.read(size)
        self.proc.stdout.read(1)                               # the newline after every object
        return data

    def close(self):
        if self.proc and self.proc.poll() is None:
            self.proc.stdin.close()
            self.proc.wait()
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def blob_stream(pool, batch=2000):
    """Yield (candidate, bytes) in the order of `pool` (candidates with ref = (repository, blob id)). Reads ahead
    `batch` candidates at a time with one `cat-file` process per repository, so a random order over a hundred
    repositories costs a handful of processes, not one per file."""
    for lo in range(0, len(pool), batch):
        part = pool[lo:lo + batch]
        by_repo = {}
        for c in part:
            by_repo.setdefault(c.ref[0], set()).add(c.ref[1])
        got = {}
        for repo in sorted(by_repo):
            with BlobReader(repo) as r:
                for sha in sorted(by_repo[repo]):
                    got[(repo, sha)] = r.read(sha)
        for c in part:
            yield c, got[c.ref]


class BlobReaders:
    """At most `size` open readers, least recently used first out."""

    def __init__(self, size=16):
        self.size = size
        self.open = {}

    def read(self, repo, sha):
        r = self.open.pop(repo, None) or BlobReader(repo)
        self.open[repo] = r
        while len(self.open) > self.size:
            old = next(iter(self.open))
            self.open.pop(old).close()
        return r.read(sha)

    def close(self):
        for r in self.open.values():
            r.close()
        self.open = {}

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
