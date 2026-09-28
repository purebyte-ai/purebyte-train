"""The IDENTITY half of the exclusion guarantee.

Every entry of every manifest is reduced to identity keys `(namespace, key)`:

    repo     normalised repository URL (host/owner/name, lower case, no scheme, no `.git`)   git entries, CredData repos
    deb      package=version                                                                   deb entries
    pypi     normalised-name==version                                                          pypi entries
    maven    group:artifact@version  (and nuget, npm: name@version)                            url entries with `package`
    hf       dataset:split  (lower case)                                                       hf entries, `derived_from`
    url      normalised download URL                                                           url / deb / pypi entries
    sha256   the digest of an artifact or of a member file                                     pins and `files`

A source and an exam that share ANY key are the same material: `check()` returns the conflicts and every build refuses
to start while there is one. Commits are deliberately left out of repository keys: another commit of an evaluation
repository is still that repository.
"""
import re

from .util import DataError

_GIT_SCP = re.compile(r"^[\w.-]+@([\w.-]+):(.+)$")


def norm_repo(url):
    u = url.strip()
    m = _GIT_SCP.match(u)
    if m:
        u = f"{m.group(1)}/{m.group(2)}"
    u = re.sub(r"^[a-z+]+://", "", u, flags=re.I)
    u = re.sub(r"^[^@/]+@", "", u)                     # credentials in a URL (never expected, never kept)
    u = u.lower().rstrip("/")
    if u.endswith(".git"):
        u = u[:-4]
    if u.startswith("www."):
        u = u[4:]
    return u


def norm_url(url):
    u = re.sub(r"^https?://", "", url.strip(), flags=re.I)
    host, _, rest = u.partition("/")
    return host.lower() + "/" + rest


def norm_pypi(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def entry_keys(e):
    """The identity keys of one manifest entry."""
    keys = set()
    t = e.get("type")
    pin = e.get("pin") or {}
    if t == "git":
        keys.add(("repo", norm_repo(e["url"])))
        for r in e.get("repos") or []:
            keys.add(("repo", norm_repo(r["url"])))
    elif t == "deb":
        keys.add(("deb", f"{e['package']}={pin['version']}"))
    elif t == "pypi":
        keys.add(("pypi", f"{norm_pypi(e['package'])}=={pin['version']}"))
    elif t == "hf":
        for s in e.get("splits") or []:
            keys.add(("hf", f"{e['dataset'].lower()}:{s.lower()}"))
    if t in ("url", "deb", "pypi"):
        keys.add(("url", norm_url(e["url"])))
        for m in e.get("mirrors") or []:
            keys.add(("url", norm_url(m)))
    pkg = e.get("package")
    if isinstance(pkg, dict):
        eco = pkg["ecosystem"].lower()
        name = norm_pypi(pkg["name"]) if eco == "pypi" else pkg["name"].lower()
        keys.add((eco, f"{name}@{pkg['version']}".lower()))
    if pin.get("sha256"):
        keys.add(("sha256", pin["sha256"]))
    for f in e.get("files") or []:
        if isinstance(f, dict) and f.get("sha256"):
            keys.add(("sha256", f["sha256"]))
    for s in list(e.get("identities") or []) + list(e.get("derived_from") or []):
        ns, _, key = s.partition(":")
        ns = ns.lower()
        if ns == "repo":
            key = norm_repo(key)
        elif ns == "hf":
            key = key.lower()
        keys.add((ns, key))
    return keys


def manifest_keys(manifest):
    """{key: [where...]} for every entry of a manifest (`where` = `manifest:set:entry` for humans)."""
    out = {}
    if manifest.kind == "source":
        for e in manifest.entries:
            for k in entry_keys(e):
                out.setdefault(k, []).append(f"{manifest.name}:{e['id']}")
    else:
        for s in manifest.sets:
            for e in s["entries"]:
                for k in entry_keys(e):
                    out.setdefault(k, []).append(f"{manifest.name}:{s['name']}:{e['id']}")
    return out


def check(sources, exams, extra_exam_keys=None):
    """Conflicts between source manifests and exam manifests: [(key, source places, exam places)].

    `extra_exam_keys` ({key: [where]}) adds identities discovered at fetch time (for example the member files of an
    evaluation set downloaded from an archive)."""
    exam_keys = {}
    for m in exams:
        for k, where in manifest_keys(m).items():
            exam_keys.setdefault(k, []).extend(where)
    for k, where in (extra_exam_keys or {}).items():
        exam_keys.setdefault(k, []).extend(where)
    conflicts = []
    for m in sources:
        for k, where in sorted(manifest_keys(m).items()):
            if k in exam_keys:
                conflicts.append((k, sorted(where), sorted(exam_keys[k])))
    return conflicts


def require_disjoint(sources, exams, extra_exam_keys=None):
    """Raise when a source is also an evaluation set. Called by every build before it reads a single byte."""
    conflicts = check(sources, exams, extra_exam_keys)
    if conflicts:
        lines = [f"{ns}:{key}  source {', '.join(s)}  <->  exam {', '.join(x)}" for (ns, key), s, x in conflicts[:30]]
        raise DataError(f"identity exclusion failed: {len(conflicts)} source(s) are also evaluation material:\n  "
                        + "\n  ".join(lines) + "\nRemove them from data/sources/*.yaml (evaluation sets never train).")
    return 0
