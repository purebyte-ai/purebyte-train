#!/usr/bin/env python3
"""Build the credential-format catalogs of the secrets generators from pinned gitleaks and trufflehog sources.

    python data/build_secret_catalogs.py            -> $PUREBYTE_DATA/catalogs/{gitleaks,trufflehog}.json + manifest
    python data/build_secret_catalogs.py --dry-run  the plan
    python data/build_secret_catalogs.py --verify   re-hash the outputs against catalogs.manifest.json

Inputs: data/sources/secret-catalogs.yaml (both repositories pinned by the commit of a release tag). Only the files
named in each entry's `take.include` are read, from git objects. The outputs hold credential-LIKE example strings from
the scanners' tests and are never committed. Deterministic: sorted file order, first-seen order of values, duplicates
removed.
"""
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pbdata import cli, fetch, globs, manifest, paths, tomlsub  # noqa: E402
from pbdata.util import DataError, ensure_dir, log, read_json, sha256_bytes, sha256_file, write_json  # noqa: E402

MANIFEST = os.path.join(paths.SOURCE_MANIFESTS, "secret-catalogs.yaml")
_GO_ESC = {"a": "\a", "b": "\b", "f": "\f", "n": "\n", "r": "\r", "t": "\t", "v": "\v", "\\": "\\", "'": "'", '"': '"'}


# ------------------------------------------------------------------------------------------------ Go literals
def go_tokens(text):
    """Tokens of Go source: ('str', value), ('ident', name), ('punct', ch). Comments are skipped; interpreted strings
    are unescaped, raw strings kept as they are."""
    i, n = 0, len(text)
    while i < n:
        c = text[i]
        if c in " \t\r\n":
            i += 1
        elif text.startswith("//", i):
            j = text.find("\n", i)
            i = n if j < 0 else j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            i = n if j < 0 else j + 2
        elif c == "`":
            j = text.find("`", i + 1)
            j = n if j < 0 else j
            yield "str", text[i + 1:j].replace("\r", "")
            i = j + 1
        elif c == '"':
            out, j = [], i + 1
            while j < n and text[j] != '"' and text[j] != "\n":
                if text[j] == "\\" and j + 1 < n:
                    e = text[j + 1]
                    if e in _GO_ESC:
                        out.append(_GO_ESC[e])
                        j += 2
                    elif e == "x":
                        out.append(chr(int(text[j + 2:j + 4], 16)))
                        j += 4
                    elif e in "uU":
                        k = 4 if e == "u" else 8
                        out.append(chr(int(text[j + 2:j + 2 + k], 16)))
                        j += 2 + k
                    elif e in "01234567":
                        out.append(chr(int(text[j + 1:j + 4], 8)))
                        j += 4
                    else:
                        out.append(e)
                        j += 2
                else:
                    out.append(text[j])
                    j += 1
            yield "str", "".join(out)
            i = j + 1
        elif c == "'":
            j = text.find("'", i + 1)
            i = n if j < 0 else j + 1
            yield "punct", "rune"
        elif c.isalnum() or c == "_":
            j = i
            while j < n and (text[j].isalnum() or text[j] == "_"):
                j += 1
            yield "ident", text[i:j]
            i = j
        else:
            yield "punct", c
            i += 1


_OPEN = {("punct", "{"), ("punct", "("), ("punct", "[")}
_CLOSE = {("punct", "}"), ("punct", ")"), ("punct", "]")}


def _slice_start(toks, k, name):
    """Index just after `{` when toks[k:] is `name := []string{` or `name = []string{`, else None."""
    if toks[k] != ("ident", name):
        return None
    j = k + 1
    if j < len(toks) and toks[j] == ("punct", ":"):
        j += 1
    if j >= len(toks) or toks[j] != ("punct", "="):
        return None
    j += 1
    if toks[j:j + 4] != [("punct", "["), ("punct", "]"), ("ident", "string"), ("punct", "{")]:
        return None
    return j + 4


def _append_start(toks, k, name):
    """Index of the first element when toks[k:] is `name = append(name, `, else None."""
    if toks[k:k + 6] == [("ident", name), ("punct", "="), ("ident", "append"), ("punct", "("), ("ident", name),
                         ("punct", ",")]:
        return k + 6
    return None


def _elements(toks, j, closer):
    """The elements (token lists) of a comma-separated list starting at j and ending at `closer` (depth 0)."""
    depth, elem, out = 0, [], []
    while j < len(toks):
        t = toks[j]
        j += 1
        if depth == 0 and (t == ("punct", ",") or t == ("punct", closer)):
            if elem:
                out.append(elem)
            elem = []
            if t == ("punct", closer):
                break
            continue
        if t in _OPEN:
            depth += 1
        elif t in _CLOSE:
            depth -= 1
        elem.append(t)
    return out


def slice_literals(text, name):
    """The elements of `name := []string{...}` and of `name = append(name, ...)` that are exactly one string literal
    (elements built by calls or concatenations are generated at test time and are not taken), plus, for `tps`, the
    literal sample secrets handed to the `GenerateSampleSecret(s)` helpers."""
    toks = list(go_tokens(text))
    out = []
    for k in range(len(toks)):
        j, closer = _slice_start(toks, k, name), "}"
        if j is None:
            j, closer = _append_start(toks, k, name), ")"
        if j is not None:
            out += [e[0][1] for e in _elements(toks, j, closer) if len(e) == 1 and e[0][0] == "str"]
            continue
        if name == "tps" and toks[k][0] == "ident" and toks[k][1] in ("GenerateSampleSecret", "GenerateSampleSecrets") \
                and k + 1 < len(toks) and toks[k + 1] == ("punct", "("):
            args = _elements(toks, k + 2, ")")
            if len(args) >= 2 and len(args[1]) == 1 and args[1][0][0] == "str":
                out.append(args[1][0][1])
    return out


def single_literal_regexes(text):
    """Arguments of regexp.MustCompile(...) / regexp.Compile(...) that are exactly one string literal."""
    toks = list(go_tokens(text))
    out = []
    for k in range(len(toks) - 5):
        if toks[k] == ("ident", "regexp") and toks[k + 1] == ("punct", ".") and \
                toks[k + 2][1] in ("MustCompile", "Compile") and toks[k + 3] == ("punct", "(") and \
                toks[k + 4][0] == "str" and toks[k + 5] == ("punct", ")"):
            out.append(toks[k + 4][1])
    return out


def unique(seq):
    seen, out = set(), []
    for x in seq:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ------------------------------------------------------------------------------------------------ builders
def read_files(e, repo):
    take = e.get("take") or {}
    files = {}
    with fetch.BlobReader(repo) as r:
        for mode, sha, size, path in fetch.ls_tree(repo, e["pin"]["commit"]):
            if globs.match_any(path, take.get("include")):
                files[path] = r.read(sha).decode("utf-8", "replace")
    if not files:
        raise DataError(f"{e['id']}: none of {take.get('include')} exists at {e['pin']['commit'][:12]}")
    return files


def gitleaks_catalog(files):
    conf = tomlsub.loads(files["config/gitleaks.toml"])
    rules = []
    for r in conf.get("rules", []):
        item = {"id": r["id"]}
        if "regex" in r:
            item["regex"] = r["regex"]
        item["kw"] = list(r.get("keywords", []))
        rules.append(item)
    tps, fps = [], []
    for path in sorted(p for p in files if p.endswith(".go")):
        tps += slice_literals(files[path], "tps")
        fps += slice_literals(files[path], "fps")
    return {"rules": rules, "tps": unique(tps), "fps": unique(fps)}


EXAMPLE = re.compile(r"^[A-Za-z0-9+/=_\-.:@$%*!~]+$")


def trufflehog_catalog(files, lo, hi):
    code = {p: t for p, t in files.items() if p.endswith(".go") and not p.endswith("_test.go")}
    tests = {p: t for p, t in files.items() if p.endswith("_test.go")}
    providers = sorted({os.path.basename(os.path.dirname(p)) for p in code
                        if p.startswith("pkg/detectors/") and os.path.dirname(p) != "pkg/detectors"})
    patterns, examples = [], []
    for path in sorted(code):
        patterns += single_literal_regexes(code[path])
    for path in sorted(tests):
        for kind, v in go_tokens(tests[path]):
            if kind == "str" and lo <= len(v) <= hi and EXAMPLE.match(v) and "://" not in v \
                    and any(ch.isdigit() for ch in v) and any(ch.isalpha() for ch in v):
                examples.append(v)
    return {"providers": providers, "patterns": unique(patterns), "examples": unique(examples)}


def main():
    ap = cli.parser(__doc__, "the credential-format catalogs")
    args = ap.parse_args()
    cli.start(args)
    man = manifest.load(MANIFEST)
    cli.identity_preamble()
    out_dir = paths.catalogs_dir()
    rec_path = os.path.join(out_dir, "catalogs.manifest.json")
    if args.verify:
        problems = []
        if not os.path.exists(rec_path):
            problems.append("catalogs not built")
        else:
            for name, sha in read_json(rec_path)["outputs"].items():
                p = os.path.join(out_dir, name)
                if not os.path.exists(p) or sha256_file(p) != sha:
                    problems.append(f"{name}: missing or changed")
        return cli.finish_verify(problems, "catalogs")
    entries = {e["id"]: e for e in man.entries}
    repos = {}
    for eid, e in entries.items():
        take = e.get("take") or {}
        repos[eid] = fetch.fetch_git(e["url"], e["pin"]["commit"], blob_limit=take.get("blob_limit"),
                                     dry_run=args.dry_run)
    if args.dry_run:
        return 0
    lo, hi = (man.doc.get("build") or {}).get("example_length", {}).get("min", 16), \
        (man.doc.get("build") or {}).get("example_length", {}).get("max", 200)
    g = gitleaks_catalog(read_files(entries["gitleaks"], repos["gitleaks"]))
    t = trufflehog_catalog(read_files(entries["trufflehog"], repos["trufflehog"]), lo, hi)
    ensure_dir(out_dir)
    outputs = {}
    import json
    for name, obj in (("gitleaks.json", g), ("trufflehog.json", t)):
        data = (json.dumps(obj, ensure_ascii=False, indent=1) + "\n").encode("utf-8")
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(data)
        outputs[name] = sha256_bytes(data)
    write_json(rec_path, dict(
        builder="data/build_secret_catalogs.py", manifest=dict(path="data/sources/secret-catalogs.yaml", sha256=man.sha256),
        inputs={eid: dict(url=e["url"], commit=e["pin"]["commit"], licence=e["licence"]) for eid, e in entries.items()},
        counts=dict(gitleaks_rules=len(g["rules"]), gitleaks_keywords=sum(len(r["kw"]) for r in g["rules"]),
                    gitleaks_tps=len(g["tps"]), gitleaks_fps=len(g["fps"]), trufflehog_providers=len(t["providers"]),
                    trufflehog_patterns=len(t["patterns"]), trufflehog_examples=len(t["examples"])),
        outputs=outputs))
    log(f"catalogs -> {paths.rel(out_dir)}: gitleaks {len(g['rules'])} rules, {len(g['tps'])} tps, {len(g['fps'])} fps; "
        f"trufflehog {len(t['providers'])} providers, {len(t['patterns'])} patterns, {len(t['examples'])} examples")
    return 0


if __name__ == "__main__":
    sys.exit(cli.main_guard(main))
