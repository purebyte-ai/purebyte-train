#!/usr/bin/env python3
"""Build the code-context corpus: real source code and configuration text from pinned public repositories.

    python data/build_code_corpus.py               fetch what is missing, build, verify, write the manifest
    python data/build_code_corpus.py --dry-run     print the plan (repositories, pins, quotas, exam status) and stop
    python data/build_code_corpus.py --verify      re-check the corpus on disk (output and input pins, exam exclusion)

Reads data/sources/code-context.yaml (repositories pinned by commit) and every exam fingerprint file; writes
$PUREBYTE_DATA/corpora/code-context.bin, .manifest.json and .index.tsv.gz. Same inputs -> same bytes.

Before reading a single byte the build checks that no source is an evaluation set (identity), and it refuses to run
while an exam fingerprint file is missing (content), unless --allow-missing-exams, which is written into the manifest.
Every file that shares a 64-byte block with an exam loses the lines around that block (or is dropped); the finished
corpus is verified as a whole. Exit status: 0 built and all guards passed, 1 built but a guard failed (see the
manifest), 2 could not build.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pbdata import cli, fetch, licences, manifest, paths  # noqa: E402
from pbdata.corpus import Candidate, CorpusWriter, dedup  # noqa: E402
from pbdata.exclusion import ContentFilter, load_all  # noqa: E402
from pbdata.util import DataError, human, log  # noqa: E402

DEFAULT_MANIFEST = os.path.join(paths.SOURCE_MANIFESTS, "code-context.yaml")


def group_of(path, groups):
    base = path.rsplit("/", 1)[-1]
    for g, spec in groups.items():
        if base in (spec.get("names") or ()):
            return g
    ext = os.path.splitext(base)[1].lower()
    if not ext:
        return None
    for g, spec in groups.items():
        if ext in (spec.get("ext") or ()):
            return g
    return None


def candidates_of(entry, repo, build, seed):
    """[Candidate] of one repository at its pinned commit, and the texts of its root licence files."""
    from pbdata import globs
    groups, skip = build["groups"], set(build.get("skip_dirs") or ())
    lo, hi = build["file_bytes"]["min"], build["file_bytes"]["max"]
    take = entry.get("take") or {}
    commit = entry["pin"]["commit"]
    out, licence_files = [], {}
    for mode, sha, size, path in fetch.ls_tree(repo, commit):
        if "/" not in path and licences.LICENCE_FILE.match(path) and size <= 256 << 10:
            licence_files[path] = sha
        parts = path.split("/")
        if any(p in skip for p in parts[:-1]):
            continue
        if not globs.selected(path, take.get("include"), take.get("exclude")):
            continue
        if not (lo <= size <= min(hi, take.get("max_file_bytes", hi))):
            continue
        g = group_of(path, groups)
        if g is None:
            continue
        out.append(Candidate(g, entry["id"], path, size, sha, (repo, sha), seed))
    texts = {}
    if licence_files:
        with fetch.BlobReader(repo) as r:
            for name, sha in sorted(licence_files.items()):
                texts[name] = r.read(sha).decode("utf-8", "replace")
    return out, texts


def make_transform(cf, build):
    """Raw file bytes -> (corpus bytes or None, reason): text only, one trailing newline, exam blocks cut out."""
    max_nul = build.get("max_nul_bytes", 2)
    min_bytes = build["file_bytes"]["min"]

    def transform(c, raw):
        if raw.count(0) > max_nul:
            return None, "binary"
        try:
            raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, "not-utf8"
        if not raw.endswith(b"\n"):
            raw += b"\n"
        clean, _ = cf.clean_text(raw, min_bytes=min_bytes)
        return (clean, "ok") if clean is not None else (None, "exam-overlap")
    return transform


def assemble(man, build, cf, repo_of, out_dir, target, extra=None, name="code-context"):
    """The whole build once the checks are done: candidates of every repository, selection, writing, verification,
    guards, manifest. `repo_of(entry)` gives a git directory holding the entry's pinned commit."""
    extra = extra or {}
    seed = build.get("seed", 0)
    inputs, cands, restrictive = [], [], []
    for e in man.entries:
        repo = repo_of(e)
        c, texts = candidates_of(e, repo, build, seed)
        lic = licences.check(e["licence"], texts)
        if lic["status"] == "restrictive":
            restrictive.append(f"{e['id']}: {lic['note']}")
        inputs.append(dict(id=e["id"], type="git", url=e["url"], commit=e["pin"]["commit"],
                           tree=fetch.tree_id(repo, e["pin"]["commit"]), licence=e["licence"], licence_check=lic,
                           candidates=len(c)))
        cands += c
    if restrictive:
        raise DataError("licence check failed (declared permissive, root licence is not):\n  " + "\n  ".join(restrictive))
    cands, duplicates = dedup(cands)
    log(f"{len(cands)} distinct candidate files ({duplicates} duplicates across repositories left out)")
    groups = {g: dict(spec, min_files=spec.get("min_files", build.get("min_files", 0)))
              for g, spec in build["groups"].items()}
    writer = CorpusWriter(name, out_dir, groups, target, seed=seed, tolerance=build.get("tolerance", 0.03),
                          decile_tolerance=build.get("decile_tolerance", 0.03))
    log(f"selecting {human(target)} in {len(groups)} groups (seed {seed})")
    pieces, stats = writer.select(cands, fetch.blob_stream, make_transform(cf, build))
    rows, verification = writer.write(pieces, cf)
    used = {}
    for off, n, g, src, path, cid in rows:
        u = used.setdefault(src, [0, 0])
        u[0] += 1
        u[1] += n
    for inp in inputs:
        inp["files_used"], inp["bytes_used"] = used.get(inp["id"], [0, 0])
    content = dict(extra.get("exam_report") or {}, filter=cf.report(), final_verification=verification,
                   rule="whole lines around every 64-byte block shared with an exam are cut out; a file with less than "
                        f"{build['file_bytes']['min']} bytes left is dropped; the finished corpus is verified as a whole")
    return writer.finish(rows, stats, dict(
        schema=1, builder="data/build_code_corpus.py",
        manifest=cli.manifest_ref(man),
        inputs=inputs, build=dict(build, target_bytes=target),
        exclusion=dict(identity=extra.get("exclusion_identity"), content=content),
        duplicates_left_out=duplicates))


def plan(man, build):
    target = int(build["target_mib"] * 2 ** 20)
    log(f"code-context: {len(man.entries)} repositories -> {paths.rel(os.path.join(paths.corpora_dir(), 'code-context.bin'))}"
        f" ({human(target)})")
    for e in man.entries:
        log(f"  {e['id']:44s} {e['pin']['commit'][:12]}  {e['licence']}")
    log("groups (declared share -> quota):")
    for g, spec in build["groups"].items():
        log(f"  {g:12s} {spec['share']:.3f} -> {human(int(target * spec['share']))}")
    cli.print_exam_status()
    for e in man.entries:
        fetch.fetch_git(e["url"], e["pin"]["commit"], blob_limit=build["file_bytes"]["max"], dry_run=True)


def verify(args, man):
    out_dir = args.out or paths.corpora_dir()
    corpus = os.path.join(out_dir, "code-context.bin")
    built, problems = cli.verify_outputs(os.path.join(out_dir, "code-context.manifest.json"))
    if built.get("manifest", {}).get("sha256") != man.sha256:
        problems.append("data/sources/code-context.yaml changed since the build (rebuild)")
    for inp in built.get("inputs", []):
        repo = fetch.repo_cache(inp["url"])
        if not fetch.has_commit(repo, inp["commit"]):
            problems.append(f"{inp['id']}: commit {inp['commit'][:12]} is not in the cache (fetch it to re-verify)")
        elif fetch.tree_id(repo, inp["commit"]) != inp["tree"]:
            problems.append(f"{inp['id']}: tree of {inp['commit'][:12]} differs from the build")
    if os.path.exists(corpus):
        problems += cli.verify_exclusion(corpus, built, allow_missing=args.allow_missing_exams)
    return cli.finish_verify(problems, "code-context corpus")


def main():
    ap = cli.parser(__doc__, "the code-context corpus", excludes=True, fetches=True, verify_exclusion=True)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="source manifest (default: %(default)s)")
    ap.add_argument("--mib", type=float, help="target size in MiB (default: build.target_mib of the manifest)")
    ap.add_argument("--out", help="output directory (default: $PUREBYTE_DATA/corpora)")
    args = ap.parse_args()
    cli.start(args)
    man = manifest.load(args.manifest)
    build = man.doc["build"]
    if args.out:
        args.out = cli.outside_repository(args.out, "--out")
    _, _, identity_report = cli.identity_preamble(built=man)
    if args.dry_run:
        plan(man, build)
        return 0
    if args.verify:
        return verify(args, man)

    sets, exam_report = load_all(allow_missing=args.allow_missing_exams)
    target = int((args.mib if args.mib else build["target_mib"]) * 2 ** 20)

    def repo_of(e):
        repo = fetch.repo_cache(e["url"])
        if args.no_fetch and not fetch.has_commit(repo, e["pin"]["commit"]):
            raise DataError(f"{e['id']}: not fetched (run `python data/fetch.py {os.path.relpath(args.manifest)}`)")
        return fetch.fetch_git(e["url"], e["pin"]["commit"], blob_limit=build["file_bytes"]["max"])

    result = assemble(man, build, ContentFilter(sets), repo_of, args.out or paths.corpora_dir(), target,
                      extra=dict(exclusion_identity=identity_report, exam_report=exam_report))
    log(f"wrote {paths.rel(os.path.join(args.out or paths.corpora_dir(), 'code-context.bin'))}: "
        f"{human(result['output']['bytes'])}, sha256 {result['output']['sha256']}")
    if not result["guard"]["ok"]:
        log("GUARD FAILED (the corpus is written; its manifest says why):")
        for f in result["guard"]["failures"]:
            log(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(cli.main_guard(main))
