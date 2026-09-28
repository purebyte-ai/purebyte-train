"""What the build scripts share: common options, the exclusion preamble, input verification."""
import argparse
import os
import sys

from . import identity, manifest, paths
from .util import DataError, log, polite, sha256_file, warn


def common_args(ap, what, excludes=False, fetches=False, verify_exclusion=False):
    """The options every script shares, and the two that only some scripts act on: `excludes` registers
    --allow-missing-exams (the script checks its output against the exam fingerprints), `fetches` registers --no-fetch
    (the script downloads a missing input on its own). A script never accepts an option it would ignore."""
    ap.add_argument("--dry-run", action="store_true", help=f"print the plan for {what} and stop (downloads nothing)")
    ap.add_argument("--verify", action="store_true",
                    help="re-hash what is on disk against the recorded manifests" +
                         (" and re-check the exam exclusion" if verify_exclusion else "") + ", and stop")
    if excludes:
        ap.add_argument("--allow-missing-exams", action="store_true",
                        help="build even if an exam fingerprint file is missing or stale: those sets are NOT excluded, "
                             "and the build manifest says so")
    if fetches:
        ap.add_argument("--no-fetch", action="store_true", help="fail instead of downloading a missing input")
    ap.add_argument("--threads", type=int, default=2, help="thread cap for numeric libraries (default 2)")
    return ap


def parser(doc, what, **flags):
    return common_args(argparse.ArgumentParser(description=doc, formatter_class=argparse.RawDescriptionHelpFormatter),
                       what, **flags)


def start(args):
    polite(args.threads)
    if not os.path.isdir(os.path.join(paths.TRAINING, "pbtrain")):
        raise DataError("run the data scripts from a full checkout of the repository (training/ is needed)")


def identity_preamble(built=None):
    """Load every manifest and refuse to go on if a source is also an evaluation set. `built`: the source manifest the
    script builds from, when it is not one of data/sources/*.yaml (`--manifest`); it is checked too. Returns (sources,
    exams, report)."""
    sources, exams = manifest.load_sources(), manifest.load_exams()
    extra = None
    if built is not None and all(os.path.normcase(m.path) != os.path.normcase(built.path) for m in sources):
        extra = built
        sources = sources + [built]
    identity.require_disjoint(sources, exams)
    report = dict(conflicts=0,
                  source_manifests={m.name: m.sha256 for m in sources if m is not extra},
                  exam_manifests={m.name: m.sha256 for m in exams},
                  rule="no repository URL, package name+version, dataset split, URL or file sha256 in common between "
                       "data/sources/*.yaml (and the manifest built from) and data/exams/*.yaml")
    if extra is not None:
        report["built_manifest"] = manifest_ref(extra)
    log(f"identity check: {sum(len(m.entries) for m in sources)} source entries vs "
        f"{sum(len(m.entries) for m in exams)} exam entries in {len(exams)} exam manifests: no conflict")
    return sources, exams, report


def manifest_ref(m):
    """How a build manifest names the source manifest it was built from: its path in the repository (or its file name
    when it lives elsewhere, with `outside_repository`) and the sha256 of its bytes."""
    repo = os.path.abspath(paths.REPO)
    p = os.path.abspath(m.path)
    if os.path.normcase(p).startswith(os.path.normcase(repo + os.sep)):
        return dict(path=os.path.relpath(p, repo).replace("\\", "/"), sha256=m.sha256)
    return dict(path=os.path.basename(p), outside_repository=True, sha256=m.sha256)


def outside_repository(path, what):
    """`path` (absolute) when it is outside the source tree; data never goes inside it. Raises otherwise."""
    p = os.path.abspath(path)
    repo = os.path.abspath(paths.REPO)
    if os.path.normcase(p) == os.path.normcase(repo) or os.path.normcase(p).startswith(os.path.normcase(repo + os.sep)):
        raise DataError(f"{what} would be written inside the repository ({p}): data is only ever written under "
                        f"$PUREBYTE_DATA or another folder outside the source tree")
    return p


def exam_status():
    """[(set, role, status)] with status present | missing | stale (for --dry-run)."""
    from .exclusion import declared_sets
    F = None
    out = []
    for name, role, msha in declared_sets():
        p = paths.fingerprint_path(name)
        if not os.path.exists(p):
            out.append((name, role, "missing"))
            continue
        if F is None:
            F = paths.fingerprint_module()
        _, meta = F.load(p)
        out.append((name, role, "present" if meta.get("manifest_sha256") == msha else "stale"))
    return out


def print_exam_status():
    rows = exam_status()
    log(f"exam fingerprints ({paths.rel(paths.fingerprints_dir())}):")
    for name, role, st in rows:
        log(f"  {name:22s} {role:5s} {st}")
    missing = [n for n, _, s in rows if s != "present"]
    if missing:
        log(f"  -> a build would FAIL until these are fingerprinted (or pass --allow-missing-exams): {', '.join(missing)}")
    return rows


def verify_outputs(manifest_path):
    """Re-hash the outputs recorded in a build manifest. Returns (manifest, problems)."""
    from .util import read_json
    if not os.path.exists(manifest_path):
        raise DataError(f"{manifest_path} does not exist: nothing to verify (build it first)")
    man = read_json(manifest_path)
    problems = []
    root = paths.data_dir()
    for key in ("output",):
        out = man.get(key) or {}
        p = os.path.join(root, out.get("path", ""))            # relative to $PUREBYTE_DATA, or absolute (--out)
        if not os.path.exists(p):
            problems.append(f"{out.get('path')} is missing")
        elif sha256_file(p) != out.get("sha256"):
            problems.append(f"{out.get('path')}: sha256 differs from the manifest")
        idx = out.get("index")
        if idx:
            q = os.path.join(root, idx["path"])
            if not os.path.exists(q) or sha256_file(q) != idx["sha256"]:
                problems.append(f"{idx['path']}: missing or sha256 differs")
    return man, problems


def verify_exclusion(corpus_path, man, allow_missing=False):
    """Re-check a finished corpus against the exam fingerprints on disk. Returns a list of problems."""
    from .exclusion import ContentFilter, fp_digest, load_all
    problems = []
    sets, report = load_all(allow_missing=allow_missing)
    built_with = (man.get("exclusion") or {}).get("content", {}).get("sets", {})
    for s in sets:
        rec = built_with.get(s.name)
        if rec is None:
            problems.append(f"exam set `{s.name}` was not excluded at build time (rebuild the corpus)")
        elif rec.get("fp_sha256") != fp_digest(s.fp):
            problems.append(f"exam set `{s.name}` changed since the build (rebuild the corpus)")
    cf = ContentFilter(sets)
    with open(corpus_path, "rb") as f:
        data = f.read()
    regions = cf.verify_regions(data)
    if regions:
        problems.append(f"{len(regions)} region(s) of the corpus share a block with an exam")
    for w in report.get("warnings", []):
        warn(w)
    return problems


def finish_verify(problems, what):
    if problems:
        log(f"VERIFY FAILED for {what}:")
        for p in problems:
            log(f"  - {p}")
        return 1
    log(f"verify: {what} OK")
    return 0


def main_guard(fn):
    """Run a script's main; a DataError becomes a clear message and exit status 2."""
    try:
        return fn()
    except DataError as e:
        print(f"error: {e}", file=sys.stderr, flush=True)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130
