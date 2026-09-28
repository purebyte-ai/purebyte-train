#!/usr/bin/env python3
"""Build the binary-context corpus: real binaries from pinned public artifacts.

    python data/build_binary_corpus.py               fetch what is missing, build, verify, write the manifest
    python data/build_binary_corpus.py --dry-run     print the plan (artifacts, pins, quotas, exam status) and stop
    python data/build_binary_corpus.py --verify      re-check the corpus on disk (output and input pins, exams)

Reads data/sources/binary-context.yaml (Debian packages, PyPI wheels, release archives, Maven jars, all pinned by
sha256) and every exam fingerprint file; writes $PUREBYTE_DATA/corpora/binary-context.bin, .manifest.json and
.index.tsv.gz. Same inputs -> same bytes.

Members are recognised by content (ELF, ar, PE, Mach-O, Java class/jar, WebAssembly; .pyc by name; firmware by the
entry's `group`). An ELF with many C++ symbol names joins `elf_symbols`, whose span starts inside the symbol table.
Every file contributes ONE contiguous span of at most `span.max` bytes, never spliced. Exclusion:
  identity  no artifact may be an evaluation artifact (manifests), and no member may have the sha256 of an evaluation
            file (checked while reading): either stops the build;
  content   the span is chosen inside the part of the file that shares no 64-byte block with any exam; a file with no
            such part of at least `span.min` bytes is dropped whole (--binary-policy drop-file drops every file that
            shares any block, which for ELF files removes nearly all of them: see data/README.md).
Exit status: 0 built and all guards passed, 1 built but a guard failed (see the manifest), 2 could not build.
"""
import itertools
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pbdata import archives, cli, fetch, manifest, paths  # noqa: E402
from pbdata.corpus import Candidate, CorpusWriter, dedup  # noqa: E402
from pbdata.exclusion import ContentFilter, load_all  # noqa: E402
from pbdata.util import DataError, ensure_dir, human, log, sha256_bytes, sha256_file  # noqa: E402

DEFAULT_MANIFEST = os.path.join(paths.SOURCE_MANIFESTS, "binary-context.yaml")
SYMBOL = b"\x00_ZN"
REGION_MAX = 4 << 20             # the part of a large member that is kept as a candidate (the span is chosen in it)
MACHO = {b"\xfe\xed\xfa\xce", b"\xfe\xed\xfa\xcf", b"\xce\xfa\xed\xfe", b"\xcf\xfa\xed\xfe"}


def is_pe(data):
    if data[:2] != b"MZ" or len(data) < 64:
        return False
    off = int.from_bytes(data[0x3C:0x40], "little")
    return 0 < off < len(data) - 4 and data[off:off + 4] == b"PE\0\0"


def detect_group(name, data, forced=None, symbols_min=300):
    """The corpus group of a member, from its content (and its name for .pyc / .jar), or None."""
    if forced:
        return forced
    head = data[:8]
    lname = name.lower()
    if head[:4] == b"\x7fELF":
        return "elf_symbols" if data.count(SYMBOL) >= symbols_min else "elf"
    if head == b"!<arch>\n":
        return "ar"
    if lname.endswith(".pyc") and len(data) > 16:
        return "pyc"
    if is_pe(data):
        return "pe"
    if head[:4] in MACHO:
        return "macho"
    if head[:4] == b"\xca\xfe\xba\xbe":                     # Mach-O universal or a Java class file
        n = int.from_bytes(data[4:8], "big")
        if 0 < n < 30 and not lname.endswith(".class"):
            return "macho"
        return "java" if lname.endswith(".class") else None
    if head[:4] == b"\0asm":
        return "wasm"
    if head[:4] == b"PK\x03\x04" and lname.endswith((".jar", ".aar")):
        return "java"
    return None


def region_of(data, group, pick):
    """(offset, bytes): the whole member, or a deterministic REGION_MAX part of a large one (for the symbol group,
    the part that starts at the first C++ symbol name)."""
    if len(data) <= REGION_MAX:
        return 0, data
    if group == "elf_symbols":
        start = max(0, data.find(SYMBOL))
    else:
        start = pick % (len(data) - REGION_MAX + 1)
    start = min(start, len(data) - REGION_MAX)
    return start, data[start:start + REGION_MAX]


def member_cache(sha, offset):
    return os.path.join(paths.cache_dir(), "members", sha[:2], f"{sha}.{offset}")


def deb_licences(path, package):
    """The `License:` names of a Debian package's machine-readable copyright file (empty when it has none)."""
    try:
        text = archives.read_member(path, f"usr/share/doc/{package}/copyright", fmt="deb").decode("utf-8", "replace")
    except DataError:
        return []
    return sorted({m.group(1).strip() for m in re.finditer(r"^License:\s*(\S[^\n]*)$", text, re.M)})[:40]


def exam_file_digests():
    """{sha256: 'set:entry:file'} of every file listed in the exam manifests (the member-level identity check)."""
    out = {}
    for s, m in manifest.exam_sets():
        for e in s["entries"]:
            for f in e.get("files") or []:
                if f.get("sha256"):
                    out[f["sha256"]] = f"{s['name']}:{e['id']}:{f.get('name') or f.get('member') or f.get('path')}"
    return out


def enumerate_entry(e, path, build, seed, exam_digests):
    """Candidates of one artifact (members recognised, sizes checked, regions cached), and an input record."""
    take = e.get("take") or {}
    mins = build.get("group_member_min") or {}
    lo, hi = min([build["member_bytes"]["min"]] + list(mins.values())), build["member_bytes"]["max"]
    sym_min = (build.get("symbols") or {}).get("min_count", 300)
    fmt = archives.entry_format(e)
    members = archives.iter_members(path, fmt=fmt, include=take.get("members"), exclude=take.get("exclude"),
                                    nested=take.get("nested"), max_bytes=hi)
    if take.get("self"):                                    # the downloaded file is itself a candidate (a jar)
        with open(path, "rb") as f:
            own = [(os.path.basename(e["url"]), f.read())]
        members = itertools.chain(own, members)
    out, seen, groups = [], 0, {}
    for name, data in members:
        seen += 1
        if not lo <= len(data) <= hi:
            continue
        g = detect_group(name, data, e.get("group"), sym_min)
        if g is None or len(data) < mins.get(g, build["member_bytes"]["min"]):
            continue
        sha = sha256_bytes(data)
        if sha in exam_digests:
            raise DataError(f"identity exclusion failed: {e['id']}:{name} is the evaluation file {exam_digests[sha]} "
                            f"(same sha256). Remove the artifact from the source manifest.")
        c = Candidate(g, e["id"], name, len(data), sha, None, seed)
        off, region = region_of(data, g, int(c.key[:12], 16))
        cache = member_cache(sha, off)
        if not os.path.exists(cache):
            ensure_dir(os.path.dirname(cache))
            with open(cache + ".tmp", "wb") as f:
                f.write(region)
            os.replace(cache + ".tmp", cache)
        c.ref = (cache, off)
        out.append(c)
        groups[g] = groups.get(g, 0) + 1
    pin = e.get("pin") or {}
    pkg = e.get("package")
    package, version = (pkg.get("name"), pkg.get("version")) if isinstance(pkg, dict) else (pkg, pin.get("version"))
    rec = dict(id=e["id"], type=e["type"], url=e["url"], sha256=pin.get("sha256") or sha256_file(path),
               package=package, version=version, licence=e["licence"], members_read=seen,
               candidates=len(out), candidate_groups=dict(sorted(groups.items())))
    if e["type"] == "deb":
        rec["copyright_licences"] = deb_licences(path, e["package"])
    return out, rec


def cached_stream(pool):
    for c in pool:
        with open(c.ref[0], "rb") as f:
            yield c, f.read()


def make_transform(cf, build, policy):
    span_max, span_min = build["span"]["max"], build["span"]["min"]

    def transform(c, region):
        if policy == "drop-file" and cf.shared(region):
            for name in cf.shared(region):
                cf.stats[name]["files_dropped"] += 1
                cf.stats[name]["bytes_dropped"] += len(region)
            return None, "exam-overlap"
        span = cf.clean_span(region, span_max, min(span_min, len(region)), pick=int(c.key[12:28], 16),
                             anchor=SYMBOL if c.group == "elf_symbols" else None)
        if span is None:
            return None, "no-clean-span"
        return region[span[0]:span[1]], "ok"
    return transform


def plan(man, build):
    target = int(build["target_mib"] * 2 ** 20)
    total = sum((e.get("pin") or {}).get("size") or 0 for e in man.entries)
    log(f"binary-context: {len(man.entries)} artifacts -> {paths.rel(os.path.join(paths.corpora_dir(), 'binary-context.bin'))}"
        f" ({human(target)}); known download sizes add up to {human(total)} (debs and wheels not counted)")
    by_type = {}
    for e in man.entries:
        by_type.setdefault(e["type"], []).append(e["id"])
    for t, ids in sorted(by_type.items()):
        log(f"  {t:5s} {len(ids):4d}  {', '.join(ids[:6])}{' ...' if len(ids) > 6 else ''}")
    log("groups (declared share -> quota):")
    for g, spec in build["groups"].items():
        log(f"  {g:12s} {spec['share']:.3f} -> {human(int(target * spec['share']))}")
    cli.print_exam_status()


def verify(args, man):
    out_dir = args.out or paths.corpora_dir()
    corpus = os.path.join(out_dir, "binary-context.bin")
    built, problems = cli.verify_outputs(os.path.join(out_dir, "binary-context.manifest.json"))
    if built.get("manifest", {}).get("sha256") != man.sha256:
        problems.append("data/sources/binary-context.yaml changed since the build (rebuild)")
    for inp in built.get("inputs", []):
        p = fetch.blob_path(inp["sha256"])
        if not os.path.exists(p):
            problems.append(f"{inp['id']}: not in the cache (fetch it to re-verify)")
        elif sha256_file(p) != inp["sha256"]:
            problems.append(f"{inp['id']}: cached file does not match its sha256")
    if os.path.exists(corpus):
        problems += cli.verify_exclusion(corpus, built, allow_missing=args.allow_missing_exams)
    return cli.finish_verify(problems, "binary-context corpus")


def main():
    ap = cli.parser(__doc__, "the binary-context corpus", excludes=True, fetches=True,
                    verify_exclusion=True)
    ap.add_argument("--manifest", default=DEFAULT_MANIFEST, help="source manifest (default: %(default)s)")
    ap.add_argument("--mib", type=float, help="target size in MiB (default: build.target_mib of the manifest)")
    ap.add_argument("--out", help="output directory (default: $PUREBYTE_DATA/corpora)")
    ap.add_argument("--binary-policy", choices=("clean-span", "drop-file"), default="clean-span",
                    help="clean-span (default): the span avoids every shared block; drop-file: a file sharing any "
                         "block with an exam is dropped whole")
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
    cf = ContentFilter(sets)
    seed = build.get("seed", 0)
    target = int((args.mib if args.mib else build["target_mib"]) * 2 ** 20)
    exam_digests = exam_file_digests()
    inputs, cands = [], []
    for e in sorted(man.entries, key=lambda e: e["id"]):
        pin = e.get("pin") or {}
        cached = fetch.blob_path(pin["sha256"]) if pin.get("sha256") else fetch.unpinned_path(e["url"])
        if args.no_fetch and not os.path.exists(cached):
            raise DataError(f"{e['id']}: not fetched (run `python data/fetch.py {os.path.relpath(args.manifest)}`)")
        path = fetch.fetch_file(e)
        c, rec = enumerate_entry(e, path, build, seed, exam_digests)
        inputs.append(rec)
        cands += c
        log(f"  {e['id']:48s} {len(c):5d} candidates {rec['candidate_groups']}")
    cands, duplicates = dedup(cands)
    log(f"{len(cands)} distinct candidate files ({duplicates} duplicates left out)")
    groups = {g: dict(spec) for g, spec in build["groups"].items()}
    writer = CorpusWriter("binary-context", args.out or paths.corpora_dir(), groups, target, seed=seed,
                          tolerance=build.get("tolerance", 0.03), decile_tolerance=build.get("decile_tolerance", 0.05))
    log(f"selecting {human(target)} in {len(groups)} groups (seed {seed}, policy {args.binary_policy})")
    pieces, stats = writer.select(cands, cached_stream, make_transform(cf, build, args.binary_policy))
    rows, verification = writer.write(pieces, cf)
    used = {}
    for off, n, g, src, path, cid in rows:
        u = used.setdefault(src, [0, 0])
        u[0] += 1
        u[1] += n
    for inp in inputs:
        inp["files_used"], inp["bytes_used"] = used.get(inp["id"], [0, 0])
    content = dict(exam_report, filter=cf.report(), final_verification=verification, policy=args.binary_policy,
                   rule="one contiguous span per file, chosen where the file shares no 64-byte block with any exam; "
                        f"files without such a span of {build['span']['min']} bytes are dropped whole; the finished "
                        "corpus is verified as a whole")
    result = writer.finish(rows, stats, dict(
        schema=1, builder="data/build_binary_corpus.py",
        manifest=cli.manifest_ref(man),
        inputs=inputs, build=dict(build, target_bytes=target, region_max=REGION_MAX),
        exclusion=dict(identity=dict(identity_report, member_files_checked=len(exam_digests)), content=content),
        duplicates_left_out=duplicates))
    log(f"wrote {paths.rel(writer.bin_path)}: {human(result['output']['bytes'])}, sha256 {result['output']['sha256']}")
    if not result["guard"]["ok"]:
        log("GUARD FAILED (the corpus is written; its manifest says why):")
        for f in result["guard"]["failures"]:
            log(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(cli.main_guard(main))
