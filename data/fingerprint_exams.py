#!/usr/bin/env python3
"""Fingerprint every evaluation set: the content half of the exclusion guarantee.

    python data/fingerprint_exams.py                       every set declared in data/exams/*.yaml
    python data/fingerprint_exams.py --set code-fp-a       one set (repeatable)
    python data/fingerprint_exams.py --set bin-fp-c --from-dir /path/to/copies
    python data/fingerprint_exams.py --dry-run             which sets, from where, what is missing
    python data/fingerprint_exams.py --verify              recompute and compare with the files on disk

Reads only what data/fetch_exams.py and data/fetch_creddata.py put on disk (and --from-dir): nothing is downloaded.
Writes $PUREBYTE_DATA/exams/fingerprints/<set>.npz with `pbtrain.data.fingerprint.save` (the algorithm the training
factory uses too: sorted unique 64-bit hashes of 64-byte blocks at content-defined anchors). The metadata records the
set, its role (test | dev), the sha256 of the manifest it was made from (a build refuses fingerprints of another
version), how many files and bytes were read, and which files were missing.

Content per set: git entries - every file of the pinned commit, read from git objects; files - the listed files,
verified by sha256 (fetched by fetch_exams.py, or found in --from-dir by sha256); document sets - the `text` of every
JSON line; CredData - every file of the partition's repositories in the downloaded dataset.

A set that cannot be fingerprinted (not fetched, CredData not downloaded, a file that has a public download missing)
is reported and the next set is done; the summary at the end names every such set, and the exit status is then 1.
Files that have NO public download (`type: local`: 15 of bin-fp-b and 46 of bin-fp-c) never stop a set: without a
local copy (--from-dir) the set's fingerprint file is written from its other files and says it is incomplete, and every
build that uses it repeats that the content of the missing files is not excluded. --allow-partial also writes a set
whose files WITH a public download are missing (recorded the same way).

What a missing fingerprint file means: every data build refuses to run while a declared set has none (or one made from
another version of its manifest), unless it is given --allow-missing-exams, which records in the build manifest the sets
that were not excluded. The training factory compares every training window with every fingerprint file present and
requires those of the exams a recipe declares (`exams:`): a set without a file is checked by nothing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from pbdata import cli, exams, manifest, paths  # noqa: E402
from pbdata.exclusion import fp_digest  # noqa: E402
from pbdata.util import DataError, ensure_dir, human, log, sha256_bytes  # noqa: E402


def fingerprint_set(s, man, from_dir=None, allow_partial=False):
    F = paths.fingerprint_module()
    acc, acc_n, n_files, n_bytes, missing = [], 0, 0, 0, []
    import hashlib
    digest = hashlib.sha256()
    for label, data in exams.iter_set_content(s, missing, from_dir=from_dir, allow_partial=allow_partial):
        fp = F.fingerprints(data)
        acc.append(fp)
        acc_n += len(fp)
        n_files += 1
        n_bytes += len(data)
        digest.update(f"{label}\0{sha256_bytes(data)}\n".encode("utf-8", "replace"))
        if acc_n > 16_000_000:                            # bound memory on large sets
            acc = [np.unique(np.concatenate(acc))]
            acc_n = len(acc[0])
    fp = np.unique(np.concatenate(acc)) if acc else np.zeros(0, np.uint64)
    if n_files == 0:
        raise DataError(f"set {s['name']}: no content found (fetch it first, or point --from-dir at a copy)")
    local_only = set(exams.local_only_files(s))
    meta = dict(name=s["name"], role=s["role"], manifest=os.path.basename(man.path), manifest_sha256=man.sha256,
                files=n_files, bytes=n_bytes, content_sha256=digest.hexdigest(), complete=not missing,
                missing_count=len(missing), missing_local_only=sum(1 for x in missing if x in local_only),
                missing=missing[:200], source="from-dir" if from_dir else "exam directory",
                created_by="data/fingerprint_exams.py")
    return fp, meta


def _first_line(text):
    return str(text).strip().splitlines()[0][:240] if str(text).strip() else ""


def summary(results, verify=False):
    """Prints what was done for every set and what it means for the exclusion. Returns the exit status."""
    log("")
    log("summary:")
    for name, role, status, info in results:
        if status == "complete":
            log(f"  {name:16s} {role:4s} {'checked' if verify else 'fingerprinted'} ({info['files']} files)")
        elif status == "partial":
            log(f"  {name:16s} {role:4s} fingerprinted, INCOMPLETE: {info['missing_count']} file(s) missing, "
                f"{info['missing_local_only']} of them without a public download")
        else:
            log(f"  {name:16s} {role:4s} NOT {'CHECKED' if verify else 'FINGERPRINTED'}: {_first_line(info)}")
    if any(st == "partial" for _, _, st, _ in results):
        log("  incomplete sets: the content of their missing files is not excluded from anything built with these "
            "fingerprints, and the training factory cannot see it (every build says so). A local copy of the files "
            "(--from-dir) completes them; the files without a public download exist only where they were collected.")
    failed = [name for name, _, st, _ in results if st == "failed"]
    if failed and not verify:
        log(f"  without a fingerprint file ({', '.join(failed)}): every data build stops until they have one, or runs "
            "with --allow-missing-exams and records that they were not excluded; the training factory checks nothing "
            "against them and refuses a recipe that declares them in `exams:`.")
    return 1 if failed else 0


def main():
    ap = cli.parser(__doc__, "the exam fingerprints")
    ap.add_argument("--set", action="append", dest="sets", help="set name (repeatable; default: every set)")
    ap.add_argument("--from-dir", help="a directory holding local copies of listed files (matched by sha256)")
    ap.add_argument("--allow-partial", action="store_true",
                    help="also write a set whose files WITH a public download are missing (recorded as incomplete); "
                         "files without one never stop a set")
    args = ap.parse_args()
    cli.start(args)
    all_sets = manifest.exam_sets()
    names = [s["name"] for s, _ in all_sets]
    for n in args.sets or []:
        if n not in names:
            raise DataError(f"unknown set `{n}` (known: {', '.join(names)})")
    todo = [(s, m) for s, m in all_sets if not args.sets or s["name"] in args.sets]
    F = paths.fingerprint_module()
    if args.dry_run:
        for s, m in todo:
            p = paths.fingerprint_path(s["name"])
            state = "missing"
            if os.path.exists(p):
                state = "present" if F.load(p)[1].get("manifest_sha256") == m.sha256 else "stale (manifest changed)"
            local = len(exams.local_only_files(s))
            log(f"  {s['name']:22s} {s['role']:5s} {s.get('fingerprint', 'files'):10s} {len(s['entries']):4d} entries  "
                f"fingerprints: {state}" + (f"  ({local} file(s) without a public download: --from-dir)" if local else ""))
        return 0
    problems, results = [], []
    ensure_dir(paths.fingerprints_dir())
    for s, m in todo:
        path = paths.fingerprint_path(s["name"])
        log(f"{s['name']} ({s['role']})")
        try:
            fp, meta = fingerprint_set(s, m, from_dir=args.from_dir, allow_partial=args.allow_partial)
        except (DataError, OSError) as e:          # reported; the next set is done
            log(f"  not fingerprinted: {e}")
            results.append((s["name"], s["role"], "failed", str(e)))
            if args.verify:
                problems.append(f"{s['name']}: {_first_line(e)}")
            continue
        results.append((s["name"], s["role"], "complete" if meta["complete"] else "partial", meta))
        if args.verify:
            if not os.path.exists(path):
                problems.append(f"{s['name']}: no fingerprint file")
                continue
            old_fp, old_meta = F.load(path)
            if fp_digest(old_fp) != fp_digest(fp):
                problems.append(f"{s['name']}: the fingerprints on disk differ from the recomputed ones")
            if old_meta.get("manifest_sha256") != m.sha256:
                problems.append(f"{s['name']}: made from another version of {os.path.basename(m.path)}")
            continue
        F.save(path, fp, meta)
        log(f"  {meta['files']} files, {human(meta['bytes'])}, {len(fp)} fingerprints -> {paths.rel(path)}"
            + ("" if meta["complete"] else f"  (INCOMPLETE: {meta['missing_count']} file(s) missing)"))
    status = summary(results, verify=args.verify)
    if args.verify:
        return cli.finish_verify(problems, "exam fingerprints")
    return status


if __name__ == "__main__":
    sys.exit(cli.main_guard(main))
