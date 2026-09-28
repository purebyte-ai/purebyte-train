#!/usr/bin/env python3
"""Fetch the evaluation sets into $PUREBYTE_DATA/exams/<set>/, for EVALUATION only (never for training).

    python data/fetch_exams.py                      every set of data/exams/*.yaml (CredData: fetch_creddata.py)
    python data/fetch_exams.py --set code-fp-a      one set (repeatable)
    python data/fetch_exams.py --from-dir DIR       take listed files from a local copy (matched by sha256)
    python data/fetch_exams.py --dry-run            what would be fetched
    python data/fetch_exams.py --verify             re-hash what is on disk against the manifests

Repositories are written from git objects (the exact committed bytes, whatever the platform). Files taken from
packages or archives are extracted in memory and only kept when their sha256 matches the manifest. Files that have no
public download (`type: local`: 15 of bin-fp-b and 46 of bin-fp-c) can only come from --from-dir; the set manifest
written next to them says what is missing. Document sets (PII benchmarks) are sampled by pbdata.pii with fixed seeds.

A set that fails (a download that does not match its pin, a missing tool such as `zstd` for the Ubuntu packages of
bin-fp-b on Python 3.13 or older, the network) is reported and the next set is fetched; the summary at the end names
every failed or incomplete set and what that means for the exclusion, and the exit status is then 1.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pbdata import cli, exams, manifest, paths  # noqa: E402
from pbdata.util import DataError, log, read_json, sha256_file  # noqa: E402


def fetch_set(s, m, local, dry_run, allow_drift=False, sampled=None):
    mode = s.get("fingerprint", "files")
    if mode == "creddata":
        log(f"  {s['name']}: CredData is fetched by `python data/fetch_creddata.py`")
        return None
    record = dict(set=s["name"], role=s["role"], manifest=os.path.basename(m.path), manifest_sha256=m.sha256,
                  entries=[])
    if mode == "jsonl-text":
        from pbdata import pii
        record["entries"] = pii.fetch_exam_set(s, dry_run=dry_run, allow_drift=allow_drift, sampled=sampled)
    else:
        for e in s["entries"]:
            if e["type"] == "git":
                record["entries"].append(exams.fetch_git_entry(s["name"], e, dry_run=dry_run))
            elif e["type"] in ("url", "deb", "pypi", "local"):
                record["entries"].append(exams.fetch_file_entry(s["name"], e, local=local, dry_run=dry_run))
            else:
                raise DataError(f"{s['name']}:{e['id']}: type {e['type']} is not fetched by this script")
    missing = [f"{r['id']}:{x}" for r in record["entries"] for x in r.get("missing", [])]
    record["missing"] = missing
    if not dry_run:
        exams.write_set_manifest(s["name"], record)
    if missing:
        log(f"  {s['name']}: {len(missing)} file(s) have no public download and were not found locally "
            f"(use --from-dir): {', '.join(missing[:6])}{' ...' if len(missing) > 6 else ''}")
    return record


def verify_set(s):
    problems = []
    p = os.path.join(paths.exam_dir(s["name"]), "manifest.json")
    if s.get("fingerprint") == "creddata":
        return problems
    if not os.path.exists(p):
        return [f"{s['name']}: not fetched"]
    root = paths.exam_dir(s["name"])
    for e in s["entries"]:
        for f in e.get("files") or []:
            if s.get("fingerprint") == "jsonl-text":
                q = os.path.join(root, exams.file_name(f))
            else:
                q = os.path.join(root, "files", exams.file_name(f))
            if not os.path.exists(q):
                problems.append(f"{s['name']}:{e['id']}: {exams.file_name(f)} missing")
            elif f.get("sha256") and sha256_file(q) != f["sha256"]:
                problems.append(f"{s['name']}:{e['id']}: {exams.file_name(f)} does not match its sha256")
    rec = read_json(p)
    for r in rec.get("entries", []):
        if r.get("tree"):
            from pbdata import fetch
            repo = fetch.repo_cache(r["url"])
            if not fetch.has_commit(repo, r["commit"]) or fetch.tree_id(repo, r["commit"]) != r["tree"]:
                problems.append(f"{s['name']}:{r['id']}: commit {r['commit'][:12]} not cached or tree differs")
    return problems


def summary(results):
    """Prints the outcome of every set and what it means for the exclusion. Returns the exit status."""
    log("")
    log("summary:")
    for name, status, info in results:
        if status == "fetched":
            log(f"  {name:16s} fetched")
        elif status == "incomplete":
            log(f"  {name:16s} fetched without {len(info)} file(s) that have no public download")
        elif status == "creddata":
            log(f"  {name:16s} not fetched here: run `python data/fetch_creddata.py` (Linux, macOS or WSL)")
        else:
            log(f"  {name:16s} FAILED: {str(info).strip().splitlines()[0][:240]}")
    if any(st == "incomplete" for _, st, _ in results):
        log("  files without a public download (Windows system files, locally compiled files, application-bundled "
            "files) exist only where they were collected: without a copy (--from-dir) their fingerprints cannot be "
            "made, so their content is not excluded from anything built here and the training factory cannot see it "
            "(the fingerprint file and every build say so).")
    failed = [name for name, st, _ in results if st == "failed"]
    if failed:
        log(f"  failed sets ({', '.join(failed)}) have no fingerprints until they are fetched: every data build then "
            "stops, unless it is given --allow-missing-exams (it records that those sets were not excluded). Run this "
            "again for them once the cause is fixed: what was fetched is kept.")
    return 1 if failed else 0


def main():
    ap = cli.parser(__doc__, "the evaluation sets")
    ap.add_argument("--set", action="append", dest="sets", help="set name (repeatable; default: every set)")
    ap.add_argument("--from-dir", help="a directory holding local copies of listed files (matched by sha256)")
    ap.add_argument("--allow-revision-drift", action="store_true",
                    help="sample a dataset whose current revision differs from the pinned one (recorded)")
    args = ap.parse_args()
    cli.start(args)
    all_sets = manifest.exam_sets()
    names = [s["name"] for s, _ in all_sets]
    for n in args.sets or []:
        if n not in names:
            raise DataError(f"unknown set `{n}` (known: {', '.join(names)})")
    todo = [(s, m) for s, m in all_sets if not args.sets or s["name"] in args.sets]
    if args.verify:
        problems = []
        for s, _ in todo:
            problems += verify_set(s)
        return cli.finish_verify(problems, "evaluation sets")
    local = exams.LocalIndex(args.from_dir) if args.from_dir else None
    results, sampled = [], {}
    for s, m in todo:
        log(f"{s['name']} ({s['role']}): {len(s['entries'])} entries")
        try:
            rec = fetch_set(s, m, local, args.dry_run, args.allow_revision_drift, sampled)
        except (DataError, OSError) as e:          # reported; the next set is done
            log(f"  FAILED: {e}")
            results.append((s["name"], "failed", str(e)))
            continue
        if rec is None:
            results.append((s["name"], "creddata", None))
        elif rec["missing"]:
            results.append((s["name"], "incomplete", rec["missing"]))
        else:
            results.append((s["name"], "fetched", None))
    status = summary(results) if not args.dry_run else 0
    done = [name for name, st, _ in results if st in ("fetched", "incomplete")]
    if done and not args.dry_run:
        log("next: python data/fingerprint_exams.py" + ("" if not args.sets else "".join(f" --set {n}" for n in done)))
    return status


if __name__ == "__main__":
    sys.exit(cli.main_guard(main))
