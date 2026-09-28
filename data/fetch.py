#!/usr/bin/env python3
"""Download and verify the inputs of the source manifests into $PUREBYTE_DATA/cache.

    python data/fetch.py                                   every data/sources/*.yaml
    python data/fetch.py data/sources/binary-context.yaml  one manifest
    python data/fetch.py --only django-django,...          some entries
    python data/fetch.py --dry-run                         what would be downloaded
    python data/fetch.py --verify                          re-hash what is cached (sha256 / commit) and report

git entries are fetched at their pinned commit (depth 1; for the code corpus only files up to the build's size limit);
files are downloaded to a content-addressed cache and verified against their pinned sha256 before they are kept.
Hugging Face entries are fetched by `fetch_pii.py` (they are sampled, not mirrored). The builders call the same code,
so running this first is optional; it is useful to download once and build offline (`--no-fetch`).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pbdata import cli, fetch, manifest  # noqa: E402
from pbdata.util import DataError, log, sha256_file  # noqa: E402


def blob_limit(man, e):
    take = e.get("take") or {}
    if "blob_limit" in take:
        return take["blob_limit"] or None
    return (man.doc.get("build") or {}).get("file_bytes", {}).get("max")


def main():
    ap = cli.parser(__doc__, "the downloads")
    ap.add_argument("manifests", nargs="*", help="source manifests (default: every data/sources/*.yaml)")
    ap.add_argument("--only", help="comma-separated entry ids")
    args = ap.parse_args()
    cli.start(args)
    mans = [manifest.load(p) for p in args.manifests] if args.manifests else manifest.load_sources()
    for m in mans:
        if m.kind != "source":
            raise DataError(f"{m.path} is an exam manifest: use fetch_exams.py (exams are fetched apart on purpose)")
    cli.identity_preamble()
    only = set(args.only.split(",")) if args.only else None
    problems, done = [], 0
    for m in mans:
        log(f"{m.name} ({os.path.basename(m.path)})")
        for e in m.entries:
            if only and e["id"] not in only:
                continue
            t = e["type"]
            if t == "hf":
                log(f"  {e['id']}: Hugging Face dataset, sampled by fetch_pii.py")
                continue
            if t == "git":
                repo = fetch.repo_cache(e["url"])
                if args.verify:
                    if not fetch.has_commit(repo, e["pin"]["commit"]):
                        problems.append(f"{m.name}:{e['id']}: commit {e['pin']['commit'][:12]} not cached")
                    continue
                fetch.fetch_git(e["url"], e["pin"]["commit"], blob_limit=blob_limit(m, e), dry_run=args.dry_run)
            else:
                pin = e.get("pin") or {}
                if args.verify:
                    p = fetch.blob_path(pin["sha256"]) if pin.get("sha256") else fetch.unpinned_path(e["url"])
                    if not os.path.exists(p):
                        problems.append(f"{m.name}:{e['id']}: not cached")
                    elif pin.get("sha256") and sha256_file(p) != pin["sha256"]:
                        problems.append(f"{m.name}:{e['id']}: cached file does not match its sha256 (delete it and fetch)")
                    continue
                fetch.fetch_file(e, dry_run=args.dry_run)
            done += 1
    if args.verify:
        return cli.finish_verify(problems, "the cache")
    log(f"{done} entries {'planned' if args.dry_run else 'fetched and verified'}")
    return 0


if __name__ == "__main__":
    sys.exit(cli.main_guard(main))
