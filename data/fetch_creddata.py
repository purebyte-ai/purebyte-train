#!/usr/bin/env python3
"""Fetch CredData with its official obfuscation and its DEV/TEST split (evaluation, and the labelled DEV stream).

    python data/fetch_creddata.py                     checkout at the pinned commit, the labelled files by FileID,
                                                      CredData's own obfuscation, split, manifest
    python data/fetch_creddata.py --method official   the same through CredData's own download_data.py
    python data/fetch_creddata.py --dry-run           the plan
    python data/fetch_creddata.py --skip-download     CredData/data is already there: verify it and write the split
    python data/fetch_creddata.py --verify            re-check checkout, snapshot, split and dataset digests

Everything lands in $PUREBYTE_DATA/exams/creddata/:
    CredData/        the Samsung/CredData checkout at the commit pinned in data/exams/creddata.yaml, with data/ holding
                     the labelled files of the 337 upstream repositories at their commits, every credential value
                     OBFUSCATED by CredData's own obfuscate_creds.py
    split.json       repository -> dev | test, computed by importing benchmarks/creddata/score.py (the public scorer):
                     the split is the scorer's by construction, and it must equal the lists in creddata.yaml
    manifest.json    pins, snapshot digest, per-partition repositories, files, bytes and content digest

Two ways to get the files, same result:
    partial   (default) per repository, its commit and trees only (`git fetch --filter=blob:none`), the FileID of every
              path (sha256(path)[:8], as CredData computes it), then exactly the blobs the meta lists, written into
              CredData's layout; about 1 GB of downloads, resumable (fetch-status.json). Then obfuscate_creds.py once
    official  CredData's download_data.py: it clones every repository whole at its commit (tens of GB) and obfuscates;
              its output is captured (the lines where the obfuscation quotes a value are only counted)

Requirements: Linux, macOS or WSL (the obfuscation writes files in text mode, so on Windows the line endings would
differ; reading the dataset on Windows can also trigger antivirus quarantines of security-tool sources), git, and
CredData's own dependency `pybase62` (`pip install pybase62==1.0.0`; the scripts in data/ never import it).
"""
import concurrent.futures
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pbdata import cli, creddata, fetch, paths  # noqa: E402
from pbdata.creddata import load_scorer, yaml_partitions  # noqa: E402,F401  (kept importable from this script)
from pbdata.util import DataError, ensure_dir, log, read_json, sha256_file, write_json  # noqa: E402


def split_of(root, scorer=None):
    """{repo id: 'dev' | 'test'} exactly as the public scorer computes it for this checkout."""
    scorer = scorer or load_scorer()
    cd = scorer.CredData(root)
    return dict(cd.split), scorer, cd


def checkout(entry, root, dry_run):
    commit = entry["pin"]["commit"]
    if os.path.isdir(os.path.join(root, ".git")):
        head = fetch.git("rev-parse", "HEAD", cwd=root).decode().strip()
        if head != commit:
            raise DataError(f"{root} is at {head[:12]}, not at the pinned {commit[:12]}: remove it and fetch again")
        return
    if dry_run:
        log(f"  would clone {entry['url']} @ {commit[:12]} into {paths.rel(root)}")
        return
    ensure_dir(root)
    fetch.git("init", "-q", cwd=root)
    fetch.git("config", "core.autocrlf", "false", cwd=root)
    fetch.git("remote", "add", "origin", entry["url"], cwd=root)
    log(f"  fetching {entry['url']} @ {commit[:12]}")
    fetch.git("-c", "protocol.version=2", "fetch", "-q", "--depth", "1", "origin", commit, cwd=root)
    fetch.git("checkout", "-q", "FETCH_HEAD", cwd=root)


def check_snapshot(entry, root, parts):
    snap_path = os.path.join(root, "snapshot.json")
    want = {f["path"]: f["sha256"] for f in entry.get("files") or []}
    got = sha256_file(snap_path)
    if want.get("snapshot.json") and got != want["snapshot.json"]:
        raise DataError(f"snapshot.json has sha256 {got}, the manifest pins {want['snapshot.json']}")
    with open(snap_path, encoding="utf-8") as f:
        snap = json.load(f)
    listed = {}
    for rid, url in snap.items():
        listed[creddata.new_repo_id(rid)] = (url, rid[:40])
    declared = {rid: r for p in parts.values() for rid, r in p.items()}
    if set(listed) != set(declared):
        raise DataError(f"snapshot.json lists {len(listed)} repositories, creddata.yaml {len(declared)}: they differ")
    for rid, (url, commit) in listed.items():
        if declared[rid]["commit"] != commit or declared[rid]["url"].rstrip("/") != url.rstrip("/"):
            raise DataError(f"repository {rid}: snapshot.json and creddata.yaml disagree on URL or commit")
    return snap


def partition_digest(data_root, repo_ids):
    """(files, bytes, sha256 over sorted relative path + file sha256) of the dataset files of some repositories."""
    h, files, nbytes = hashlib.sha256(), 0, 0
    for rid in sorted(repo_ids):
        base = os.path.join(data_root, rid)
        if not os.path.isdir(base):
            continue
        for dp, dns, fns in os.walk(base):
            dns.sort()
            for fn in sorted(fns):
                p = os.path.join(dp, fn)
                rel = os.path.relpath(p, data_root).replace("\\", "/")
                h.update(f"{rel}\0{sha256_file(p)}\n".encode("utf-8"))
                files += 1
                nbytes += os.path.getsize(p)
    return files, nbytes, h.hexdigest()


def run_official_download(root, base, jobs):
    """CredData's own download_data.py, unmodified. It ends by running obfuscate_creds.py, which logs every labelled
    value it leaves unchanged: its whole output is captured (pbdata/creddata.py, `run_captured`), its progress lines
    are shown, those value lines are only counted, and nothing else is printed unless it fails."""
    if os.name == "nt":
        raise DataError("CredData's download script needs a POSIX shell and file names Windows cannot store: run "
                        "this step on Linux, macOS or inside WSL")
    try:
        __import__("base62")
    except ImportError:
        raise DataError("CredData's download script needs its own requirement `pybase62`: run "
                        f"`{sys.executable} -m pip install -r {os.path.join(root, 'requirements.txt')}` "
                        "(or install pybase62), then run this again") from None
    script = os.path.join(root, "obfuscate_creds.py")
    write_json(os.path.join(base, creddata.OBFUSCATION_STATE),
               dict(status="running", method="official", script_sha256=sha256_file(script)))
    log(f"  running CredData's download_data.py (jobs {jobs}); this clones 337 repositories whole. Its output is "
        f"captured: progress lines only")
    code, same, tail = creddata.run_captured(
        [sys.executable, "download_data.py", "--data_dir", "data", "--jobs", str(jobs)], root, show=log)
    if code != 0:
        raise DataError(f"CredData's download_data.py failed (exit {code}); its last lines that cannot hold a value: "
                        + (" | ".join(tail) or "none") + f". Remove {os.path.join(root, 'data')} and "
                        f"{os.path.join(base, creddata.OBFUSCATION_STATE)} before running it again")
    write_json(os.path.join(base, creddata.OBFUSCATION_STATE),
               dict(status="done", method="official", script_sha256=sha256_file(script), values_left_identical=same))
    log(f"  downloaded and obfuscated ({same} value(s) the official script left identical, not shown)")


def run_partial_download(root, base, snap, jobs):
    """The labelled files of every repository, by FileID (pbdata/creddata.py), resuming from fetch-status.json; then
    CredData's obfuscation, once."""
    state = creddata.obfuscation_state(base)
    if state and state.get("status") == "done":
        log("  CredData/data is fetched and obfuscated: not downloading again (remove it to rebuild)")
        return
    if state and state.get("status") == "running":
        creddata.obfuscate(root, base)          # raises: an interrupted obfuscation left the files in an unknown state
    if state is None and os.path.isdir(os.path.join(root, "data")) and \
            not os.path.exists(os.path.join(base, creddata.FETCH_STATUS)):
        raise DataError(f"{os.path.join(root, 'data')} exists but was not written by this script (an official "
                        "download?): pass --method official --skip-download to use it, or remove it")
    if creddata.TEXT_MODE_WRITES_CRLF:
        raise DataError("CredData's obfuscate_creds.py writes files in text mode, so on Windows their line endings "
                        "would differ from everyone else's: run this on Linux, macOS or inside WSL")
    wanted = creddata.wanted_files(os.path.join(root, "meta"))
    with open(os.path.join(root, "word_in_path.json"), encoding="utf-8") as f:
        words = json.load(f)
    status_path = os.path.join(base, creddata.FETCH_STATUS)
    status = read_json(status_path) if os.path.exists(status_path) else {}
    todo = []
    for sid, url in snap.items():
        rid = creddata.new_repo_id(sid)
        st = status.get(rid) or {}
        want = wanted.get(rid, {})
        done = not st.get("error") and st.get("got") == len(want) and \
            not creddata.missing_files(root, {rid: want})
        if not done:
            todo.append((sid, url, rid, want))
    log(f"  partial fetch: {len(snap)} repositories, {len(snap) - len(todo)} already complete, {len(todo)} to fetch "
        f"({jobs} at a time)")
    scratch = ensure_dir(os.path.join(base, "tmp"))
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
        futures = [pool.submit(creddata.fetch_repository, root, sid, url, want, words, scratch)
                   for sid, url, rid, want in todo]
        for k, fut in enumerate(concurrent.futures.as_completed(futures), 1):
            st = fut.result()
            status[st["repo"]] = st
            write_json(status_path, status)
            log(f"  [{k}/{len(todo)}] {st['repo']} {st['got']}/{st['wanted']} files, {st['licences']} licence file(s)"
                + (f"  ERROR {st['error']}" if st["error"] else ""))
    try:
        os.rmdir(scratch)                                     # every scratch repository is already deleted
    except OSError:
        pass
    missing = creddata.missing_files(root, wanted)
    if missing:
        raise DataError(f"{sum(missing.values())} labelled file(s) of {len(missing)} repositories did not arrive "
                        f"({', '.join(sorted(missing)[:8])}...): run this again to resume (repositories deleted "
                        f"upstream cannot be fetched; see fetch-status.json)")
    log("  running CredData's obfuscate_creds.py over the fetched files (once)")
    state = creddata.obfuscate(root, base)
    log(f"  obfuscated ({state['values_left_identical']} value(s) the official script left identical, not shown)")


def fetch_creddata(method="partial", jobs=4, skip_download=False, dry_run=False):
    """Checkout, files, split and manifest. Returns the checkout root (None on a dry run)."""
    m, entry, parts = yaml_partitions()
    base = creddata.creddata_dir()
    root = creddata.checkout_dir()
    if dry_run:
        log(f"CredData {entry['url']} @ {entry['pin']['commit'][:12]} -> {paths.rel(root)}")
        checkout(entry, root, dry_run=True)
        if method == "partial":
            log(f"  then the labelled files of 337 repositories by FileID ({jobs} at a time) and CredData's "
                "obfuscate_creds.py")
        else:
            log(f"  then CredData's download_data.py --data_dir data --jobs {jobs} (337 repositories, cloned whole)")
        log(f"  then the split by {os.path.relpath(paths.CREDDATA_SCORER, paths.REPO).replace(os.sep, '/')}: "
            f"{len(parts.get('test', {}))} TEST and {len(parts.get('dev', {}))} DEV repositories expected")
        return None
    checkout(entry, root, dry_run=False)
    snap = check_snapshot(entry, root, parts)
    if not skip_download:
        state = creddata.obfuscation_state(base) or {}
        if method == "partial":
            run_partial_download(root, base, snap, jobs)
        elif state.get("status") == "running":
            raise DataError(f"an earlier download of CredData was interrupted: {os.path.join(root, 'data')} is in an "
                            f"unknown state. Remove it and {os.path.join(base, creddata.OBFUSCATION_STATE)}, then run "
                            f"this again")
        elif os.path.isdir(os.path.join(root, "data")):
            log("  CredData/data exists: not downloading again (remove it to rebuild)")
        else:
            run_official_download(root, base, jobs)
    if not os.path.isdir(os.path.join(root, "data")):
        raise DataError("CredData/data is missing (the download did not run or failed)")
    split, scorer, cd = split_of(root)
    for part in ("test", "dev"):
        computed = {r for r, p in split.items() if p == part}
        if computed != set(parts[part]):
            raise DataError(f"the public scorer puts {len(computed)} repositories in {part}, creddata.yaml lists "
                            f"{len(parts[part])}, and they differ: the snapshot or the scorer changed. Stop and look.")
    write_json(os.path.join(base, "split.json"), dict(
        seed=scorer.SPLIT_SEED, repositories=dict(sorted(split.items())),
        scorer=os.path.relpath(paths.CREDDATA_SCORER, paths.REPO).replace("\\", "/"),
        scorer_sha256=sha256_file(paths.CREDDATA_SCORER), creddata_commit=entry["pin"]["commit"]))
    state = creddata.obfuscation_state(base) or {}
    record = dict(set_manifest=os.path.basename(m.path), manifest_sha256=m.sha256, url=entry["url"],
                  commit=entry["pin"]["commit"], snapshot_sha256=sha256_file(os.path.join(root, "snapshot.json")),
                  repositories=len(snap), missing_labelled_files=cd.missing_files, partitions={},
                  method=method if not skip_download else "existing",
                  obfuscation_script_sha256=state.get("script_sha256"))
    for part in ("test", "dev"):
        files, nbytes, digest = partition_digest(os.path.join(root, "data"), parts[part])
        record["partitions"][part] = dict(set=f"creddata-{part}", repositories=len(parts[part]), files=files,
                                          bytes=nbytes, content_sha256=digest)
        log(f"  {part}: {len(parts[part])} repositories, {files} files, {nbytes / 2 ** 20:.1f} MiB")
    if cd.missing_files:
        log(f"  {cd.missing_files} labelled file(s) are missing on disk (an antivirus?): the scorer leaves them out")
    write_json(os.path.join(base, "manifest.json"), record)
    return root


def main():
    ap = cli.parser(__doc__, "CredData")
    ap.add_argument("--method", choices=("partial", "official"), default="partial",
                    help="partial: the labelled files only, by FileID (default); official: CredData's download_data.py")
    ap.add_argument("--jobs", type=int, default=4, help="repositories fetched at a time")
    ap.add_argument("--skip-download", action="store_true", help="CredData/data is already built: do not run it")
    args = ap.parse_args()
    cli.start(args)
    if args.verify:
        m, entry, parts = yaml_partitions()
        base, root = creddata.creddata_dir(), creddata.checkout_dir()
        problems = []
        try:
            checkout(entry, root, dry_run=True)
            check_snapshot(entry, root, parts)
            split, _, _ = split_of(root)
            for part in ("test", "dev"):
                if {r for r, p in split.items() if p == part} != set(parts[part]):
                    problems.append(f"the scorer's {part} split differs from creddata.yaml")
            rec = read_json(os.path.join(base, "manifest.json"))
            for part in ("test", "dev"):
                files, nbytes, digest = partition_digest(os.path.join(root, "data"), parts[part])
                if digest != rec["partitions"][part]["content_sha256"]:
                    problems.append(f"{part}: dataset files differ from the manifest written at fetch time")
        except (DataError, OSError, KeyError) as e:
            problems.append(str(e))
        return cli.finish_verify(problems, "CredData")
    fetch_creddata(args.method, args.jobs, args.skip_download, args.dry_run)
    if not args.dry_run:
        log("next: python data/fingerprint_exams.py --set creddata-test --set creddata-dev")
    return 0


if __name__ == "__main__":
    sys.exit(cli.main_guard(main))
