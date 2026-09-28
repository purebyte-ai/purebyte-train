#!/usr/bin/env python3
"""Build the labelled real-window stream of secrets-code from CredData DEV (`train.real_windows`, `real_mode: spans`).

    python data/build_creddata_stream.py                       -> streams/creddata-dev-spans.npz + .manifest.json
    python data/build_creddata_stream.py --dry-run             the plan
    python data/build_creddata_stream.py --verify              re-hash the stream and re-run its hard check
    python data/build_creddata_stream.py --exam-overlap drop   leave out the windows that share a block with a test exam
                                                               or come from a repository of another evaluation set

What it does (the recipe is in pbdata/creddata_stream.py):
  1. CredData: fetched by fetch_creddata.py when it is missing (labelled files by FileID, CredData's own
     obfuscation), and its split, recomputed by the public scorer, compared with data/exams/creddata.yaml
  2. the TEST guard: keyed hashes of every CredData TEST credential value (the key never leaves the process)
  3. the stream from the `train` subset of CredData DEV only (117 repositories), with the hard check before any file
     is opened: 24,000 windows of 512 bytes by default, half of the credential windows re-planted with generated values
  4. the overlap with every registered exam fingerprint (the training factory refuses a real stream that shares a block
     with a `test` exam: see data/README.md), and by identity with the repositories of the other evaluation sets
     (five code-fp repositories are CredData DEV repositories); the .npz, its manifest, the hard check on the written
     file, and the comparison with the stream secrets-code 1.0.0 was trained with

Requirements: those of fetch_creddata.py when it has to download, and the credential-format catalogues
(`python data/build_secret_catalogs.py`) for the re-planted values. The stream holds CredData bytes: it is only ever
written under $PUREBYTE_DATA, and nothing here prints a value.
"""
import json
import os
import platform
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from pbdata import cli, creddata, exclusion, manifest, paths  # noqa: E402
from pbdata import creddata_stream as CS  # noqa: E402
from pbdata.util import DataError, ensure_dir, log, read_json, sha256_file, warn, write_json  # noqa: E402

OUTPUT = "streams/creddata-dev-spans.npz"
ARRAYS = ("X", "S", "Y", "kind", "repo")


def output_path(arg):
    out = cli.outside_repository(arg or os.path.join(paths.data_dir(), *OUTPUT.split("/")),
                                  "the stream (it holds CredData bytes)")
    if not out.endswith(".npz"):
        raise DataError(f"--out {arg}: the stream is an .npz file")
    return out


def manifest_path(out):
    return out[:-len(".npz")] + ".manifest.json"


def ensure_creddata(args, entry):
    base = creddata.creddata_dir()
    if not os.path.exists(os.path.join(base, "manifest.json")):
        if args.no_fetch:
            raise DataError("CredData is not fetched: run `python data/fetch_creddata.py` (or drop --no-fetch)")
        import fetch_creddata
        fetch_creddata.fetch_creddata("partial", args.jobs)
    rec = read_json(os.path.join(base, "manifest.json"))
    if rec.get("commit") != entry["pin"]["commit"]:
        raise DataError(f"the CredData on disk is at {str(rec.get('commit'))[:12]}, data/exams/creddata.yaml pins "
                        f"{entry['pin']['commit'][:12]}: fetch it again")
    return creddata.checkout_dir(), rec


def exam_overlap(arrays, meta, mode, allow_missing):
    """Windows that share a 64-byte block with a registered exam of role `test` (the training factory's check), per
    exam; with mode `drop` they are removed. Sets of role `dev` are not enforced (as in the factory): the stream IS
    CredData DEV. Returns (arrays, meta, report)."""
    F = paths.fingerprint_module()
    sets, rep = exclusion.load_all(allow_missing=(mode == "report" or allow_missing), roles=("test",))
    X, Y = arrays["X"], arrays["Y"].astype(bool)
    hit = np.zeros(len(X), dtype=bool)
    report = dict(mode=mode, checked={}, not_checked=sorted(set(rep["missing"]) | set(rep["stale"])),
                  rule="a window that shares a 64-byte block with an exam of role test (pbtrain.data.fingerprint)")
    for s in sets:
        if s.role == "dev":
            continue
        h = F.window_hits(X, s.fp)
        report["checked"][s.name] = dict(windows=int(h.sum()), positive_windows=int((h & Y).sum()),
                                         by_kind={CS.KINDS[k]: int((h & (arrays["kind"] == k)).sum())
                                                  for k in range(len(CS.KINDS))})
        hit |= h
    report["windows"] = int(hit.sum())
    if mode == "drop" and hit.any():
        keep = ~hit
        arrays = {k: v[keep] for k, v in arrays.items()}
        K = arrays["kind"]
        meta = dict(meta, windows=int(len(K)), positive_windows=int(arrays["Y"].sum()),
                    per_kind={CS.KINDS[k]: int((K == k).sum()) for k in range(len(CS.KINDS))},
                    repos_used=len({meta["repos"][i] for i in arrays["repo"].tolist()}),
                    exam_overlap_dropped=int(hit.sum()))
        report["dropped"] = int(hit.sum())
    return arrays, meta, report


def exam_repositories(exam_manifests):
    """{normalised repository URL: [exam sets]} of every repository of an evaluation set other than CredData's own
    (the stream IS CredData DEV, so the creddata sets are left out here)."""
    from pbdata.identity import norm_repo
    out = {}
    for m in exam_manifests:
        for s in m.sets:
            if s.get("fingerprint") == "creddata":
                continue
            for e in s["entries"]:
                if e["type"] == "git":
                    out.setdefault(norm_repo(e["url"]), []).append(s["name"])
    return out


def identity_overlap(arrays, meta, parts, exam_repos, mode):
    """Windows cut from a CredData DEV repository that is also a repository of another evaluation set (the identity
    rule: another commit of an evaluation repository is still that repository). Always reported; with mode `drop` those
    windows are left out. The released stream holds 641 of them (274 with a credential), from five repositories of the
    code-fp sets (data/README.md). Returns (arrays, meta, report)."""
    from pbdata.identity import norm_repo
    urls = {rid: r["url"] for p in parts.values() for rid, r in p.items()}
    R, Y = arrays["repo"], arrays["Y"].astype(bool)
    hit = np.zeros(len(R), dtype=bool)
    repos = {}
    for i, rid in enumerate(meta["repos"]):
        sets = exam_repos.get(norm_repo(urls.get(rid, "")))
        if not sets:
            continue
        sel = R == i
        hit |= sel
        repos[rid] = dict(url=urls[rid], exam_sets=sorted(set(sets)), windows=int(sel.sum()),
                          positive_windows=int((sel & Y).sum()))
    report = dict(mode=mode, repositories=repos, windows=int(hit.sum()), positive_windows=int((hit & Y).sum()),
                  rule="a window of a CredData DEV repository whose normalised URL is a repository of another "
                       "evaluation set (any commit)")
    if mode == "drop" and hit.any():
        keep = ~hit
        arrays = {k: v[keep] for k, v in arrays.items()}
        K = arrays["kind"]
        meta = dict(meta, windows=int(len(K)), positive_windows=int(arrays["Y"].sum()),
                    per_kind={CS.KINDS[k]: int((K == k).sum()) for k in range(len(CS.KINDS))},
                    repos_used=len({meta["repos"][i] for i in arrays["repo"].tolist()}),
                    exam_identity_dropped=int(hit.sum()))
        report["dropped"] = int(hit.sum())
    return arrays, meta, report


def write_npz(out, arrays, meta):
    ensure_dir(os.path.dirname(out))
    tmp = os.path.join(os.path.dirname(out), "." + os.path.basename(out)[:-4] + ".tmp.npz")
    try:
        np.savez(tmp, X=arrays["X"], S=arrays["S"], Y=arrays["Y"], kind=arrays["kind"], repo=arrays["repo"],
                 meta=np.array(json.dumps(meta, sort_keys=True)))
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def stored_digests(out):
    with np.load(out) as z:
        return CS.array_digests({k: z[k] for k in ARRAYS})


def released_line(same, reasons):
    if same:
        return "released: the arrays are bit for bit those of the stream secrets-code 1.0.0 was trained with"
    return "released: this stream differs from the one secrets-code 1.0.0 was trained with -- " + "; ".join(reasons)


def verify(out, split):
    problems = []
    mp = manifest_path(out)
    if not (os.path.exists(out) and os.path.exists(mp)):
        return cli.finish_verify([f"{paths.rel(out)} or its manifest is missing: build it first"],
                                 "the CredData stream")
    man = read_json(mp)
    if sha256_file(out) != man["output"]["sha256"]:
        problems.append(f"{paths.rel(out)}: sha256 differs from its manifest")
    try:
        meta = CS.check_file(out, split)
        digests = stored_digests(out)
        if digests != man["arrays"]:
            problems.append("the arrays differ from the manifest")
        params = {k: meta[k] for k in CS.DEFAULTS}
        same, reasons = CS.compare_with_released(digests, params, split, man["inputs"]["catalogs"]["signature"])
        left_out = int(meta.get("exam_overlap_dropped", 0)) + int(meta.get("exam_identity_dropped", 0))
        if left_out:
            reasons = [f"built with --exam-overlap drop ({left_out} windows left out)"]
        log(released_line(same, reasons))
    except (DataError, KeyError, ValueError) as e:
        problems.append(str(e))
    return cli.finish_verify(problems, "the CredData stream")


def main():
    ap = cli.parser(__doc__, "the CredData DEV stream", excludes=True, fetches=True)
    d = CS.DEFAULTS
    ap.add_argument("--n", type=int, default=d["n"], help=f"windows (default {d['n']})")
    ap.add_argument("--seq-len", type=int, default=d["seq_len"], help=f"bytes per window (default {d['seq_len']})")
    ap.add_argument("--seed", type=int, default=d["seed"], help=f"seed of the window sampler (default {d['seed']})")
    ap.add_argument("--replant", type=float, default=d["replant"],
                    help=f"share of the credential windows re-planted with a generated value and its contrast "
                         f"(default {d['replant']})")
    ap.add_argument("--exam-overlap", choices=("report", "drop"), default="report",
                    help="report (default): record the windows that share a block with a test exam, and those of "
                         "CredData repositories that are also repositories of another evaluation set; drop: leave both "
                         "out (the training factory refuses a stream that shares a block; the result is then no longer "
                         "the released stream)")
    ap.add_argument("--out", default=None, help=f"output .npz (default $PUREBYTE_DATA/{OUTPUT})")
    ap.add_argument("--jobs", type=int, default=4, help="repositories fetched at a time when CredData is missing")
    args = ap.parse_args()
    cli.start(args)
    out = output_path(args.out)
    m, entry, parts = creddata.yaml_partitions()
    split = creddata.pinned_split(parts)
    if args.verify:
        return verify(out, split)
    base = creddata.creddata_dir()
    if args.dry_run:
        fetched = os.path.exists(os.path.join(base, "manifest.json"))
        log(f"CredData @ {entry['pin']['commit'][:12]}: " + ("fetched" if fetched else
            ("MISSING (--no-fetch: the build would fail)" if args.no_fetch else "would be fetched first (partial)")))
        cats = [c for c in ("gitleaks.json", "trufflehog.json")
                if not os.path.exists(os.path.join(paths.catalogs_dir(), c))]
        log("catalogues: " + (f"MISSING {cats}: run data/build_secret_catalogs.py" if cats else "present"))
        present = [name for name, role, st in cli.exam_status() if role == "test" and st == "present"]
        log(f"exam overlap ({args.exam_overlap}) against the test exams fingerprinted so far: "
            f"{', '.join(present) or 'none'}" + ("" if args.exam_overlap == "report" else
                                                   " (drop needs all of them, or --allow-missing-exams)"))
        log(f"stream: {args.n} windows of {args.seq_len} bytes, seed {args.seed}, replant {args.replant}, "
            f"exam overlap {args.exam_overlap} -> {paths.rel(out)}")
        return 0
    if args.exam_overlap == "drop" and not args.allow_missing_exams:
        absent = [name for name, role, st in cli.exam_status() if role == "test" and st != "present"]
        if absent:
            raise DataError(f"--exam-overlap drop needs the fingerprint of every test exam; missing or stale: "
                            f"{', '.join(absent)}. Run data/fetch_exams.py and data/fingerprint_exams.py first "
                            f"(or pass --allow-missing-exams: the manifest then says which were not checked)")
    root, cd_record = ensure_creddata(args, entry)
    scorer = creddata.load_scorer()
    computed = dict(scorer.CredData(root).split)
    if computed != split:
        raise DataError("the split the public scorer computes from this CredData checkout is not the one "
                        "data/exams/creddata.yaml pins: another snapshot? Stop and look")
    G, C = CS.load_generator()
    signature = CS.catalog_signature(C)
    keyed = CS.keyed_hasher()
    test_hashes, test_info = CS.test_value_hashes(root, split, keyed)
    log(f"TEST guard: {test_info['units']} credentials in {test_info['files']} CredData TEST files, "
        f"{test_info['distinct_values']} distinct values (keyed hashes, in memory only)")
    params = dict(n=args.n, seq_len=args.seq_len, seed=args.seed, replant=args.replant)
    arrays, meta = CS.build(root, split, G, scorer.macro_category, test_hashes, keyed, **params,
                            source=dict(dataset="CredData", url=entry["url"], commit=entry["pin"]["commit"]))
    del test_hashes
    released_same, released_reasons = CS.compare_with_released(CS.array_digests(arrays), params, split, signature)
    arrays, meta, overlap = exam_overlap(arrays, meta, args.exam_overlap, args.allow_missing_exams)
    arrays, meta, ident = identity_overlap(arrays, meta, parts, exam_repositories(manifest.load_exams()),
                                           args.exam_overlap)
    dropped = int(overlap.get("dropped", 0)) + int(ident.get("dropped", 0))
    if args.exam_overlap == "drop" and dropped:
        released_same, released_reasons = False, [f"--exam-overlap drop left out {dropped} windows"]
    write_npz(out, arrays, meta)
    CS.check_file(out, split)                                  # the hard check, on the written file
    digests = stored_digests(out)
    catalogs = {c: sha256_file(os.path.join(paths.catalogs_dir(), c)) for c in ("gitleaks.json", "trufflehog.json")}
    gen_dir = os.path.join(paths.REPO, "specialists", "secrets-code", "secretsgen")
    write_json(manifest_path(out), dict(
        builder="data/build_creddata_stream.py", parameters=params, exam_overlap_mode=args.exam_overlap,
        inputs=dict(
            creddata=dict(url=entry["url"], commit=entry["pin"]["commit"], snapshot_sha256=cd_record["snapshot_sha256"],
                          dev_content_sha256=cd_record["partitions"]["dev"]["content_sha256"],
                          test_content_sha256=cd_record["partitions"]["test"]["content_sha256"]),
            split=dict(seed=scorer.SPLIT_SEED, sha256=creddata.split_digest(split), manifest_sha256=m.sha256),
            generator={f: sha256_file(os.path.join(gen_dir, f)) for f in ("credentials.py", "catalog.py")},
            catalogs=dict(catalogs, signature=signature)),
        environment=dict(python=platform.python_version(), numpy=np.__version__, system=platform.system()),
        test_guard=test_info, stream={k: v for k, v in meta.items() if k not in ("repos", "source")},
        repos=meta["repos"], exam_overlap=overlap, exam_identity=ident, arrays=digests,
        output=dict(path=paths.rel(out), sha256=sha256_file(out), bytes=os.path.getsize(out)),
        released=dict(identical=released_same, reasons=released_reasons)))
    pk = meta["per_kind"]
    log(f"stream: {meta['windows']} windows ({meta['positive_windows']} with a credential) from {meta['repos_used']} "
        f"CredData DEV repositories: {pk['original']} original, {pk['replanted']} re-planted, {pk['contrast']} "
        f"contrast, {pk['f_negative']} hard negative, {pk['background']} background")
    c = meta["counts"]
    log(f"  {c.get('T_units_hash_in_TEST', 0)} DEV credential(s) left out by the TEST guard, {c.get('X_units', 0)} X "
        f"units and {c.get('uuid_ignored_not_credential_name', 0)} UUID spans never touched")
    for name, r in sorted(overlap["checked"].items()):
        if r["windows"]:
            what = (": left out" if args.exam_overlap == "drop" else
                    ": the training factory refuses such a stream (see data/README.md; --exam-overlap drop)")
            warn(f"{r['windows']} window(s) ({r['positive_windows']} with a credential) share a block with the exam "
                 f"`{name}`{what}")
    if overlap["not_checked"]:
        log(f"  exam overlap not checked against {', '.join(overlap['not_checked'])} (no fingerprint file)")
    if ident["windows"]:
        names = ", ".join(f"{r['url'].rstrip('/').rsplit('/', 1)[-1]} ({'/'.join(r['exam_sets'])})"
                          for r in ident["repositories"].values())
        warn(f"{ident['windows']} window(s) ({ident['positive_windows']} with a credential) come from "
             f"{len(ident['repositories'])} CredData DEV repositories that are also repositories of other evaluation "
             f"sets: {names}" + (": left out" if args.exam_overlap == "drop" else
                                 ": a model trained on them has seen those repositories (data/README.md; "
                                 "--exam-overlap drop leaves them out)"))
    log(released_line(released_same, released_reasons))
    log(f"-> {paths.rel(out)} ({os.path.getsize(out) / 2 ** 20:.1f} MiB) and its manifest")
    return 0


if __name__ == "__main__":
    sys.exit(cli.main_guard(main))
