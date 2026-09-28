#!/usr/bin/env python3
"""Fetch and build the training resources of the PII specialist (data/sources/pii.yaml).

    python data/fetch_pii.py lists      public lists -> $PUREBYTE_DATA/pii/lists/*.tsv, LICENSES.md, SHA256.json
    python data/fetch_pii.py train      public TRAIN documents -> $PUREBYTE_DATA/pii/public_train/*.jsonl, MANIFEST.json
    python data/fetch_pii.py all        both
Options:
    --dry-run                   the plan
    --verify                    re-hash the outputs against the manifests written by the last build
    --raw-dir DIR               take raw list files from a local copy (matched by name and sha256) instead of downloading
    --accept-upstream-change    use a raw file whose publisher replaced it in place (recorded as a deviation)
    --skip-sentences            leave out the two Tatoeba sentence lists (the pii release recipe never reads them)
    --allow-revision-drift      sample a dataset whose current revision differs from the pinned one (recorded)
    --allow-missing-exams       build even if an exam fingerprint file is missing (recorded)

The evaluation sets of the PII specialist (data/exams/pii.yaml) are fetched by fetch_exams.py, never here. Every list
line and every TRAIN document that shares a 64-byte block with any evaluation set is dropped (and a TRAIN document whose
normalised text equals an evaluation document's, when the evaluation files are on disk); the counts are recorded.
"""
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np  # noqa: E402

from pbdata import cli, manifest, paths, pii, smoke  # noqa: E402
from pbdata.exclusion import ContentFilter, load_all  # noqa: E402
from pbdata.util import DataError, ensure_dir, log, read_json, sha256_bytes, sha256_file, write_json  # noqa: E402

MANIFEST = os.path.join(paths.SOURCE_MANIFESTS, "pii.yaml")


def exclude_lines(cf, data):
    """(kept bytes, dropped line count, {set: lines}): lines of a TSV/JSONL file that share a block with an exam go."""
    sh = cf.shared(data)
    if not sh:
        return data, 0, {}
    lines = data.split(b"\n")
    starts = np.cumsum([0] + [len(x) + 1 for x in lines[:-1]])
    drop, per = set(), {}
    for name, fps in sh.items():
        cov = cf.covered(data, {name: fps}, 256, 128)
        for i, (a, ln) in enumerate(zip(starts, lines)):
            if i and cov[a:a + len(ln)].any():
                drop.add(i)
                per[name] = per.get(name, 0) + 1
    kept = b"\n".join(ln for i, ln in enumerate(lines) if i not in drop)
    return kept, len(drop), per


def norm_hash(text):
    return hashlib.sha1(re.sub(r"\s+", " ", text).strip().encode("utf-8")).hexdigest()


def exam_text_hashes():
    """Normalised-text digests of every document of the PII evaluation sets that are on disk."""
    out = {}
    for s, _ in manifest.exam_sets():
        if s.get("fingerprint") != "jsonl-text":
            continue
        root = paths.exam_dir(s["name"])
        if not os.path.isdir(root):
            continue
        for fn in sorted(os.listdir(root)):
            if fn.endswith(".jsonl"):
                with open(os.path.join(root, fn), encoding="utf-8") as f:
                    for line in f:
                        if line.strip():
                            out.setdefault(norm_hash(json.loads(line).get("text") or ""), s["name"])
    return out


def build_lists(args, man, cf, exam_report):
    out_dir = os.path.join(paths.pii_dir(), "lists")
    raw = pii.RawFiles(man, raw_dir=args.raw_dir, accept_change=args.accept_upstream_change)
    lists = pii.Lists(raw, out_dir, man.doc.get("build") or {})
    lists.build_all(sentences=not args.skip_sentences)
    dropped = {}
    for name in sorted(os.listdir(out_dir)):
        if name.endswith(".tsv"):
            p = os.path.join(out_dir, name)
            with open(p, "rb") as f:
                data = f.read()
            kept, n, per = exclude_lines(cf, data)
            if n:
                with open(p, "wb") as f:
                    f.write(kept)
                dropped[name] = dict(lines=n, by_exam=per)
                log(f"  {name}: {n} line(s) shared a block with an exam and were dropped")
    with open(os.path.join(out_dir, "LICENSES.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(pii.LICENSES_MD)
    record = dict(builder="data/fetch_pii.py lists", manifest=dict(path="data/sources/pii.yaml", sha256=man.sha256),
                  lists={n: sha256_file(os.path.join(out_dir, n)) for n in sorted(os.listdir(out_dir))
                         if n.endswith(".tsv")},
                  rows=lists.counts, raw=raw.used, deviations=raw.deviations,
                  sentences=("left out (--skip-sentences)" if args.skip_sentences else "built"),
                  exclusion=dict(exam_report, lines_dropped=dropped))
    write_json(os.path.join(out_dir, "SHA256.json"), record)
    log(f"lists -> {paths.rel(out_dir)}" + (f"  ({len(raw.deviations)} upstream deviation(s) recorded)"
                                            if raw.deviations else ""))


def build_train(args, man, cf, exam_report):
    out_dir = os.path.join(paths.pii_dir(), "public_train")
    ensure_dir(out_dir)
    left = [r for r in smoke.standins_module().in_place(paths.data_dir()) if r.startswith("pii/public_train/")]
    if left:                                  # never deleted here: said, and refused by the plugin outside a smoke run
        log(f"  note: {', '.join(left)} is a made-up stand-in of data/build_smoke_data.py; the pii plugin reads every "
            f".jsonl of this folder and refuses a stand-in outside the smoke recipe: delete it once you train for real")
    per_source = args.per_source or (man.doc.get("build") or {}).get("train_documents_per_source", 12000)
    exam_hashes = exam_text_hashes()
    if not exam_hashes:
        log("  (no PII evaluation documents on disk: the exact-duplicate check is skipped; fingerprints still apply)")
    record = dict(builder="data/fetch_pii.py train", manifest=dict(path="data/sources/pii.yaml", sha256=man.sha256),
                  per_source=per_source, sources={}, exclusion=exam_report)
    for e in man.entries:
        if e["type"] != "hf":
            continue
        s = e.get("sampling") or {}
        name = s.get("output", e["id"] + ".jsonl")
        revs = []
        pii.check_revision(e, args.allow_revision_drift, revs)
        log(f"  {e['id']}: sampling {e['dataset']} ({', '.join(e['splits'])})")
        docs = pii.sample_train(e, per_source)
        sample = pii.jsonl_bytes(docs)
        kept, drop_fp, drop_dup = [], {}, 0
        for d in docs:
            sh = cf.shared(d["text"].encode("utf-8"))
            if sh:
                for k in sh:
                    drop_fp[k] = drop_fp.get(k, 0) + 1
                continue
            if norm_hash(d["text"]) in exam_hashes:
                drop_dup += 1
                continue
            kept.append(d)
        data = pii.jsonl_bytes(kept)
        with open(os.path.join(out_dir, name), "wb") as f:
            f.write(data)
        ref = s.get("reference_sha256")
        record["sources"][e["id"]] = dict(
            dataset=e["dataset"], split=e["splits"], revision=revs, licence=e["licence"], output=name,
            sampled=len(docs), kept=len(kept), dropped_sharing_blocks=drop_fp, dropped_exact_duplicates=drop_dup,
            sha256=sha256_bytes(data), sample_sha256=sha256_bytes(sample), reference_sha256=ref,
            sample_matches_reference=(sha256_bytes(sample) == ref) if ref else None,
            langs={lang: sum(1 for d in kept if d["lang"] == lang) for lang in ("es", "en")})
        log(f"    {len(kept)} kept of {len(docs)}" + (" (sample identical to the reference build)"
                                                       if ref and sha256_bytes(sample) == ref else ""))
        write_json(os.path.join(out_dir, "MANIFEST.json"), record)
    log(f"public TRAIN documents -> {paths.rel(out_dir)}")


def verify(args):
    problems = []
    for sub, key in (("lists", "SHA256.json"), ("public_train", "MANIFEST.json")):
        p = os.path.join(paths.pii_dir(), sub, key)
        if not os.path.exists(p):
            problems.append(f"pii/{sub}: not built")
            continue
        rec = read_json(p)
        if sub == "lists":
            for n, sha in rec["lists"].items():
                q = os.path.join(paths.pii_dir(), sub, n)
                if not os.path.exists(q) or sha256_file(q) != sha:
                    problems.append(f"pii/lists/{n}: missing or changed")
        else:
            for sid, v in rec["sources"].items():
                q = os.path.join(paths.pii_dir(), sub, v["output"])
                if not os.path.exists(q) or sha256_file(q) != v["sha256"]:
                    problems.append(f"pii/public_train/{v['output']}: missing or changed")
    return cli.finish_verify(problems, "PII resources")


def main():
    ap = cli.parser(__doc__, "the PII resources", excludes=True)
    ap.add_argument("what", choices=("lists", "train", "all"))
    ap.add_argument("--raw-dir", help="local copies of the raw list files (matched by name and sha256)")
    ap.add_argument("--accept-upstream-change", action="store_true",
                    help="use a raw list file its publisher replaced in place (INE yearly, Tatoeba weekly): the lists "
                         "built from it differ from the pinned ones, and the deviation is recorded")
    ap.add_argument("--skip-sentences", action="store_true",
                    help="build the ten lists that do not come from Tatoeba and leave out sentences_{es,en}.tsv (only "
                         "the prose and e-mail scenes read them; the pii release recipe has neither)")
    ap.add_argument("--allow-revision-drift", action="store_true")
    ap.add_argument("--per-source", type=int, help="TRAIN documents per source (default: the manifest's)")
    args = ap.parse_args()
    cli.start(args)
    man = manifest.load(MANIFEST)
    _, _, identity_report = cli.identity_preamble()
    if args.verify:
        return verify(args)
    if args.dry_run:
        for e in man.entries:
            if e["type"] == "hf":
                log(f"  {e['id']:34s} {e['dataset']} {'/'.join(e['splits'])} @ {e['pin']['revision'][:12]}")
            else:
                log(f"  {e['id']:34s} {e['url']}")
        cli.print_exam_status()
        return 0
    sets, exam_report = load_all(allow_missing=args.allow_missing_exams)
    exam_report["identity"] = identity_report
    cf = ContentFilter(sets)
    if args.what in ("lists", "all"):
        build_lists(args, man, cf, exam_report)
    if args.what in ("train", "all"):
        build_train(args, man, cf, exam_report)
    return 0


if __name__ == "__main__":
    sys.exit(cli.main_guard(main))
