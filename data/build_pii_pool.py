#!/usr/bin/env python3
"""Build the release pool of the pii specialist (data/sources/pii-pool.yaml):

    $PUREBYTE_DATA/pii/public_train_xl/      <source>.jsonl (5 files), MANIFEST.json, LEAKAGE.json,
                                             HARD_CHECK_EXCLUDED.json (and PIIMB_OVERLAP.json with --piimb-overlap)
    $PUREBYTE_DATA/pii/public_train_clean/   base_public_train.jsonl alone, with its own MANIFEST.json

from the TRAIN files of four public PII datasets pinned by revision and sha256, with the hard overlap check against
every PII evaluation set (four levels: the same text, a shared sentence or line, the same masked template, a shared
masked sentence; pbdata/pii_pool.py). The base part is the sample of data/sources/pii.yaml, drawn from the same pinned
files instead of the network and checked against its reference digest.

    python data/build_pii_pool.py                       build (downloads what is missing: about 3.9 GB)
    python data/build_pii_pool.py --raw-dir DIR         take the four files from a local copy (matched by name, sha256)
    python data/build_pii_pool.py --exam-overlap drop   also leave out every document that shares a 64-byte block with
                                                        a `test` exam (see "The PII pool" in data/README.md)
    python data/build_pii_pool.py --extra-test FILE     more evaluation documents for the check (JSONL, `text`)
    python data/build_pii_pool.py --piimb-overlap       also write which benchmark sentences share a block with the pool
    python data/build_pii_pool.py --dry-run | --verify

Before: python data/fetch_exams.py --set piimb --set pii-bench-test --set pii-bench-dev, then
python data/fingerprint_exams.py. Reading the three Parquet files needs pyarrow (`pip install pyarrow`), the one extra
package of the data scripts. About 4 minutes on one core and 3 GB of memory once the files are there.
"""
import datetime
import json
import os
import shutil
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pbdata import cli, fetch, manifest, paths, pii, pii_pool  # noqa: E402
from pbdata.util import DataError, ensure_dir, log, read_json, sha256_file, warn, write_json  # noqa: E402

MANIFEST = os.path.join(paths.SOURCE_MANIFESTS, "pii-pool.yaml")
BASE_MANIFEST = os.path.join(paths.SOURCE_MANIFESTS, "pii.yaml")
EXAM_SETS = ("piimb", "pii-bench-test", "pii-bench-dev")


def out_dirs():
    return os.path.join(paths.pii_dir(), "public_train_xl"), os.path.join(paths.pii_dir(), "public_train_clean")


def raw_files(man, raw_dir=None, no_fetch=False):
    """{dataset: verified local path} of the four pinned TRAIN files."""
    out = {}
    for e in man.entries:
        f = e["files"][0]
        path = None
        if raw_dir:
            p = os.path.join(raw_dir, f["name"])
            if os.path.exists(p) and os.path.getsize(p) == f["size"] and sha256_file(p) == f["sha256"]:
                path = p
            elif os.path.exists(p):
                raise DataError(f"{p} does not match its pin (sha256 {f['sha256'][:16]}..., {f['size']} bytes)")
        if path is None:
            dest = fetch.blob_path(e["pin"]["sha256"])
            if no_fetch and not os.path.exists(dest):
                raise DataError(f"{e['id']}: not downloaded and --no-fetch was given ({e['url']})")
            path = fetch.fetch_file(e)
        out[_dataset(e)] = path
        log(f"  {e['id']}: {paths.rel(path) if path.startswith(paths.data_dir()) else path}")
    return out


def _dataset(e):
    """The dataset of an entry: its `hf:<dataset>:train` identity."""
    for s in e.get("identities") or []:
        if s.startswith("hf:"):
            return s[3:].rsplit(":", 1)[0]
    raise DataError(f"{e['id']}: no `hf:<dataset>:<split>` identity")


def test_index(extra=None):
    idx = pii_pool.TestIndex()
    pii_pool.index_piimb(idx, paths.exam_dir("piimb"))
    pm = paths.pii_public_map()
    for name in ("pii-bench-test", "pii-bench-dev"):
        pii_pool.index_bench(idx, paths.exam_dir(name), name, pm)
    for p in extra or []:
        pii_pool.index_extra(idx, p, "extra:" + os.path.basename(p))
    return idx


def base_sample(raw, per_source):
    """The sample of data/sources/pii.yaml from the local pinned files, each source checked against its reference."""
    bman = manifest.load(BASE_MANIFEST)
    local = {"ai4privacy/pii-masking-openpii-1m": pii.LocalRanges(raw["ai4privacy/pii-masking-openpii-1m"])}
    cols = {"nvidia/Nemotron-PII": ["text", "spans"], "gretelai/gretel-pii-masking-en-v1": ["text", "entities"],
            "gretelai/synthetic_pii_finance_multilingual": ["language", "generated_text", "pii_spans"]}
    for ds, c in cols.items():
        local[ds] = _rows(raw[ds], c)
    quiet, pii.log = pii.log, (lambda *a: None)
    out, record = {}, {}
    try:
        for e in bman.entries:
            if e["type"] != "hf":
                continue
            s = e.get("sampling") or {}
            docs = pii.sample_train(e, per_source, local=local)
            digest = pii_pool_sha(docs)
            ref = s.get("reference_sha256")
            if ref and digest != ref:
                raise DataError(f"{e['id']}: the base sample drawn from the pinned file has sha256 {digest[:16]}..., its "
                                f"reference is {ref[:16]}...: the pool would not be the release pool")
            out[s["output"]] = docs
            record[e["id"]] = dict(output=s["output"], documents=len(docs), sha256=digest, matches_reference=digest == ref)
    finally:
        pii.log = quiet
    docs = []
    for name in sorted(out):                                   # the order of the base directory's file names
        docs += out[name]
    return docs, record


def _rows(path, columns):
    cache = {}

    def rows(dataset, split, offset, length):
        if "t" not in cache:
            cache["t"] = pii_pool.parquet_rows(path, columns)
        t = cache["t"]
        return t[offset:offset + length], len(t)
    return rows


def pii_pool_sha(docs):
    import hashlib
    return hashlib.sha256(pii.jsonl_bytes(docs)).hexdigest()


def block_filter(sets):
    """text -> the first `test` exam that shares a 64-byte block with it, or None."""
    from pbdata.exclusion import ContentFilter
    tests = [s for s in sets if s.role != "dev"]
    cf = ContentFilter(tests)
    return lambda text: next(iter(cf.shared(text.encode("utf-8"))), None)


def build(args, man, sets, exam_report):
    t0 = time.time()
    b = man.doc.get("build") or {}
    xl, clean = out_dirs()
    for d in (xl, clean):
        ensure_dir(d)
        stale = os.path.join(d, "EXCLUDED.json")
        if os.path.exists(stale):                           # a pool never carries one: PublicDocs would drop ids
            os.remove(stale)
    log("pinned TRAIN files:")
    raw = raw_files(man, args.raw_dir, args.no_fetch)
    log("evaluation documents (the hard check):")
    idx = test_index(args.extra_test)
    log(f"  {idx.summary()}")
    drop = args.exam_overlap == "drop"
    filt = block_filter(sets) if drop else None
    listed = [x["key"] for x in b.get("listed_exclusions") or []]
    w = pii_pool.PoolWriter(xl, idx, filt, listed=listed)
    log("base sample (data/sources/pii.yaml) from the pinned files:")
    base_docs, base_record = base_sample(raw, int(b.get("base_per_source", 12000)))
    for k, v in base_record.items():
        log(f"  {k}: {v['documents']} documents, reference {'matched' if v['matches_reference'] else 'NOT matched'}")
    st = w.emit("base_public_train", base_docs, license="CC-BY-4.0 AND Apache-2.0", from_manifest="data/sources/pii.yaml")
    log(f"  base_public_train: {st['written']} written of {st['candidates']} (excluded {st['excluded']})")
    op = b.get("openpii") or {}
    quota = pii_pool.openpii_quota(int(op.get("en", 48000)), int(op.get("per_language", 7000)))
    log("OpenPII-1M TRAIN, 23 languages:")
    cand = pii_pool.openpii_candidates(raw["ai4privacy/pii-masking-openpii-1m"], quota, float(op.get("oversample", 2.5)))
    docs, pre = w.prefilter(pii_pool.openpii_order(cand), quota)
    log(f"  prefilter: {len(docs)} kept of {sum(len(v) for v in cand.values())} candidates; excluded {pre}")
    del cand
    streams = (("openpii1m_xl", docs, "CC-BY-4.0"),
               ("nemotron_xl", pii_pool.nemotron_docs(pii_pool.parquet_rows(raw["nvidia/Nemotron-PII"],
                                                                            ["uid", "text", "spans"])), "CC-BY-4.0"),
               ("gretel_en_v1_xl", pii_pool.gretel_en_docs(pii_pool.parquet_rows(
                   raw["gretelai/gretel-pii-masking-en-v1"], ["uid", "text", "entities"])), "Apache-2.0"),
               ("gretel_finance_xl", pii_pool.gretel_finance_docs(pii_pool.parquet_rows(
                   raw["gretelai/synthetic_pii_finance_multilingual"],
                   ["level_0", "index", "language", "generated_text", "pii_spans"])), "Apache-2.0"))
    for name, stream, lic in streams:
        st = w.emit(name, stream, license=lic)
        log(f"  {name}: {st['written']} written of {st['candidates']} (excluded {st['excluded']}, duplicates "
            f"{st['duplicates']})")
    w.stats["openpii1m_xl"]["prefilter"] = dict(candidates=len(docs) + sum(pre.values()), excluded=pre)
    refs = b.get("reference_sha256") or {}
    outputs = {n: dict(file=f"{n}.jsonl", documents=w.stats[n]["written"], sha256=w.stats[n]["sha256"],
                       reference_sha256=refs.get(f"{n}.jsonl"),
                       matches_reference=w.stats[n]["sha256"] == refs.get(f"{n}.jsonl")) for n in pii_pool.OUTPUTS}
    total = sum(v["documents"] for v in outputs.values())
    same = all(v["matches_reference"] for v in outputs.values())
    with open(os.path.join(xl, "HARD_CHECK_EXCLUDED.json"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(w.excluded, f)
    exam_counts = None
    if not drop:
        filt = block_filter(sets)
        exam_counts = {}
        for n in pii_pool.OUTPUTS:
            with open(os.path.join(xl, f"{n}.jsonl"), encoding="utf-8") as f:
                for ln in f:
                    hit = filt(json.loads(ln)["text"])
                    if hit:
                        exam_counts[hit] = exam_counts.get(hit, 0) + 1
    leakage = dict(rule=pii_pool.RULE, tests=idx.summary(), excluded=len(w.excluded),
                   per_source={n: dict(candidates=s["candidates"], excluded=s["excluded"], duplicates=s["duplicates"])
                               for n, s in w.stats.items()},
                   exam_blocks=dict(policy=args.exam_overlap,
                                    documents_sharing_a_block_with_a_test_exam=exam_counts,
                                    note=("left out of the pool" if drop else
                                          "kept: the pool is the release pool; the training factory's data stage "
                                          "refuses windows that share a block with a `test` exam")))
    write_json(os.path.join(xl, "LEAKAGE.json"), leakage)
    record = dict(builder="data/build_pii_pool.py", manifest=dict(path="data/sources/pii-pool.yaml", sha256=man.sha256),
                  built=datetime.datetime.now().isoformat(timespec="seconds"), seconds=round(time.time() - t0, 1),
                  inputs={e["id"]: dict(url=e["url"], sha256=e["pin"]["sha256"], licence=e["licence"])
                          for e in man.entries},
                  base_sample=base_record, sources=w.stats, outputs=outputs, documents=total,
                  release_pool=same, exam_overlap=args.exam_overlap, listed_exclusions=len(listed),
                  extra_tests=[os.path.basename(p) for p in args.extra_test or []], exclusion=exam_report)
    write_json(os.path.join(xl, "MANIFEST.json"), record)
    shutil.copyfile(os.path.join(xl, "base_public_train.jsonl"), os.path.join(clean, "base_public_train.jsonl"))
    base_ids = {d["id"] for d in base_docs}
    write_json(os.path.join(clean, "MANIFEST.json"), dict(
        builder="data/build_pii_pool.py", what="the base sample minus every document that fails the hard overlap check",
        built=record["built"], documents=outputs["base_public_train"]["documents"],
        sha256=outputs["base_public_train"]["sha256"],
        excluded=[e for e in w.excluded if e["id"] in base_ids]))
    log(f"pool -> {paths.rel(xl)}: {total} documents, {len(w.excluded)} excluded, {record['seconds']} s; "
        + ("IDENTICAL to the release pool" if same else "NOT the release pool (see MANIFEST.json, outputs)"))
    if args.piimb_overlap:
        piimb_overlap(xl)
    return 0


def piimb_overlap(xl):
    """PIIMB_OVERLAP.json: the benchmark sentences (uids) that share a 64-byte block with the pool, per task. Local
    only: derived from the benchmark, it is never published (its licence)."""
    texts = []
    for n in pii_pool.OUTPUTS:
        with open(os.path.join(xl, f"{n}.jsonl"), encoding="utf-8") as f:
            texts += [json.loads(ln)["text"] for ln in f]
    fp = pii_pool.fingerprint_texts(texts)
    del texts
    rows = []
    with open(os.path.join(paths.exam_dir("piimb"), "test_sentences.jsonl"), encoding="utf-8") as f:
        for ln in f:
            r = json.loads(ln)
            rows.append((r["uid"], r["task_name"], r["text"]))
    hit = pii_pool.texts_sharing_blocks([r[2] for r in rows], fp)
    per = {}
    for (uid, task, _), h in zip(rows, hit):
        c = per.setdefault(task, [0, 0])
        c[0] += int(h)
        c[1] += 1
    write_json(os.path.join(xl, "PIIMB_OVERLAP.json"), dict(
        what="PIIMB `sentences` that share a 64-byte block (the training package's fingerprints) with this pool",
        per_task={t: dict(sharing=c[0], sentences=c[1]) for t, c in sorted(per.items())},
        uids=[r[0] for r, h in zip(rows, hit) if h]))
    for t, c in sorted(per.items()):
        log(f"  PIIMB {t}: {c[0]} of {c[1]} sentences share a block with the pool")


def verify(args, man):
    xl, clean = out_dirs()
    p = os.path.join(xl, "MANIFEST.json")
    if not os.path.exists(p):
        return cli.finish_verify([f"{paths.rel(xl)}: not built"], "the PII pool")
    rec = read_json(p)
    problems = []
    for n, o in rec["outputs"].items():
        q = os.path.join(xl, o["file"])
        if not os.path.exists(q) or sha256_file(q) != o["sha256"]:
            problems.append(f"{o['file']}: missing or changed since the build")
        elif not o.get("matches_reference"):
            warn(f"{o['file']}: not the release pool's file (built with {rec.get('exam_overlap')!r}, extra tests "
                 f"{rec.get('extra_tests')})")
    q = os.path.join(clean, "base_public_train.jsonl")
    if not os.path.exists(q) or sha256_file(q) != rec["outputs"]["base_public_train"]["sha256"]:
        problems.append("public_train_clean/base_public_train.jsonl: missing or changed")
    if os.path.exists(os.path.join(xl, "EXCLUDED.json")):
        problems.append("public_train_xl/EXCLUDED.json exists: the generator would drop every id it lists")
    return cli.finish_verify(problems, "the PII pool")


def main():
    ap = cli.parser(__doc__, "the PII pool", excludes=True, fetches=True)
    ap.add_argument("--raw-dir", help="local copies of the four TRAIN files (matched by name and sha256)")
    ap.add_argument("--exam-overlap", choices=("keep", "drop"), default="keep",
                    help="keep (default: the release pool, reported) or drop the documents sharing a 64-byte block "
                         "with a `test` exam")
    ap.add_argument("--extra-test", action="append", help="more evaluation documents (JSONL with `text`); repeatable")
    ap.add_argument("--piimb-overlap", action="store_true",
                    help="write PIIMB_OVERLAP.json: the benchmark sentences that share a block with the pool")
    args = ap.parse_args()
    cli.start(args)
    man = manifest.load(MANIFEST)
    _, _, identity_report = cli.identity_preamble()
    if args.verify:
        return verify(args, man)
    if args.dry_run:
        for e in man.entries:
            f = e["files"][0]
            log(f"  {e['id']:34s} {f['name']} ({f['size']:,} bytes) {e['url']}")
            fetch.fetch_file(e, dry_run=True)
        for s in EXAM_SETS:
            d = paths.exam_dir(s)
            log(f"  evaluation set {s:16s} {'on disk' if os.path.isdir(d) else 'MISSING (fetch_exams.py)'}")
        cli.print_exam_status()
        return 0
    from pbdata.exclusion import load_all
    sets, exam_report = load_all(allow_missing=args.allow_missing_exams)
    exam_report["identity"] = identity_report
    return build(args, man, sets, exam_report)


if __name__ == "__main__":
    sys.exit(cli.main_guard(main))
