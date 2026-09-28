"""The leakage checks of the `data` stage. MANDATORY: a run whose training data can see its evaluation is not a result.

1. Between splits (`check_splits`), for the three pairs train-select, train-test and select-test. What is ENFORCED
   depends on the vertical's `overlap_guard`:
     windows  whole-window hashes over ALL windows: the second split may hold at most 0.1 % of its windows identical to
              a window of the first (`tolerance`). A generator of synthetic records or of labeled public documents
              can repeat a short window by chance, or a public dataset can hold the same document under two ids; a
              generator with no variety fails.
     chunks   (verticals that cut windows from real bytes) no 120-byte chunk may be shared, zero tolerance. The chunks
              are taken at the fixed offsets 0, 120, 240... of each of the FIRST 300 windows of each split (a chunk made
              of one repeated byte value identifies nothing and is skipped); the check therefore sees two copies of the
              same real bytes only when their offsets inside the windows differ by a multiple of 120 and the copy
              covers a whole chunk. Splits are disjoint by construction (separate byte ranges of the corpus); this is a
              cheap guard against a vertical that breaks that rule, not a proof.
   MEASURED, not enforced (`fingerprint_overlap`): how many windows of each split share at least one 64-byte block with
   the other split, with the content-anchored fingerprints of point 2 (every window, any offset). Content that repeats
   legitimately across disjoint ranges -- license headers, boilerplate, the generator's own templates -- makes it
   non-zero, which is why it is recorded in done.json and shown in the report instead of failing the stage.
2. Against the EXAMS (`check_against_exams`): every registered exam (a fingerprint file in
   $PUREBYTE_DATA/exams/fingerprints/, built by `data/fingerprint_exams.py` from the exam's own manifest) is compared,
   block by block, with every window of train and select, and with every real-window set the recipe feeds (real
   streams, guard windows). One shared block fails the stage. The report names the exam and counts the windows. Sets
   whose fingerprint meta says `role: dev` are DEVELOPMENT sets (training material may come from them, e.g. a labeled
   stream from CredData DEV): their hits are counted and reported, not enforced; the part of a DEV set used to choose
   seeds or biases must be fingerprinted as its own `test` set. The test split is not compared with the exams (it
   trains nothing).
3. The corpus manifest (written by the data build scripts) lists which exams were excluded at build time
   (`exclusion.content.sets`); `corpus_exclusion_gaps` names every exam registered now that the build did not exclude,
   or that changed since. The content check of point 2 still runs on every window, so this is a recorded warning.

Missing fingerprints are an error for the exams a task requires (it cannot prove what it claims), and a recorded
warning otherwise.
"""
import glob
import hashlib
import json
import os

import numpy as np

from . import fingerprint as fpmod

CHUNK, CHUNK_WINDOWS, WINDOW_TOLERANCE = 120, 300, 0.001
PAIRS = (("train", "select"), ("train", "test"), ("select", "test"))


class LeakError(ValueError):
    pass


def chunk_set(A, n=CHUNK, k=CHUNK_WINDOWS):
    """md5 of the n-byte chunks at offsets 0, n, 2n... of the first `k` windows, the last whole chunk included (chunks
    of ONE byte value identify no file: skipped)."""
    s = set()
    for r in A[:k]:
        b = bytes(r)
        for i in range(0, len(b) - n + 1, n):
            if len(set(b[i:i + n])) < 2:
                continue
            s.add(hashlib.md5(b[i:i + n], usedforsecurity=False).hexdigest())
    return s


def split_overlap(train_X, test_X, mode="windows", tolerance=WINDOW_TOLERANCE, names=("train", "test")):
    """Returns the overlap count between two splits; raises LeakError when it exceeds what the mode allows."""
    a, b = names
    if mode == "chunks":
        shared = len(chunk_set(train_X) & chunk_set(test_X))
        if shared:
            raise LeakError(f"{shared} chunks of {CHUNK} B shared between {a} and {b}: leak")
        return shared
    if mode != "windows":
        raise LeakError(f"unknown overlap_guard {mode!r}: windows | chunks")
    h = lambda A: {hashlib.md5(np.ascontiguousarray(r).tobytes(), usedforsecurity=False).hexdigest() for r in A}
    shared = len(h(train_X) & h(test_X))
    if shared > tolerance * len(test_X):
        raise LeakError(f"{shared} identical windows between {a} and {b} (more than {tolerance:.1%} of {b}): the "
                        f"generator has no variety, or the splits share their sources")
    return shared


def fingerprint_overlap(splits):
    """{`b`_in_`a`: windows of split b sharing a 64-byte block with split a} for the three pairs (measured)."""
    out = {}
    fps = {}
    for a, b in PAIRS:
        if a not in fps:
            fps[a] = fpmod.window_fingerprints(splits[a])
        n = int(fpmod.window_hits(splits[b], fps[a]).sum())
        out[f"{b}_in_{a}"] = dict(windows=n, of=int(len(splits[b])), share=round(n / max(1, len(splits[b])), 6))
    return out


def check_splits(t, mode):
    """`t` Partitions -> {enforced: {pair: count}, mode, tolerance, measured: fingerprint_overlap}; raises LeakError."""
    X = {"train": t.train[0], "select": t.select[0], "test": t.test[0]}
    enforced = {f"{a}-{b}": split_overlap(X[a], X[b], mode, names=(a, b)) for a, b in PAIRS}
    rule = (f"whole-window hashes of every window, at most {WINDOW_TOLERANCE:.1%} of the second split identical"
            if mode == "windows" else
            f"{CHUNK}-byte chunks at fixed offsets of the first {CHUNK_WINDOWS} windows of each split, zero shared")
    return dict(mode=mode, rule=rule, enforced=enforced, measured=fingerprint_overlap(X),
                measured_rule="windows sharing a 64-byte block (content-anchored fingerprints, every window); "
                              "recorded, not enforced")


def registered_exams(exams_dir):
    """{name: path} of every fingerprint file under exams/fingerprints/."""
    return {os.path.splitext(os.path.basename(p))[0]: p
            for p in sorted(glob.glob(os.path.join(exams_dir, "fingerprints", "*.npz")))}


def fp_digest(fp):
    """sha256 of a fingerprint array's content (the data scripts' `exclusion.fp_digest`: independent of how the .npz
    was compressed)."""
    return hashlib.sha256(np.ascontiguousarray(np.asarray(fp, dtype="<u8")).tobytes()).hexdigest()


def registry(exams_dir):
    """{name: {role, fp_sha256}} of the exams registered now."""
    out = {}
    for name, path in registered_exams(exams_dir).items():
        fp, meta = fpmod.load(path)
        out[name] = dict(role=meta.get("role", "test"), fp_sha256=fp_digest(fp))
    return out


def check_against_exams(sets, exams_dir, required=()):
    """`sets` {label: X uint8 [n, T]} -> report {exams: {exam: {label: windows_hit, role, fp_sha256}}, n_exams,
    dev_hits, warning}; raises LeakError on any hit of an exam whose role is not `dev`.

    `required`: exam names whose fingerprint file must exist; a missing one raises.
    """
    found = registered_exams(exams_dir)
    missing = sorted(set(required) - set(found))
    if missing:
        raise LeakError(f"the task requires the exam(s) {missing} and their fingerprints are not in "
                        f"{os.path.join(exams_dir, 'fingerprints')}: build them with data/fingerprint_exams.py")
    report, hits_total, dev_hits = {}, [], {}
    for name, path in found.items():
        fp, meta = fpmod.load(path)
        role = meta.get("role", "test")
        report[name] = dict(blocks=int(len(fp)), manifest=meta.get("manifest_sha256"), role=role,
                            fp_sha256=fp_digest(fp))
        for label, X in sets.items():
            if X is None or len(X) == 0:
                continue
            n = int(fpmod.window_hits(X, fp).sum())
            report[name][label] = n
            if n and role != "dev":
                hits_total.append(f"{name}: {n} window(s) of {label}")
            elif n:
                dev_hits.setdefault(name, {})[label] = n
    if hits_total:
        raise LeakError("training data shares content with an exam -> " + "; ".join(hits_total))
    return dict(exams=report, n_exams=len(found), dev_hits=dev_hits,
                warning=None if found else "no exam fingerprints registered: the exam-leak check checked nothing")


def corpus_manifest(corpus_path):
    """The build manifest written next to a corpus by the data scripts (`<corpus>.manifest.json`), or None."""
    for p in (corpus_path + ".manifest.json", os.path.splitext(corpus_path)[0] + ".manifest.json"):
        if os.path.exists(p):
            with open(p, encoding="utf-8") as f:
                return json.load(f)
    return None


def corpus_exclusion_gaps(manifest, exams):
    """Exams registered now that the corpus build did not exclude, or that changed since the build (`exams`: the
    report of `check_against_exams`). [] when the corpus has no manifest to compare with."""
    if not manifest:
        return []
    built = ((manifest.get("exclusion") or {}).get("content") or {}).get("sets") or {}
    gaps = []
    for name, rec in sorted((exams.get("exams") or {}).items()):
        if name not in built:
            gaps.append(f"{name}: not excluded when the corpus was built")
        elif built[name].get("fp_sha256") not in (None, rec.get("fp_sha256")):
            gaps.append(f"{name}: its fingerprints changed since the corpus was built")
    return gaps
