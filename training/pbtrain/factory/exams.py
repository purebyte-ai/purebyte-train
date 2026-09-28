"""Exam results, recorded with `pbtrain eval TASK --record EXAM --metrics FILE`.

Append-only: `<work>/<name>/<version>/exams/<exam>.jsonl` holds one line per record, never rewritten: when it was
recorded, the metrics (a JSON object {metric: value}), the sha256 of the metrics file, and which model files of the run
existed at that moment (the sha256 of the `export` stage's GGUF and of every `finetune` checkpoint). Recording an exam
again adds a line: the criteria read the LATEST record, and the report and the release MANIFEST show how many records
every exam has, so a result that was replaced stays visible.
"""
import json
import os
import re
import time

from .chain import StageError, read_done, sha_file

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


def exams_dir(work, spec):
    return os.path.join(work, spec["name"], str(spec["version"]), "exams")


def _models(work, spec):
    """The model files of the run right now (what a record can have measured)."""
    out = {}
    try:
        out["export_gguf_sha256"] = read_done(work, spec, "export").get("sha256")
    except StageError:
        out["export_gguf_sha256"] = None
    try:
        out["checkpoints_sha256"] = {str(r["seed"]): r.get("ckpt_sha256")
                                     for r in read_done(work, spec, "finetune")["runs"]}
    except StageError:
        out["checkpoints_sha256"] = None
    return out


def record(work, spec, exam, metrics_path):
    """Appends one record. Returns (the record, how many records the exam had before)."""
    if not NAME.fullmatch(str(exam)):
        raise StageError(f"exam name {exam!r}: letters, digits, `.`, `_` and `-` only (it names a file)")
    with open(metrics_path, encoding="utf-8") as f:
        try:
            metrics = json.load(f)
        except ValueError as e:
            raise StageError(f"{metrics_path}: not JSON ({e})")
    if not isinstance(metrics, dict) or not metrics:
        raise StageError(f"{metrics_path}: must be a JSON object {{metric: value}}")
    wanted = {c["metric"] for c in (spec.get("criteria") or {}).get("exams") or [] if c["exam"] == exam}
    bad = sorted(k for k in wanted & set(metrics)
                 if isinstance(metrics[k], bool) or not isinstance(metrics[k], (int, float)))
    if bad:
        raise StageError(f"{metrics_path}: the criteria compare {bad} with numbers, and the file gives "
                         f"{ {k: metrics[k] for k in bad} }")
    before = len(history(work, spec).get(exam, []))
    rec = dict(recorded=time.strftime("%Y-%m-%dT%H:%M:%S"), metrics=metrics,
               metrics_file=os.path.basename(metrics_path), metrics_sha256=sha_file(metrics_path),
               task_digest=spec["_digest"], models=_models(work, spec))
    d = exams_dir(work, spec)
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, exam + ".jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
    return rec, before


def history(work, spec):
    """{exam: [records, oldest first]}. A `<exam>.json` of the earlier one-file format reads as one record."""
    d = exams_dir(work, spec)
    out = {}
    if not os.path.isdir(d):
        return out
    for fn in sorted(os.listdir(d)):
        p = os.path.join(d, fn)
        if fn.endswith(".jsonl"):
            with open(p, encoding="utf-8") as f:
                out.setdefault(fn[:-6], []).extend(json.loads(ln) for ln in f if ln.strip())
        elif fn.endswith(".json"):
            with open(p, encoding="utf-8") as f:
                out.setdefault(fn[:-5], []).insert(0, dict(recorded=None, metrics=json.load(f), legacy=True))
    return out


def results(work, spec):
    """{exam: metrics of its LATEST record}."""
    return {k: v[-1]["metrics"] for k, v in history(work, spec).items() if v}


def summary(work, spec):
    """{exam: {records, first, last}} for the report and the MANIFEST."""
    return {k: dict(records=len(v), first=v[0].get("recorded"), last=v[-1].get("recorded"))
            for k, v in history(work, spec).items() if v}
