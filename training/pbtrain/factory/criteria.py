"""PRE-REGISTERED criteria: what the run must show, written in the task BEFORE it trains, checked by machine.

    criteria:
      written: 2026-09-23              # when they were written (before any training of this version)
      prediction: "..."                # what you expect, in words (recorded; a wrong prediction is a result too)
      synthetic:                       # checked by the factory's `eval` stage on its own splits
        - {metric: test_f1_mean, op: ">=", value: 0.97}
        - {metric: control_f1, op: "<=", value: 0.10}
      exams:                           # checked against the results recorded with
        - {exam: creddata-test, metric: f1, op: ">=", value: 0.55}          # `pbtrain eval TASK --record EXAM --metrics FILE`
        - {exam: code-fp-b, metric: findings_max_seed, op: "<=", value: 129}

Rules enforced here and in the factory:
- the criteria's sha256 is frozen by the `data` stage; any later stage refuses to run if they changed (re-run from `data`
  or restore them): nothing is re-tuned after looking at results;
- every criterion gets PASS / FAIL / PENDING: the `eval` stage records them in its done.json with the exam results
  recorded so far, and the `report` stage recomputes them with every exam result recorded by then, into the report, its
  done.json and the release MANIFEST; a FAIL is a result, not a bug;
- an exam named by a criterion and absent from the task's `exams:` list is accepted with a warning that the `data` stage
  records (it names whether that exam's fingerprints were registered, that is, whether its content was checked);
- three seeds INFORM, six DECIDE a release; criteria on seed tails read the MAX, not only the mean.

Synthetic metrics available (from the finetune/eval stages): test_f1_mean, test_f1_min, test_f1_max, spread,
control_f1, recall_select_min, doc_acc_mean, gate_margin, n_seeds.
"""
import hashlib
import json
import math

OPS = {">=": lambda a, b: a >= b, "<=": lambda a, b: a <= b, ">": lambda a, b: a > b, "<": lambda a, b: a < b,
       "==": lambda a, b: a == b}
SYNTHETIC_METRICS = {"test_f1_mean", "test_f1_min", "test_f1_max", "spread", "control_f1", "recall_select_min",
                     "doc_acc_mean", "gate_margin", "n_seeds"}


class CriteriaError(ValueError):
    pass


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check_item(c, kind):
    if not isinstance(c, dict):
        raise CriteriaError(f"criterion {c!r}: must be a mapping {{metric, op, value}}")
    need = {"metric", "op", "value"} | ({"exam"} if kind == "exams" else set())
    if set(c) - need - {"why"} or need - set(c):
        raise CriteriaError(f"criterion {c}: needs exactly {sorted(need)} (+ optional `why`)")
    if c["op"] not in OPS:
        raise CriteriaError(f"criterion {c}: op must be one of {sorted(OPS)}")
    if not _number(c["value"]):
        raise CriteriaError(f"criterion {c}: `value` must be a number")
    if not isinstance(c["metric"], str) or not c["metric"]:
        raise CriteriaError(f"criterion {c}: `metric` must be a name")
    if kind == "exams" and (not isinstance(c["exam"], str) or not c["exam"]):
        raise CriteriaError(f"criterion {c}: `exam` must be a name")
    if "why" in c and not isinstance(c["why"], str):
        raise CriteriaError(f"criterion {c}: `why` must be text")
    if kind == "synthetic" and c["metric"] not in SYNTHETIC_METRICS:
        raise CriteriaError(f"criterion {c}: unknown synthetic metric; known: {sorted(SYNTHETIC_METRICS)}")


def validate(criteria):
    if not isinstance(criteria, dict):
        raise CriteriaError("`criteria:` must be a mapping")
    extra = set(criteria) - {"written", "prediction", "synthetic", "exams"}
    if extra:
        raise CriteriaError(f"unknown keys in criteria: {sorted(extra)}")
    if not criteria.get("written"):
        raise CriteriaError("`criteria.written` (the date they were written, before training) is required")
    if not (criteria.get("synthetic") or criteria.get("exams")):
        raise CriteriaError("`criteria:` declares nothing to check")
    for kind in ("synthetic", "exams"):
        items = criteria.get(kind) or []
        if not isinstance(items, list):
            raise CriteriaError(f"`criteria.{kind}` must be a list")
        for c in items:
            _check_item(c, kind)
    return criteria


def sha(criteria):
    return hashlib.sha256(json.dumps(criteria or {}, sort_keys=True, default=str).encode()).hexdigest()[:16]


def exams_named(criteria):
    """The exams the criteria read, in order of first appearance."""
    out = []
    for c in (criteria or {}).get("exams") or []:
        if c["exam"] not in out:
            out.append(c["exam"])
    return out


def _verdict(c, v):
    if v is None:
        return "PENDING"
    if not _number(v):
        return "INVALID"                 # a recorded value that is not a number cannot pass a numeric criterion
    return "PASS" if OPS[c["op"]](v, c["value"]) else "FAIL"


def evaluate(criteria, synthetic=None, exams=None):
    """-> list of {kind, criterion, observed, verdict}. `exams` = {exam: {metric: value}} of recorded results."""
    out = []
    for c in (criteria or {}).get("synthetic") or []:
        v = (synthetic or {}).get(c["metric"])
        out.append(dict(kind="synthetic", criterion=c, observed=v, verdict=_verdict(c, v)))
    for c in (criteria or {}).get("exams") or []:
        v = ((exams or {}).get(c["exam"]) or {}).get(c["metric"])
        out.append(dict(kind="exams", criterion=c, observed=v, verdict=_verdict(c, v)))
    return out


def summary(verdicts):
    n = {k: sum(1 for v in verdicts if v["verdict"] == k) for k in ("PASS", "FAIL", "PENDING", "INVALID")}
    overall = "FAIL" if n["FAIL"] or n["INVALID"] else ("PENDING" if n["PENDING"] else "PASS")
    return dict(overall=overall, **n)
