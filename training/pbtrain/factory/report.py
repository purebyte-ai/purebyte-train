"""The run report: generated ONLY from the done.json files of the stages and the recorded exam results (never from the
console or from memory), with every number next to its control, its rival and its conditions, and every warning the
stages recorded."""
import os

from . import criteria as crit
from . import exams as exam_records
from .chain import read_done, stage_dir


def _fmt_criteria(verdicts):
    lines = []
    for v in verdicts:
        c = v["criterion"]
        what = f"{c.get('exam', 'synthetic')}: {c['metric']} {c['op']} {c['value']}"
        obs = "-" if v["observed"] is None else (f"{v['observed']:.4f}" if isinstance(v["observed"], float)
                                                   else str(v["observed"]))
        lines.append(f"| {what} | {obs} | **{v['verdict']}** |")
    return lines


def _released_line(ft):
    line = f"- Seeds {[r['seed'] for r in ft['runs']]}; released seed {ft['released_seed']}"
    if not ft.get("forced_seed"):
        return line + " (the best on `select`)"
    line += " (FORCED by `train.released_seed`"
    if ft.get("select_best_seed") is not None and ft["select_best_seed"] != ft["released_seed"]:
        line += f"; the best on `select` was seed {ft['select_best_seed']}"
    reason = ft.get("released_seed_reason")
    return line + (f"; reason given: {reason})" if reason else "; the task gives no reason)")


def _training_line(ft):
    t = ft.get("training") or {}
    if not t:
        return "- Trained on: not recorded"
    where = t.get("device", "?") + (f" ({t['gpu']})" if t.get("gpu") else "")
    return (f"- Trained on {where}, backend `{t.get('backend')}`, PyTorch {t.get('torch')}, NumPy {t.get('numpy')}, "
            f"Python {t.get('python')}, {t.get('torch_threads')} torch thread(s), CPU kernels {t.get('cpu_capability')}, "
            f"{t.get('os')}; trainer digest `{str(t.get('trainer_digest'))[:16]}`, code commit `{t.get('code_commit')}`")


def _split_lines(ov):
    if not isinstance(ov, dict) or "enforced" not in ov:
        return [f"- Between splits: {ov}"]
    enf = ", ".join(f"{k} {v}" for k, v in ov["enforced"].items())
    meas = ", ".join(f"{k.replace('_in_', ' in ')} {x['windows']} of {x['of']} ({x['share']:.2%})"
                     for k, x in ov["measured"].items())
    return [f"- Between splits, enforced (`{ov['mode']}`: {ov['rule']}): {enf}",
            f"- Between splits, measured ({ov['measured_rule']}): {meas}"]


def _exam_lines(dd):
    leak = dd["exam_leak_check"]
    n = leak.get("n_exams", 0)
    sets = sorted({k for rec in (leak.get("exams") or {}).values() for k in rec
                   if k not in ("blocks", "manifest", "role", "fp_sha256")})
    lines = [f"- Exam-leak check: {n} registered exam(s) compared with {', '.join(sets) or 'nothing'}; 0 windows share a "
             f"64-byte block with an exam of role `test` (one would have failed the stage)"
             + (f"; required by the task: {dd.get('required_exams')}" if dd.get("required_exams") else "")]
    dev = leak.get("dev_hits") or {}
    if dev:
        lines.append("- DEV exams (role `dev`: training material may come from them; counted, not enforced): "
                     + "; ".join(f"{name}: " + ", ".join(f"{k} {v} window(s)" for k, v in hits.items())
                                 for name, hits in sorted(dev.items())))
    return lines


def write_report(work, spec):
    """Writes report/report.md. Returns (its path, the verdicts of every criterion with the exams recorded now)."""
    h = {s: read_done(work, spec, s) for s in ("data", "baselines", "finetune", "eval", "calibrate", "export",
                                               "verify")}
    dd, bl, ft, ev, cal, ex, vf = (h[k] for k in ("data", "baselines", "finetune", "eval", "calibrate", "export",
                                                   "verify"))
    m = ev["metrics"]
    verdicts = crit.evaluate(spec.get("criteria"), synthetic=m, exams=exam_records.results(work, spec))
    records = exam_records.summary(work, spec)
    warnings = list(dd.get("warnings") or []) + list(ft.get("warnings") or []) + list(ev.get("notes") or [])
    labels = spec["output_schema"]["labels"]
    if cal.get("operating_bias_vec"):
        op = ("per type " + ", ".join(f"{a}: {b}" for a, b in zip(labels, cal["operating_bias_vec"]))
              + f" (around the scalar {cal['operating_bias']})")
    else:
        op = f"{cal['operating_bias']}"
    ece = cal.get("document_head_ece") or {}
    gate = ("PASSES" if ev["gate_passes"] else "DOES NOT PASS") + ": " + (
        "; ".join(r["reason"] for r in ev.get("refused") or []) or ev["gate_reason"])
    lines = [
        f"# Run report: {spec['name']} {spec['version']}",
        "",
        "Generated from the stages' done.json files and the recorded exam results. Synthetic numbers measure the model",
        "on the task's own generated test split; they do NOT predict real-world performance (see the exams section and",
        "training/docs/evaluate.md).",
        "",
        "## Identification",
        f"- Task `{os.path.basename(spec['_path'])}` sha256[:16] `{spec['_sha256']}`, digest[:16] "
        f"`{spec['_digest'][:16]}`; specialist `{spec['specialist']}`",
        f"- Model: {ex['mb']} MB GGUF, format version {ex['format_version']}, sha256 `{ex['sha256']}`",
        _released_line(ft),
        _training_line(ft),
        f"- Experimental options in use: {ft.get('experimental') or 'none'}",
        "",
        "## Data",
        f"- Vertical `{dd['vertical']}`; {dd['n_train']} train windows, {dd['n_eval']} test windows of {dd['seq_len']} B",
        f"- Positives {dd['positives']:.3f}; spans per train window {dd['spans_per_window']}",
        *_split_lines(dd["split_overlap"]),
        *_exam_lines(dd),
        "",
        "## Warnings",
        *([f"- {w}" for w in warnings] or ["- none"]),
        "",
        "## Result on the synthetic test (with its control and its rival)",
        "| measure | value |", "|---|---|",
        f"| span-F1, mean of {m['n_seeds']} seeds | **{m['test_f1_mean']:.4f}** (min {m['test_f1_min']:.4f}, max "
        f"{m['test_f1_max']:.4f}) |",
        f"| spread between seeds | {m['spread']:.4f} |",
        f"| control with shuffled spans (must be ~0) | {m['control_f1']:.4f} |",
        f"| classic rival ({bl['rival']}) | {bl['rival_f1']:.4f} |",
        f"| delivery gate | {gate} |",
        f"| scope | {ev['scope']} |",
        "",
        "## Pre-registered criteria",
        f"Frozen on {dd.get('criteria_frozen_at')} by the first `data` stage of this version (sha256[:16] "
        f"`{dd.get('criteria_sha256')}`); summary: {crit.summary(verdicts)} (exams: the latest recorded result of each)",
        "",
        "| criterion | observed | verdict |", "|---|---|---|",
        *(_fmt_criteria(verdicts) or ["| (none declared) | | |"]),
        "",
        "Exam results recorded: " + ("; ".join(f"{k}: {v['records']} record(s), last {v['last']}"
                                              for k, v in sorted(records.items())) or "none"),
        "",
        "## Operating point",
        f"- Viterbi bias {op}, chosen on `select` (policy `{spec['output_schema'].get('bias_policy', 'conservative')}`); "
        f"it travels in the GGUF.",
        (f"- Document head: expected calibration error {ece['ece']} on {ece['windows']} `select` windows, estimator "
         f"floor p95 {ece['floor_p95']} ({'above the floor' if ece['resolved'] else 'below the floor: not a measurement'}"
         f"); it gates by its argmax, and the file carries no temperature." if ece else "- Document head: not measured"),
        "",
        "## Runtime equivalence",
        f"- {vf['equal']}/{vf['n']} test windows ({vf['windows_with_spans']} with spans in Python) give IDENTICAL spans "
        f"in Python and in the {vf['runtime']} (gating included; exact comparison of positions)"
        + (f", {vf['engine_threads']} engine thread(s)" if vf.get("engine_threads") else "") + ".",
        "",
        "## Not measured here",
        "- Real-data performance: run the specialist's exams, record each with",
        "  `pbtrain eval TASK --record EXAM --metrics FILE`, and read them against the pre-registered exam criteria",
        "  before any release (training/docs/evaluate.md).",
    ]
    p = os.path.join(stage_dir(work, spec, "report"), "report.md")
    with open(p, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    return p, verdicts
