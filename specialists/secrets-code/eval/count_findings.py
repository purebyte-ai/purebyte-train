"""False-alarm exams: count what the released models report on data with no credential (or with known ones).

    python specialists/secrets-code/eval/count_findings.py --models DIR --exam PATH --out metrics.json
        [--purebyte BIN] [--threads 1] [--votes 2]

DIR holds the ensemble as the release writes it (`model.gguf`, `member_1.gguf`, ...). Runs the engine's CLI once with
the whole ensemble and `--per-model`, and writes the metrics the recipes' criteria use:

    findings_ensemble        findings reported by the ensemble (its vote)
    findings_per_seed        findings of every member alone, in file order
    findings_mean_seed, findings_max_seed, findings_median_seed

Values are never printed (the CLI masks them); only counts leave this script. Record the result with
`python -m pbtrain eval TASK --record EXAM --metrics metrics.json`.
"""
import argparse
import glob
import json
import os
import shutil
import statistics
import subprocess
import sys


def _members(folder):
    first = os.path.join(folder, "model.gguf")
    rest = sorted(glob.glob(os.path.join(folder, "member_*.gguf")))
    if not os.path.exists(first):
        raise SystemExit(f"{folder} has no model.gguf")
    return [first] + rest


def _per_model_counts(report, n):
    counts = [0] * n
    for res in report.get("results", []):
        pm = res.get("per_model") or []
        for i, entry in enumerate(pm):
            if isinstance(entry, list):                     # one list of findings per member
                counts[i] += len(entry)
            elif isinstance(entry, dict) and "model" in entry:
                counts[int(entry["model"])] += 1
    return counts


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--exam", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--purebyte", default=shutil.which("purebyte") or "purebyte")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--votes", type=int, default=None)
    a = ap.parse_args(argv)
    files = _members(a.models)
    cmd = [a.purebyte, "scan", "--model", files[0]] + sum((["--ensemble", f] for f in files[1:]), [])
    cmd += ["--per-model", "--format", "json", "--threads", str(a.threads)]
    if a.votes:
        cmd += ["--votes", str(a.votes)]
    r = subprocess.run(cmd + [a.exam], capture_output=True, text=True)
    # Exit status 2 with a report means some inputs could not be analyzed (listed in `failures`); without one, the
    # command itself failed.
    if r.returncode not in (0, 1, 2) or not r.stdout.startswith("{"):
        raise SystemExit(f"purebyte scan failed ({r.returncode}): {r.stderr[-400:]}")
    report = json.loads(r.stdout)
    failures = report.get("failures", [])
    if failures:
        print(f"warning: {len(failures)} input(s) could not be analyzed", file=sys.stderr)
    per = _per_model_counts(report, len(files))
    m = dict(findings_ensemble=len(report.get("findings", [])), findings_per_seed=per,
             findings_mean_seed=statistics.mean(per), findings_max_seed=max(per),
             findings_median_seed=statistics.median(per), inputs=len(report.get("results", [])),
             failures=len(failures), models=len(files))
    with open(a.out, "w", encoding="utf-8") as f:
        json.dump(m, f, indent=2)
    print(json.dumps({k: v for k, v in m.items() if k != "findings_per_seed"}))


if __name__ == "__main__":
    main()
