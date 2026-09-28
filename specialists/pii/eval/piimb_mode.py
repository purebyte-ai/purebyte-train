"""The "PIIMB mode" operating point of a pii run, and the PII Masking Benchmark scored there: REPORTED beside the default
operating point, never used to choose a seed, a bias or a release.

    python specialists/pii/eval/piimb_mode.py TASK --piimb DIR [--seeds 0,1,2] [--device cpu|cuda|mps|auto] [--out FILE]
    python benchmarks/piimb/score.py metrics --seeds ... (or read the JSON directly)

Why a second point: the default operating point is chosen on `select` WINDOWS of 512 bytes for the best F1; the
benchmark ranks by F2 (recall weighs four times precision) and scans single sentences, where the window gate, trained on
512-byte windows, says "nothing here" more often. The rule, written before any model of the recipe was scored:

  per seed, the documents of the run's own `select` split (public TRAIN documents whose id hashes to select; never a
  test split, never the benchmark) are cut into sentences the way the benchmark cuts its documents (a sentence end or a
  line break), up to 4,000 of them in a seeded order; the (bias, window gate) pair with the highest label-agnostic F2
  over their bytes wins, the bias over the factory's grid and the gate on or off (ties: gate on, then the smaller
  |bias|). The benchmark's `sentences` are then scored at that point.

The model runs here in PyTorch on the checkpoints of the run, with the engine's windows (512 bytes, stride 384) and its
decoding (Viterbi with the operating bias; spans only where the window head says positive unless the gate is off):
`verify` proves that the exported file gives the same spans. With the gate on, `purebyte scan --bias B` on the file
gives the same point through the CLI (benchmarks/piimb/run.sh with BIAS=B); the CLI cannot switch the gate off.
The default point (the file's own bias, gate on) is scored beside it, for comparison with the CLI's figures.

Writes one JSON object: the rule, and per seed the chosen point, its select F2, the sweep and the benchmark's figures at
both points (per task and the mean of the six tasks). Nothing is printed but numbers.
"""
import argparse
import importlib.util
import json
import os
import re
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
SPECIALIST = os.path.dirname(HERE)
REPO = os.path.dirname(os.path.dirname(SPECIALIST))
for p in (os.path.join(REPO, "training"), SPECIALIST):
    if p not in sys.path:
        sys.path.insert(0, p)

CUT = re.compile(rb"(?<=[.!?])[ \t]+|[ \t]*\n[ \t\r]*")
IGNORE_T = 254                                    # the generator's span «type» that paints IGNORE


def _scorer():
    spec = importlib.util.spec_from_file_location("piimb_score", os.path.join(REPO, "benchmarks", "piimb", "score.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def windows_of(n, size=512, stride=384):
    """spec/FORMAT.md section 12.1 with `window.min` 1: starts 0, stride, ... until a window reaches the end."""
    out, start = [], 0
    while True:
        out.append(start)
        if start + size >= n:
            return out
        start += stride


def select_sentences(pub, part_of, max_n, seed=0):
    """[(bytes, gold [(a, b)], ignored [(a, b)])]: the `select` documents cut into sentences, in a seeded order."""
    import numpy as np
    docs = [d for d, i in zip(pub.docs, pub.ids) if part_of(i) == "select"]
    out = []
    for j in np.random.default_rng(seed).permutation(len(docs)):
        data, spans = docs[j]
        cuts, last = [], 0
        for m in CUT.finditer(data):
            cuts.append((last, m.start()))
            last = m.end()
        cuts.append((last, len(data)))
        for a0, b0 in cuts:
            if b0 - a0 < 2:
                continue
            g = [(max(a, a0) - a0, min(b, b0) - a0) for a, b, t in spans if t != IGNORE_T and a < b0 and b > a0]
            ig = [(max(a, a0) - a0, min(b, b0) - a0) for a, b, t in spans if t == IGNORE_T and a < b0 and b > a0]
            out.append((data[a0:b0], g, ig))
            if len(out) >= max_n:
                return out, len(docs)
    return out, len(docs)


def emissions(m, th, ch, datas, batch=64, bucket=64, device="cpu"):
    """-> [(input, window start, log-probs (T, L), P(window positive))]. A causal body cannot see padding placed after
    the valid bytes, so its windows are padded to a multiple of `bucket` and the window head pools over the valid
    positions only: the same outputs as exact-length batches, with far fewer calls. Other bodies: exact lengths."""
    import numpy as np
    import torch
    import torch.nn.functional as F
    causal = m.cfg.direction == "causal" and not m.cfg.lookahead
    groups = {}
    for i, d in enumerate(datas):
        for s in (windows_of(len(d)) if d else []):
            c = d[s:s + 512]
            groups.setdefault((-(-len(c) // bucket) * bucket) if causal else len(c), []).append((i, s, c))
    jobs = []
    with torch.no_grad():
        for L, js in sorted(groups.items()):
            for b in range(0, len(js), batch):
                part = js[b:b + batch]
                x = np.zeros((len(part), L), dtype=np.int64)
                mk = np.zeros((len(part), L), dtype=bool)
                for j, (_, _, c) in enumerate(part):
                    x[j, :len(c)] = np.frombuffer(c, np.uint8)
                    mk[j, :len(c)] = True
                h, _ = m(torch.from_numpy(x).to(device))
                lp = F.log_softmax(th(h).float(), -1).cpu().numpy()
                pd = F.softmax(ch(h, torch.from_numpy(mk).to(device)).float(), -1)[:, 1].cpu().numpy()
                for j, (i, s, c) in enumerate(part):
                    jobs.append((i, s, lp[j, :len(c)], float(pd[j])))
    return jobs


def decode(jobs, n_inputs, bias, gate=True):
    """-> per input [(byte a, byte b)]: Viterbi with the operating bias per window (the windows' spans, unmerged)."""
    import numpy as np
    from pbtrain.model.decode import label_bias, spans, viterbi_batch
    lb = label_bias(1, float(bias))
    out = [[] for _ in range(n_inputs)]
    by_len = {}
    for j, (_, _, lp, _) in enumerate(jobs):
        by_len.setdefault(lp.shape[0], []).append(j)
    for _, js in by_len.items():
        labs = viterbi_batch(np.stack([jobs[j][2] for j in js]), 1, lb)
        for r, j in enumerate(js):
            i, s, _, pd = jobs[j]
            if gate and pd <= 0.5:
                continue
            out[i] += [(s + a, s + b) for a, b, _ in spans(labs[r], 1)]
    return out


def f2_bytes(sents, preds):
    """Label-agnostic F2 over bytes (ignored bytes count neither way): (F2, precision, recall)."""
    import numpy as np
    p = t = i = 0
    for (data, g, ig), pr in zip(sents, preds):
        n = len(data)
        G, P, I_ = np.zeros(n, bool), np.zeros(n, bool), np.zeros(n, bool)
        for a, b in g:
            G[a:b] = True
        for a, b in ig:
            I_[a:b] = True
        for a, b in pr:
            P[a:b] = True
        P &= ~I_
        G &= ~I_
        p += int(P.sum())
        t += int(G.sum())
        i += int((P & G).sum())
    pr_, rc = (i / p if p else 0.0), (i / t if t else 0.0)
    return (5 * pr_ * rc / (4 * pr_ + rc) if pr_ + rc else 0.0), pr_, rc


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task")
    ap.add_argument("--piimb", required=True, help="the folder holding the benchmark's test_sentences.jsonl")
    ap.add_argument("--work", default=None, help="factory work dir (default $PUREBYTE_WORK/runs)")
    ap.add_argument("--seeds", default=None, help="comma-separated seeds (default: every seed of the run)")
    ap.add_argument("--max-sentences", type=int, default=4000)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps", "auto"), help="where the model runs "
                    "(cuda: NVIDIA or AMD ROCm; mps: Apple silicon); the spans are decoded on the CPU either way")
    ap.add_argument("--limit", type=int, default=0, help="benchmark sentences to score (0 = all; smoke runs only)")
    ap.add_argument("--out", default=None, help="the JSON to write (default: <run>/piimb_mode.json)")
    a = ap.parse_args(argv)
    from pbtrain import paths, runtime
    runtime.setup(threads=a.threads, lowprio=True, gpu=a.device != "cpu")
    import torch
    torch.set_num_threads(a.threads)
    from pbtrain.eval.operating_point import BIAS_GRID
    from pbtrain.factory import spec as S
    from pbtrain.factory.chain import read_done
    from pbtrain.train import checkpoint
    from pbtrain.verticals import load_specialist
    spec = S.load(a.task)
    load_specialist(spec["specialist"])
    plugin = sys.modules["pbtrain_specialist_" + spec["specialist"].replace("-", "_")]
    if len(spec["output_schema"]["labels"]) != 1:
        raise SystemExit("the PIIMB mode is defined for the label-agnostic policy (one type)")
    work = a.work or paths.runs_dir()
    ft = read_done(work, spec, "finetune")
    runs = [r for r in ft["runs"] if a.seeds is None or r["seed"] in {int(x) for x in a.seeds.split(",")}]
    pt = spec["data"]["public_train"]
    pub = plugin._load_public(paths.expand(pt["dir"]), pt.get("sources"), collapse=True)[0]
    t0 = time.time()
    sents, ndocs = select_sentences(pub, plugin._part_of, a.max_sentences)
    sc = _scorer()
    rows = sc.load(a.piimb, "sentences")
    if a.limit:
        rows = rows[:a.limit]
    bench = [r["text"].encode("utf-8") for r in rows]
    out = dict(rule="per seed: argmax byte F2 on the select documents cut into sentences, over the bias grid x window "
                    "gate on/off; ties -> gate on, then the smaller |bias|; the benchmark's `sentences` scored there",
               select_sentences=len(sents), select_documents=ndocs, benchmark_sentences=len(rows), seeds={})
    import contextlib
    from pbtrain.runtime import gpu_lock
    device = runtime.resolve_device(a.device)
    lock = gpu_lock(f"piimb_mode {spec['name']} {spec['version']}") if device != "cpu" else contextlib.nullcontext()
    with lock:                                          # one GPU job at a time
        for r in runs:
            ck = checkpoint.load(r["ckpt"], require_complete=True)
            m, th, ch = checkpoint.rebuild(ck, device)
            for x in (m, th, ch):
                x.eval()
            jobs = emissions(m, th, ch, [s[0] for s in sents], device=device)
            sweep = []
            for gate in (True, False):
                for b in BIAS_GRID:
                    f2, p, rc = f2_bytes(sents, decode(jobs, len(sents), b, gate))
                    sweep.append(dict(bias=b, gate=gate, f2=round(f2, 4), precision=round(p, 4),
                                      recall=round(rc, 4)))
            best = max(sweep, key=lambda x: (x["f2"], x["gate"], -abs(x["bias"])))
            jobs_b = emissions(m, th, ch, bench, device=device)
            res = {}
            points = (("piimb_mode", best["bias"], best["gate"]), ("default", float(ck["bias"]), True))
            for name, bias, gate in points:
                res[name] = sc.score(rows, dict(enumerate(decode(jobs_b, len(bench), bias, gate))))
            out["seeds"][str(r["seed"])] = dict(
                point=dict(bias=best["bias"], gate=best["gate"], select_f2=best["f2"]),
                default_point=dict(bias=float(ck["bias"]), gate=True),
                edge_of_grid=best["bias"] in (BIAS_GRID[0], BIAS_GRID[-1]), sweep=sweep,
                piimb_mode=res["piimb_mode"], default=res["default"])
            print(f"seed {r['seed']}: PIIMB mode bias {best['bias']} gate {'on' if best['gate'] else 'off'} (select F2 "
                  f"{best['f2']}); mean-of-6 F2 {res['piimb_mode']['_mean']['f2']:.4f} (default point "
                  f"{res['default']['_mean']['f2']:.4f}); {time.time() - t0:.0f}s", flush=True)
    dest = a.out or os.path.join(work, spec["name"], str(spec["version"]), "piimb_mode.json")
    os.makedirs(os.path.dirname(os.path.abspath(dest)), exist_ok=True)
    with open(dest, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    print(f"written {dest}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
