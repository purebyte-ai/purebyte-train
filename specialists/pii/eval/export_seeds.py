"""Export every seed of a finished pii run as its own GGUF file, each at its own operating point: the six files the
recipe's criteria are measured on (benchmarks/piimb/run.sh scores each one with the purebyte CLI).

    python specialists/pii/eval/export_seeds.py TASK --out DIR [--work W]

Writes DIR/seed_<s>.gguf for every seed of the run's `finetune` stage, with the same declaration as the released file
(the specialist's profile, window and n-gram precision; the recipe's labels), and DIR/seeds.json: the seed, its select
F1, its bias, the file's sha256 and which seed the factory released. Then, for the criteria (`run.sh` is in the runtime
repository, purebyte-ai/purebyte, cloned at RUNTIME; it writes piimb-results/ into the folder it runs in, and
benchmarks/piimb/score.py is also pinned in this repository; keep DIR and that folder outside the source tree):

    cd DIR/.. && for f in DIR/seed_*.gguf; do bash RUNTIME/benchmarks/piimb/run.sh PIIMB "$f"; done
    python benchmarks/piimb/score.py metrics --released piimb-results/seed_<released>/scores.json \\
        --seeds piimb-results/seed_*/scores.json --out piimb-metrics.json
    python -m pbtrain eval TASK --record piimb --metrics piimb-metrics.json

specialists/pii/README.md ("Evaluate") has the same commands with the paths written out.
"""
import argparse
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "training"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("task")
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default=None, help="factory work dir (default $PUREBYTE_WORK/runs)")
    a = ap.parse_args(argv)
    from pbtrain import paths, runtime
    runtime.setup(threads=2, lowprio=True, gpu=False)
    from pbtrain.factory import spec as S
    from pbtrain.factory.chain import read_done
    from pbtrain.factory.stages import export_checkpoint
    spec = S.load(a.task)
    ft = read_done(a.work or paths.runs_dir(), spec, "finetune")
    os.makedirs(a.out, exist_ok=True)
    rows = []
    for r in ft["runs"]:
        p = os.path.join(a.out, f"seed_{r['seed']}.gguf")
        info = export_checkpoint(r["ckpt"], p, spec)
        with open(p, "rb") as f:
            digest = hashlib.sha256(f.read()).hexdigest()
        rows.append(dict(seed=r["seed"], select_f1=r["select_f1"], bias=r["bias"], file=os.path.basename(p),
                         sha256=digest, bytes=info["bytes"], released=r["seed"] == ft["released_seed"]))
        print(f"seed {r['seed']}: {os.path.basename(p)} ({info['bytes']:,} bytes, bias {r['bias']}, select F1 "
              f"{r['select_f1']}){'  <- released' if rows[-1]['released'] else ''}", flush=True)
    with open(os.path.join(a.out, "seeds.json"), "w", encoding="utf-8") as f:
        json.dump(dict(task=os.path.basename(a.task), released_seed=ft["released_seed"], seeds=rows), f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
