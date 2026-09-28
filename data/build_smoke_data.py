#!/usr/bin/env python3
"""Write the made-up stand-ins the two CPU smoke recipes read, so that they run on a fresh checkout (seconds, offline).

    python data/build_smoke_data.py              write what is missing, keep what is there
    python data/build_smoke_data.py --dry-run    the plan
    python data/build_smoke_data.py --verify     which stand-ins are still in place

    then: python -m pbtrain run specialists/secrets-code/recipes/smoke-cpu.yaml
          python -m pbtrain run specialists/pii/recipes/smoke-cpu.yaml --stages data,baselines,finetune,eval

Writes under $PUREBYTE_DATA (pbdata/smoke.py has the generators):
    corpora/smoke-context.bin (+ .manifest.json)   1.2 MB of generated pseudo-code; read by the smoke recipes only
    catalogs/gitleaks.json, trufflehog.json        two made-up credential formats           } only when the folder
    pii/lists/*.tsv                                the twelve lists, made-up words           } holds nothing yet (or
    pii/public_train/lookalike.jsonl               240 made-up labelled documents            } earlier stand-ins)
    smoke-standins.json                            every stand-in written, with its sha256

Nothing is downloaded, nothing real is used, and nothing is deleted. Real data are never replaced: a folder that already
holds catalogues, lists or documents is kept, and the smoke recipes then run on those. The stand-ins train nothing else:
the secrets-code and pii plugins refuse a stand-in among the files a build reads unless the recipe's corpus is
corpora/smoke-context.bin, and build_creddata_stream.py refuses stand-in catalogues. The real builds overwrite them
(build_secret_catalogs.py, fetch_pii.py lists); pii/public_train/lookalike.jsonl is the one stand-in no real build
overwrites: fetch_pii.py train says so, and it is removed by hand before a real recipe reads that folder.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pbdata import cli, paths, smoke  # noqa: E402
from pbdata.util import log  # noqa: E402


def main():
    ap = cli.parser(__doc__, "the smoke stand-ins")
    args = ap.parse_args()
    cli.start(args)
    data = paths.data_dir()
    S = smoke.standins_module()
    if args.verify:
        rec = os.path.join(data, S.MARKER)
        if not os.path.exists(rec):
            return cli.finish_verify([f"{paths.rel(rec)} is missing: nothing was written by this script"],
                                     "the smoke stand-ins")
        still = S.in_place(data)
        log(f"stand-ins still in place under $PUREBYTE_DATA: {len(still)}")
        for rel in still:
            log(f"  {rel}")
        problems = [] if S.SMOKE_CORPUS in still else [f"{S.SMOKE_CORPUS} is missing or changed (run this again)"]
        return cli.finish_verify(problems, "the smoke stand-ins")
    log("smoke stand-ins -> $PUREBYTE_DATA" + (" (dry run)" if args.dry_run else ""))
    r = smoke.build(data, log=log, dry_run=args.dry_run)
    for rel in r["written"]:
        log(f"  wrote {rel}")
    for k in r["kept"]:
        log(f"  kept {k}")
    if not args.dry_run:
        log(f"  recorded in {paths.rel(r['marker'])}")
        log("next: python -m pbtrain run specialists/secrets-code/recipes/smoke-cpu.yaml")
    return 0


if __name__ == "__main__":
    sys.exit(cli.main_guard(main))
