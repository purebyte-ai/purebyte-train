# Publish a model

The public bundle is made here, with `pbtrain`. The weights are published in the runtime repository,
[purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte): as files of its GitHub releases (outside git), under
the [PureByte Model License](https://github.com/purebyte-ai/purebyte/blob/main/MODEL_LICENSE.md), and listed with
their sha256 in its catalog, [models/models.json](https://github.com/purebyte-ai/purebyte/blob/main/models/models.json).
This page is the checklist from a finished run to that catalog.

## 1. Before anything

- The run's `eval` stage passed the delivery gate (the code refuses a release otherwise), and, as a checklist the code
  does not enforce, its pre-registered criteria are PASS (none PENDING) with SIX seeds. A release that does not meet
  the checklist says so where users read it (the model card, and the recorded exceptions below).
- The exams ran on the model files that will ship (not on checkpoints), were recorded with `pbtrain eval TASK --record
  EXAM --metrics FILE` BEFORE the internal release (its MANIFEST copies the verdicts of that moment), and the numbers
  carry their conditions.
- The golden fingerprints pass on the recording platform ([AGENTS.md](../../AGENTS.md#build-and-test): CI does not
  compare them).
- `verify` found identical spans between Python and the runtime's engine (`PUREBYTE_MODEL_DUMP` set to the engine's
  `model_dump` tool, built as in [AGENTS.md](../../AGENTS.md#build-and-test), gives the fast, strict check; the pinned
  NumPy reference in [reference/](../../reference/README.md) is the fallback).

## 2. The internal release and the public bundle

```bash
python -m pbtrain release specialists/secrets-code/recipes/secrets-code-1.0.0.yaml --public --semver 1.0.0
```

One command, run once the exams are recorded. It runs `report` (every criterion, exams included), then the `release`
stage, then makes the public bundle. Without `--public --semver` it stops after the internal release; the same
command with them later makes the bundle from it. When the internal release exists already (`pbtrain run` runs every
stage, `release` included), `--public` checks that it was made from this task (its digest) and that every file still
has its MANIFEST sha256, and makes only the bundle. `--public`, `--semver` and the version are checked before
anything runs.

The `release` stage refuses unless the whole done.json chain holds (every stage ran, in order, on this task and on the
results on disk), the delivery gate passed, the real-window streams and the registered exams are the ones the `data`
stage checked, every exam the specialist lists was checked, and the run trained at least as many seeds as the
specialist's `release.ensemble`. It writes `$PUREBYTE_WORK/specialists/<name>/<version>/`: the model and the other
ensemble members as GGUF, the report, the task file, every stage's done.json, and a MANIFEST with:

- the task (file sha256 and digest), the code commit at release (marked `+dirty` when `training/` or `specialists/`
  had local changes), and the sha256 of every file and of every done.json;
- the members, the vote, the seeds, the released seed and how it was chosen (`released_seed_rule`, the seed `select`
  chose, `released_seed_reason`), the operating point (`operating_bias_vec` when there is one per type);
- the synthetic metrics, the gate, every criterion's verdict with the exams recorded so far, when the criteria were
  frozen, the exam records, the experimental options and every warning of the run;
- `training`: where and with what the seeds were trained (device, backend, GPU, PyTorch, NumPy and Python versions,
  torch threads, CPU kernels, the trainer's digest, the code commit at training time); `release_machine`: the machine
  that ran the release.

It is built in a temporary folder and moved into place in one step (a failure leaves nothing), is never overwritten,
and its files are read-only.

## 3. The public bundle

Writes `$PUREBYTE_WORK/releases/<name>-<semver>/`:

- `purebyte-<name>-<semver>.gguf` (and `-e1`, `-e2` for ensemble members), each with a `.sha256`;
- `MODEL_CARD.md` from the specialist's `MODEL_CARD.template.md`;
- `models.json.fragment`: the catalog entry (name, version, files, sizes, sha256, URLs).

It refuses to write a bundle whose card or GGUF metadata carries an internal path (a drive letter, a UNC share, `~/`,
`/home/`, `/Users/`, `/root/`, `/tmp/` and the other usual POSIX roots), a user directory or an internal run name
(`release/package.py`, `hygiene`, plus the private patterns of `$PUREBYTE_WORK/releases/forbidden.txt`), whose card
names another version of the specialist than `--semver`, or whose files declare another `general.version` than
`--semver` (the exporter writes the recipe's version there, and the license keys of `specialist.yaml`:
`export.license`, `export.license_name`). One line per bundle goes to `$PUREBYTE_WORK/releases/MAPPING.jsonl`
(private, never in git): which run and seeds each public file came from.

## Recorded exceptions of the released specialists

The released recipes do not all meet the checklist of section 1. The code records each exception in the release's
MANIFEST (`warnings`, `released_seed_rule`, `criteria_verdicts`), and the recipes say it in their comments:

- **secrets-code 1.0.0** trains three seeds with `decide: true` (not six), releases seed 1 with `released_seed: 1`,
  chosen on the exam code-fp-b, with the CredData TEST scores of the three seeds in view, and not on `select` (where
  seed 2 was best; the recipe gives no `released_seed_reason`, so the run records a warning), and was released with
  its two `code-fp-b` false-alarm criteria FAILED, after a sample of such findings was read (the recipe's comment).
  Its training data overlap exams: nine code-fp repositories (five through its CredData DEV stream, five through its
  code corpus, one through both), 847 windows of the stream that share 64-byte blocks with CredData TEST files, and
  four CredData TEST repositories in the corpus of that time (its published CredData figures leave those four out).
  Its weights were trained on a GPU with that corpus; a retraining lands close, not bit for bit.
- **secrets-bin 1.0.0**: its criteria read the exams `bin-pyext` and `bin-probe`, which are not in its `exams:` list and
  have no fingerprints in `data/exams/`: their content is not compared with the training data, and the run records a
  warning saying so. Its probe criterion FAILED for two of the six seeds (the released seed passed it); the probe is
  synthetic and in distribution (its credentials come from the training generator). 10 exam files (8 of the 25 Linux
  files of bin-fp-b, 2 of bin-fp-c) overlap the earlier binary corpus it was trained with. Its real negative windows
  come from binaries that cannot be redistributed: a retraining with another pool lands close, not bit for bit.
- **pii 1.0.0** passes its criterion at the default operating point, where the reading was moved after the criterion
  was written; by the criterion as first written it fails. Its data mix shipped against the pre-registered tie rule
  (a product decision). About 3 % of the release pool's documents share a 64-byte block with the PII Masking Benchmark
  or pii-bench-test, so the `data` stage refuses the release pool as it is (`--exam-overlap drop` builds one it
  accepts). Its code windows are cut from the rebuilt code corpus, not the one of the released weights; a retraining
  differs in those windows.

The runtime's [benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md)
publishes every one of these verdicts and overlaps, with the figures they affect.

## 4. The model card

Every number with its conditions: dataset, split and its size, operating point, ensemble and vote, seeds, and CPU
and thread count for speeds. What was not measured says TBD. What the model does NOT detect is a section of its own.

## 5. Catalog and upload

Both happen in the runtime repository. Merge the fragment into its catalog,
[models/models.json](https://github.com/purebyte-ai/purebyte/blob/main/models/models.json), by a pull request to
[purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte). Then the GGUF and `.sha256` files are uploaded to its
GitHub release `<name>-<semver>`, tagged on the commit whose catalog lists them; the steps are in its
[docs/releasing.md](https://github.com/purebyte-ai/purebyte/blob/main/docs/releasing.md#model-weights).
`purebyte models pull <name>` must download and verify the new files before the release is announced.
