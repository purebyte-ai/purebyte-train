# Training

PureByte is a farm of **AI Specialists**: tiny models (1 to 29 MiB) that do ONE thing very well, on a CPU, privately
and measurably. Everything needed to train a new AI Specialist, and to retrain the `secrets-code` and `pii` examples
from public data (close to the released weights, not bit for bit), is in this repository: the training stack
(`training/`, the `pbtrain` package), the specialists' recipes and data generators (`specialists/`), and the scripts
that download and build the data from pinned public sources (`data/`). The real-negative stream of `secrets-bin` and
two of its exams are not public yet (TBD: [specialists/secrets-bin](../specialists/secrets-bin/README.md)). The runtime that runs the models
(engine, CLI, model format, benchmarks, model cards and catalog) is a separate repository,
[purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte). Datasets and weights are never in git: data goes to
`$PUREBYTE_DATA` (default `~/.purebyte/data`), runs and releases to `$PUREBYTE_WORK` (default `~/.purebyte/work`).

## Vibe Training

**Vibe Training** is training an AI Specialist by describing what you want to an AI coding agent, the way vibe
coding builds software: the agent follows the golden path of [AGENTS.md](../AGENTS.md#the-golden-path-vibe-training) -- choose or create the
vertical, download or generate the data, fix the success criterion BEFORE measuring, train, evaluate honestly and
publish -- and the factory makes every step checkable (leak checks, pre-registered criteria, a control and a rival,
runtime equivalence). A request can be as short as "train an AI Specialist that finds IBANs in CSV exports, and
tell me honestly how good it is"; the commands below are what the agent runs.

## The golden path

```bash
python3 -m venv .venv && . .venv/bin/activate      # Windows: py -m venv .venv, then .venv\Scripts\Activate.ps1
python -m pip install -e "training[test]"           # Python 3.10+, PyTorch 2.4+ (README.md#quick-start)
python -m pbtrain doctor                            # what is installed and what is found

# the pipeline on a CPU in minutes, on made-up stand-ins (proves the machinery, not a model)
python data/build_smoke_data.py
python -m pbtrain run specialists/pii/recipes/smoke-cpu.yaml --stages data,baselines,finetune,eval,calibrate,export,verify,report
python -m pbtrain run specialists/secrets-code/recipes/smoke-cpu.yaml   # ends at the gate: GATE NOT PASSED, exit 2

# data (see data/README.md): the evaluation sets FIRST, they are what the builds exclude
python data/fetch_exams.py                          # public evaluation sets
python data/fetch_creddata.py                       # CredData (Linux, macOS or WSL)
python data/fingerprint_exams.py                    # exams/fingerprints/<set>.npz; sets without a public download are fingerprinted incomplete
python data/build_code_corpus.py                    # corpora/code-context.bin
python data/build_binary_corpus.py                  # corpora/binary-context.bin (secrets-bin)
python data/fetch_pii.py all                        # PII lists and public TRAIN documents (pii)
python data/build_pii_pool.py --exam-overlap drop   # pii/public_train_xl, without its documents that overlap an exam (needs pyarrow)
python data/build_secret_catalogs.py                # credential-format catalogs
python data/build_creddata_stream.py --exam-overlap drop   # streams/creddata-dev-spans.npz (secrets-code)

# the released recipe, every stage up to `verify` (three seeds and the control: 24.5 minutes on an RX 6800 XT with
# `--device cuda --backend radeon --parallel 4`; `--device auto` never turns the Radeon backend on)
python -m pbtrain run specialists/secrets-code/recipes/secrets-code-1.0.0.yaml --stages data,baselines,finetune,eval,calibrate,export,verify --device auto --parallel 3

# exams, pre-registered criteria, then the report, the immutable release and the public bundle, once
python -m pbtrain eval specialists/secrets-code/recipes/secrets-code-1.0.0.yaml --record creddata-test --metrics m.json
python -m pbtrain release specialists/secrets-code/recipes/secrets-code-1.0.0.yaml --public --semver 1.0.0
```

`m.json` holds the metrics the criteria read (`f1_ensemble`, `precision_ensemble` for `creddata-test`), copied by
hand from the CredData scorer's output (a converter is TBD: [secrets-code](../specialists/secrets-code/README.md#evaluate)).
Stopping `run` after `verify` lets the release record the exams: `pbtrain run` without `--stages` also runs `report`
and `release`, and `release --public` then bundles that release, whose MANIFEST holds only the exams recorded before
it. The `data` stage needs the fingerprints of every exam the recipe lists, and the release refuses a run whose `data`
stage did not check every exam of its specialist.

The same commands run on an NVIDIA, AMD or Apple silicon GPU through stock PyTorch, or on a CPU: `--device auto` takes
CUDA or ROCm first, then Apple silicon (MPS), then the CPU, and says which one it chose. Only the AMD Radeon path has
been run by the maintainers so far. The CPU is the reference. [Training on any hardware](../training/docs/gpu.md) has
the PyTorch build for each, the memory per job and which ones the maintainers have run.

The secrets-code recipe adds labeled windows of CredData DEV to every training step. Without `--exam-overlap drop`,
`build_creddata_stream.py` writes, bit for bit, the stream the released weights were trained with (Python 3.14), but
847 of its windows share content with CredData TEST files and the `data` stage refuses it; with the option those
windows are left out. Details: [data/README.md](../data/README.md#the-creddata-dev-stream).

## What the factory guarantees

Each stage (`data`, `baselines`, `finetune`, `eval`, `calibrate`, `export`, `verify`, `report`, `release`) writes a
`done.json` with what it read and produced, and refuses to run unless every earlier stage ran, in order, on the same
task and on the results on disk (a task edited after `data` is run again from `data`).

- **No leakage**: splits are disjoint by construction (separate ranges of the corpora, or a hash of each document's id
  for the pii documents), and the `data` stage checks the three pairs of splits (whole-window hashes, or 120-byte
  chunks of real content in the first 300 windows of each split) and measures the 64-byte blocks they share; no train
  or select window, real-window set or guard window may share a 64-byte block with a registered exam of role `test`
  (exams of role `dev` are counted and reported, not enforced). Exactly what is checked, and why:
  [evaluate.md](../training/docs/evaluate.md#what-the-leak-checks-check).
- **No silent keys**: unknown keys, keys that act on nothing, values of the wrong type and `data:` keys the generator
  did not read fail the run.
- **Pre-registered criteria**: written in the task before training, frozen by hash by the first `data` stage of the
  version (they cannot change afterwards under that version), each one PASS / FAIL / PENDING; a FAIL is recorded and
  shown (the release checklist asks for none; the code blocks a release only on the delivery gate).
- **A control and a rival**: spans shuffled between windows must give ~0 (span-F1 at most 0.10); the model must beat
  the classic rule of its specialist by more than the spread between seeds, and a specialist without one is refused
  at `baselines`.
- **Runtime equivalence**: the exported file gives the same spans in the engine (or the NumPy reference) as in Python,
  on 40 test windows, half of them with a span.
- **Reproducibility**: on CPU the trainer reproduces, bit for bit, the weights of the code the released models were
  trained with, on small golden cases compared on the platform that recorded them (Windows x86-64, Python 3.10.14,
  PyTorch 2.4.1+cpu, NumPy 2.2.1, one thread; not in CI); the released weights were trained on a GPU, and a retraining
  reproduces them only approximately. Releases record the code commit, every sha256, and where and with what the seeds
  were trained.

**How the three released examples relate to these rules.** They were trained while these checks were being built.
Every pre-registered verdict is published, failures included, in the runtime's
[benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md), and
[release.md](../training/docs/release.md#recorded-exceptions-of-the-released-specialists) lists them:

- `secrets-code` 1.0.0 trained three seeds, not six; its two false-alarm criteria on code-fp-b failed, and it was
  released after a sample of such findings was read; its released seed was chosen on code-fp-b, with the CredData TEST
  scores of the three seeds in view, not on `select`.
- `secrets-bin` 1.0.0 failed its probe criterion on two of its six seeds (356 of 360; the released seed found all
  360), on a probe that is synthetic and in distribution; that exam and one more are private.
- `pii` 1.0.0 passes its criterion at the operating point the reading was moved to after the criterion was written
  (by the criterion as first written it fails), and its data mix shipped against the pre-registered tie rule.
- Their training data overlap some exams: nine code-fp repositories (secrets-code), 847 windows of the CredData DEV
  stream that share blocks with CredData TEST files and four TEST repositories in the code corpus of the time
  (secrets-code; the published CredData figures leave those four out), about 3 % of the pii pool's documents (the pii
  figures are published with and without the 1,018 affected benchmark sentences), and 10 exam files of secrets-bin.
  Today the `data` stage refuses a training window that shares a block with a registered `test` exam.

## Pages

- [Train a new specialist](../training/docs/new-specialist.md), in ten steps
- [Evaluate honestly](../training/docs/evaluate.md): exams, seeds, criteria, leakage
- [Publish a model](../training/docs/release.md)
- [Recipes](../training/docs/recipes.md): every key of a task file, the sizes, the secrets generator
- [Training on any hardware](../training/docs/gpu.md): CPU, NVIDIA, AMD and Apple silicon; backends, memory, the Radeon kernels
- [Known pitfalls](../training/docs/pitfalls.md)
- The specialists: [secrets-code](../specialists/secrets-code/README.md), [secrets-bin](../specialists/secrets-bin/README.md),
  [pii](../specialists/pii/README.md)
