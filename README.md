<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/logo-lockup-dark.png">
    <img src="docs/assets/logo-lockup-light.png" alt="PureByte" width="420">
  </picture>
</p>

<h3 align="center">Train tiny AI Specialists on your own data, on the hardware you have,<br>or describe one to an AI agent and let it train it.</h3>

<p align="center">
  <a href="https://github.com/purebyte-ai/purebyte-train/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/purebyte-ai/purebyte-train/actions/workflows/ci.yml/badge.svg"></a>
  <a href="LICENSE"><img alt="License: Apache-2.0" src="https://img.shields.io/badge/license-Apache--2.0-C6F16B?labelColor=0D0F13"></a>
  <img alt="Python 3.10+" src="https://img.shields.io/badge/python-3.10%2B-5EEAD4?labelColor=0D0F13">
  <img alt="PyTorch 2.4+" src="https://img.shields.io/badge/PyTorch-2.4%2B-5EEAD4?labelColor=0D0F13">
  <a href="training/docs/gpu.md"><img alt="Tested on: CPU, AMD Radeon (Windows)" src="https://img.shields.io/badge/tested%20on-CPU%20%C2%B7%20AMD%20Radeon%20(Windows)-C6F16B?labelColor=0D0F13"></a>
  <a href="https://github.com/purebyte-ai/purebyte"><img alt="Runtime: purebyte" src="https://img.shields.io/badge/runtime-purebyte-C6F16B?labelColor=0D0F13"></a>
</p>

<p align="center">
  <a href="#what-this-is">What this is</a> ·
  <a href="#vibe-training">Vibe Training</a> ·
  <a href="#quick-start">Quick start</a> ·
  <a href="#train-on-any-hardware">Hardware</a> ·
  <a href="#the-golden-path">The golden path</a> ·
  <a href="#what-the-factory-guarantees">Guarantees</a> ·
  <a href="#the-three-examples">Examples</a> ·
  <a href="#repository-map">Map</a>
</p>

## What this is

**purebyte-train** is the training stack of PureByte's **AI Specialists**: tiny byte-level models that each do one job
very well, read raw bytes (text, code, logs, JSON, binaries) and answer with decisions, scores, labels and byte
ranges instead of generated text. It takes a task, written as a short file, to a released model file that the open
runtime, [purebyte](https://github.com/purebyte-ai/purebyte), runs on any CPU, offline, with the same bits from every
kernel and thread count of a platform, and the same decisions and spans expected across platforms.

Today the factory trains span specialists, like the three examples: a head that tags bytes, gated by a decision on
the whole window. The model format and the runtime also run decision, score, grade and per-byte-map heads; their
training recipes are not in the factory yet.

A specialist does not need to know about the world: it needs to know its job. So it trains from scratch, on data
that describes that job (generated look-alikes grounded in real bytes, plus your own labeled data:
[how](training/docs/new-specialist.md#bring-your-own-labeled-data)), in minutes to an hour on one consumer GPU, and
ends up at 1 to 29 MiB. **Train with your data, not a generalist.**

| Part | What it is | Where |
|---|---|---|
| **`pbtrain`** | The package and CLI: the byte model in PyTorch, the trainer, the factory stages (`data` → `baselines` → `finetune` → `eval` → `calibrate` → `export` → `verify` → `report` → `release`), evaluation, export to the model format | [training/](training/README.md) |
| **Specialists** | One folder per AI Specialist: task file, plugin, data generator, recipes, tests, model card template. The three released examples are here | [specialists/](specialists/README.md) |
| **Data scripts** | Download and build every dataset from pinned public sources (URLs, commits, SHA-256, licenses). Datasets never enter git | [data/](data/README.md) |
| **Pinned runtime pieces** | Exact copies of the runtime's NumPy reference and benchmark scorers, so that every exported file is checked without cloning the runtime; CI compares them byte for byte with those of the runtime's release tag it pins | [reference/](reference/README.md), [benchmarks/](benchmarks/README.md) |

## Vibe Training

**Vibe Training** is training an AI Specialist by describing what you want to an AI coding agent, the way vibe coding
builds software. A request can be this short:

```text
Train an AI Specialist that finds IBANs in CSV exports, and tell me honestly how good it is.
```

The agent follows the golden path of [AGENTS.md](AGENTS.md#the-golden-path-vibe-training): it creates the specialist,
downloads or generates the data, writes the success criteria **before** training, trains several seeds and a control,
evaluates on real data, and releases the model with its card. The stack makes every step checkable, so neither you nor
the agent has to trust the other: leaked evaluation data, a typo in a recipe, a criterion changed after it was frozen
or an exported file that the runtime reads differently all stop the run with a message that says why. What the checks
cover, and where the three released examples depart from them, is [below](#what-the-factory-guarantees).

Why an agent training a model, instead of an agent writing code or reasoning at run time? For tasks that are about
patterns in bytes (is this a credential, a name, a malformed record, a known file format?), a learned model captures
what hand-written rules miss, and running it takes about a millisecond on a CPU for a short input (0.82 ms for 64
bytes and 2.8 ms for 256 bytes, with 8 threads of a desktop Ryzen 9 5900X: the runtime's
[performance](https://github.com/purebyte-ai/purebyte/blob/main/docs/performance.md)), where a call to a hosted model
takes a network round trip (third-party measurements of a hosted decision API give 236 to 276 ms per call, median;
cited, not measured, in the runtime's comparison). The agent does the engineering once; the specialist then does the
work locally, every time.

## Quick start

The whole pipeline, on a CPU, in minutes, on made-up data (it proves the machinery, not a model):

```bash
git clone https://github.com/purebyte-ai/purebyte-train
cd purebyte-train
python3 -m venv .venv && . .venv/bin/activate   # Windows: py -m venv .venv, then .venv\Scripts\Activate.ps1
python -m pip install -e "training[test]"        # Python 3.10+, PyTorch 2.4+ (requirements below)
python -m pbtrain doctor                         # what is installed and what is found
python data/build_smoke_data.py                  # made-up stand-ins for the smoke recipes (seconds, offline)
python -m pbtrain run specialists/pii/recipes/smoke-cpu.yaml --stages data,baselines,finetune,eval,calibrate,export,verify,report
python -m pytest -q training/tests               # the training contract, offline
```

The smoke run trains a tiny `pii` model on look-alike documents (about two minutes on a desktop CPU), checks it against
its classic rival and its control, exports it and checks that the runtime's reference gives the same spans as PyTorch.
Try the file it made with the runtime (`pip install purebyte`, or a release binary):

```bash
purebyte redact --model ~/.purebyte/work/runs/pii-smoke/0.0.1/export/model.gguf notes.txt
```

Runs are written under `$PUREBYTE_WORK` (default `~/.purebyte/work`; on Windows `%USERPROFILE%\.purebyte\work`). The
`release` stage is left out on purpose: it refuses a run whose exams were never fingerprinted, and made-up data
have none. The smoke recipe of `secrets-code` runs the same way and, on a CPU in minutes, ends at the gate: with so few
steps its model does not beat the classic rival, and `GATE NOT PASSED` (exit status 2) is its documented outcome.

**Requirements.** Python 3.10 or newer and PyTorch 2.4 or newer, any build (CPU, CUDA, ROCm, Apple silicon). CI runs
PyTorch 2.4.1 with NumPy 2.2 on Linux, Windows and macOS; the maintainers train with a PyTorch 2.10 ROCm build.
PyTorch 2.2 and 2.3 work only with NumPy 1.x: on an Intel Mac, where PyTorch stops at 2.2, run
`python -m pip install "torch==2.2.2" "numpy<2"` (Python 3.10 to 3.12) before installing the package.

## Train on any hardware

AI Specialists are small (the released examples have 1.9 to 54 million parameters), so they train on what you have.
The CPU is the reference: on the machine that recorded the golden fingerprints (Windows x86-64, Python 3.10.14,
PyTorch 2.4.1+cpu, NumPy 2.2.1, one thread), the trainer reproduces bit for bit what the code that trained the
released examples computes; other processors and builds round differently, so those comparisons skip there. The
released weights themselves were trained on a GPU: a retraining lands close to them, not bit for bit.

```bash
python -m pbtrain run TASK --device auto --parallel 3     # the best device found: cuda, then mps, then cpu
```

| Hardware | `--device` | Status |
|---|---|---|
| Any x86-64 or Arm CPU (Linux, Windows, macOS) | `cpu` (default) | The reference; tested in CI on the three systems |
| AMD Radeon on Windows (ROCm) | `cuda` (+ `--backend radeon` for fused kernels) | Tested: the released examples were trained on an RX 6800 XT with a PyTorch 2.10 ROCm build ([the build](training/docs/gpu.md#the-tested-rocm-build-on-windows)); the device-parity test passes on it |
| AMD on Linux (ROCm), NVIDIA (CUDA) | `cuda` | The same PyTorch code path as above; not yet run by the maintainers |
| Apple silicon (M1 or newer) | `mps` | Stock PyTorch operations in float32; not yet run by the maintainers |

`training/tests/test_devices.py` checks that every GPU it finds trains like the CPU; a report from an NVIDIA or Apple
machine is welcome. Details, memory per job and the rules (one GPU job at a time, a missing GPU is an error, never a
silent CPU run): [Training on any hardware](training/docs/gpu.md).

## The golden path

What the agent runs, and what you can run by hand:

```bash
python3 -m venv .venv && . .venv/bin/activate      # 1. set up (Windows: py -m venv .venv, then .venv\Scripts\Activate.ps1)
python -m pip install -e "training[test]"
python -m pbtrain doctor

python data/fetch_exams.py                          # 2. data: the evaluation sets FIRST, then what the builds exclude
python data/fetch_creddata.py                       #    CredData (Linux, macOS or WSL)
python data/fingerprint_exams.py                    #    sets without a public download are fingerprinted incomplete (data/README.md)

python -m pbtrain new-vertical invoice-ids          # 3. a new specialist; then write its generator, classic rival and tests

python -m pbtrain run specialists/invoice-ids/recipes/smoke-cpu.yaml                    # 4. the pipeline on a CPU
# 5. copy smoke-cpu.yaml to invoice-ids-0.1.0.yaml, set the sizes and write its criteria:, then train (all stages,
#    up to the immutable internal release)
python -m pbtrain run specialists/invoice-ids/recipes/invoice-ids-0.1.0.yaml --device auto --parallel 3

python -m pbtrain eval TASK --record EXAM --metrics m.json   # 6. exams on real data; six seeds decide
python -m pbtrain release TASK --public --semver 0.1.0       # 7. from that release: a .gguf, its SHA-256 and its card
```

Step 4 needs the classic rival of step 3: until it exists, the `baselines` stage refuses the run. Step 7 checks that
the internal release of step 5 is this task's and intact, then writes the public bundle. The release records the exams
recorded before it; to have them in its MANIFEST, stop step 5 after `verify` (`--stages
data,baselines,finetune,eval,calibrate,export,verify`) and let step 7 write the report and the release too
([Publish a model](training/docs/release.md)).

The guide: [docs/training.md](docs/training.md). The playbooks: [train a new specialist](training/docs/new-specialist.md)
in ten steps, [evaluate honestly](training/docs/evaluate.md), [recipes](training/docs/recipes.md),
[publish a model](training/docs/release.md), [known pitfalls](training/docs/pitfalls.md).

## What the factory guarantees

Each stage writes a `done.json` with what it read and produced, and refuses to run unless every earlier stage ran, in
order, on the same task and on the results on disk. What each check covers exactly:
[evaluate.md](training/docs/evaluate.md#what-the-leak-checks-check) and [release.md](training/docs/release.md).

- **No leakage**: train, select and test are disjoint by construction (separate byte ranges of a corpus, or a hash of
  each document's id), and the `data` stage checks every pair with the guard of the specialist (whole-window hashes of
  every window, at most 0.1 % identical, or 120-byte chunks of real content in the first 300 windows, none shared). No
  window of train or select, of a real-window stream or of the guard may share a 64-byte block with a registered exam
  of role `test`; sets of role `dev`, which training streams may use, are counted and reported, not enforced.
- **No silent keys**: unknown keys, keys that act on nothing and data keys the generator did not read fail the run.
- **Pre-registered criteria**: written in the task before training and frozen by hash when the `data` stage first runs
  for that version; a later stage, or a new `data` run of the same version, refuses criteria that changed. Each one is
  PASS, FAIL or PENDING, never re-tuned after looking at an exam. A FAIL is recorded and shown; the code blocks a
  release only when the gate fails, and the release checklist asks for no FAIL and six seeds.
- **A control and a rival**: labels shuffled between windows must give about zero (span-F1 at most 0.10); the model
  must beat the classic rule of its specialist by more than the spread between seeds, and a specialist without one is
  refused at `baselines`.
- **Runtime equivalence**: the exported file gives the same spans in the engine (or the NumPy reference) as in PyTorch,
  on 40 test windows, half of them with a span.
- **Reproducibility**: on the machine that recorded the golden fingerprints, the trainer reproduces them bit for bit
  (CI and other machines skip that comparison; the maintainers run it there before each release); a release records
  the code commit and every SHA-256.

**How the three released examples relate to these rules.** They were trained while these checks were being built.
Every pre-registered verdict is published, failures included, in the runtime's
[benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md), and their recipes
record the same:

- `secrets-code` 1.0.0 trained three seeds, not six. Its two false-alarm criteria on code-fp-b failed, and it was
  released after a sample of such findings was read. Its released seed was chosen on code-fp-b, with the CredData TEST
  scores of the three seeds in view, instead of by the rule of this stack (the best seed on `select`).
- `secrets-bin` 1.0.0 failed its probe criterion on two of its six seeds (356 of 360 decidable credentials found; the
  released seed found all 360). The probe is synthetic and in distribution: its credentials come from the training
  generator. It and one more exam are private.
- `pii` 1.0.0 passes its criterion at the operating point the reading was moved to after the criterion was written; by
  the criterion as first written it fails. Its data mix shipped against the pre-registered tie rule.
- Their training data overlap some exams. Nine code-fp repositories were in the training data of `secrets-code`; 847
  windows of its CredData DEV stream share 64-byte blocks with CredData TEST files, and its code corpus held four TEST
  repositories (the published CredData figures leave those four out). About 3 % of the `pii` pool's documents share a
  block with the PII Masking Benchmark or pii-bench-test (its figures are published with and without the 1,018
  affected sentences). 10 exam files of `secrets-bin` overlap its earlier binary corpus (its figures are also given
  without them). Today the `data` stage refuses any training window that shares a block with a registered `test` exam,
  and the data scripts leave the exam repositories and files out of what they build (`--exam-overlap drop` for the
  CredData stream and the pii pool).

## The three examples

Each is one recipe in [specialists/](specialists/README.md), trained from scratch with no pretrained body on one AMD
Radeon RX 6800 XT (16 GB), and one model file that the runtime runs with no change to its code.

| Example | Model | Training time | Result (conditions and rivals in the runtime's [benchmarks](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md)) |
|---|---|---|---|
| [`pii`](specialists/pii/README.md): personal data in prose, code, JSON, CSV, SQL and logs | `nano_n15`, 8.3 M parameters | 7.5 minutes per seed | PII Masking Benchmark F2 0.769, above OpenAI's Privacy Filter (0.662; 1.4 B parameters, about 50 M active) |
| [`secrets-code`](specialists/secrets-code/README.md): leaked credentials in code and configuration | `nano`, 1.9 M parameters | 24.5 minutes for three seeds and the control | CredData F1 0.797 (TEST without the four repositories that overlap its training corpus), against 0.337 for gitleaks |
| [`secrets-bin`](specialists/secrets-bin/README.md): credentials inside compiled binaries | `tiny_n18`, 54 M parameters | 58 minutes for six seeds and the control | all 360 decidable credentials found (of 400 planted) on a synthetic probe, whose credentials come from its training generator |

All three, with the bias sweeps and the checks of every stage: about 2.5 GPU-hours. The models, their cards and the
download commands are in the runtime repository: [models/](https://github.com/purebyte-ai/purebyte/tree/main/models).

## Repository map

| Path | Contents |
|---|---|
| [`training/`](training/README.md) | the `pbtrain` package, its tests (golden training fingerprints, export against the reference and the engine, factory smoke) and playbooks |
| [`specialists/`](specialists/README.md) | `secrets-code`, `secrets-bin`, `pii`: task files, plugins, generators, recipes, evaluation, model card templates |
| [`data/`](data/README.md) | fetch and build scripts with their manifests; the data lives in `$PUREBYTE_DATA` (default `~/.purebyte/data`) |
| [`reference/`](reference/README.md) | pinned copy of the runtime's NumPy reference implementation |
| [`benchmarks/`](benchmarks/README.md) | pinned copies of the runtime's CredData and PII benchmark scorers |
| [`docs/training.md`](docs/training.md) | the entry page of the documentation |
| [`AGENTS.md`](AGENTS.md) | the golden path and the invariants, for AI coding agents and for humans |

Runs and releases go to `$PUREBYTE_WORK` (default `~/.purebyte/work`). No dataset, run or weight is ever committed.

## Contributing, security and license

- New specialists, data sources, hardware backends, fixes and documentation are welcome: [CONTRIBUTING.md](CONTRIBUTING.md).
  AI coding agents: [AGENTS.md](AGENTS.md).
- Please follow the [Code of Conduct](CODE_OF_CONDUCT.md). Vulnerabilities: privately, as in [SECURITY.md](SECURITY.md).
- **Code**: [Apache License 2.0](LICENSE). **Datasets** fetched by the data scripts keep their own licenses, recorded
  in their manifests. **The released example weights** are published with the runtime under the
  [PureByte Model License](https://github.com/purebyte-ai/purebyte/blob/main/MODEL_LICENSE.md); any model you train
  yourself with this code is yours.
- A specialist for your own data, trained for you: [pablo@purebyte.ai](mailto:pablo@purebyte.ai).

## Author

Created and built by **Pablo Sirvent Jiménez**:

- **Websites:** [sirvent.ai](https://sirvent.ai) · [pablosirvent.com](https://pablosirvent.com)
- **GitHub:** [@sirventai](https://github.com/sirventai)
- **X / Twitter:** [@sirventai](https://x.com/sirventai)
- **Blog:** [sirventai.medium.com](https://sirventai.medium.com)
- **Contact:** [pablo@purebyte.ai](mailto:pablo@purebyte.ai)

---

<p align="center">
  <sub>Built by <a href="https://sirvent.ai">Pablo Sirvent Jiménez</a> (<a href="https://github.com/sirventai">@sirventai</a>) · <a href="https://purebyte.ai">purebyte.ai</a> · the runtime: <a href="https://github.com/purebyte-ai/purebyte">purebyte-ai/purebyte</a></sub>
</p>
