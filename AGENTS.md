# AGENTS.md

Instructions for AI coding agents (and for humans who like checklists) working in this repository. Read this file
before changing anything. [CONTRIBUTING.md](CONTRIBUTING.md) has the same rules in a shorter form.

## What this repository is

The training stack of PureByte's **AI Specialists**: tiny byte-level models that each do one thing very well, on a
CPU, privately and measurably. It takes a task file to a released model file that the open runtime,
[purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte), runs on any CPU:

- `training/`: the `pbtrain` package and CLI (`python -m pbtrain`): the byte model in PyTorch, the trainer, the
  factory stages, evaluation, export to the model format, releases;
- `specialists/<name>/`: one folder per specialist (task file, plugin, data generator, recipes, tests, model card
  template). The three released examples are `secrets-code`, `secrets-bin` and `pii`;
- `data/`: the scripts that download and build the data from pinned public sources (URLs, commits, SHA-256,
  licenses). Never data.

When a user asks you to train an AI Specialist, that is **Vibe Training**: follow [the golden path](#the-golden-path-vibe-training)
step by step and show them the checks at every step. Entry page: [docs/training.md](docs/training.md); playbooks in
[training/docs/](training/docs/).

No dataset file and no model weight is ever committed: data lives in `$PUREBYTE_DATA`, runs and releases in
`$PUREBYTE_WORK` (defaults: `~/.purebyte/data` and `~/.purebyte/work`).

## Map

| Path | Contents |
|---|---|
| `training/pbtrain/model/` | the byte model in PyTorch; block and head types named as in the engine (`ssm_v2`, `attention`, `tag`, `choice`, ...) |
| `training/pbtrain/train/` | THE span trainer (`loop.py`), recipes (`recipe.py`), real-window streams, tail guard, bodies/adapters, multi-seed runs |
| `training/pbtrain/factory/` | stages `data -> baselines -> finetune -> eval -> calibrate -> export -> verify -> report -> release`, criteria |
| `training/pbtrain/verticals/` | the plugin contract (`base.py`) and the discovery of `specialists/<name>/plugin.py` |
| `training/pbtrain/data/` | corpora, content fingerprints, the exam-leak checks |
| `training/pbtrain/eval/` | span metrics decoded like the engine, the operating point, classic rivals and the gate, calibration |
| `training/pbtrain/export/` | model -> GGUF following the runtime's [spec/FORMAT.md](https://github.com/purebyte-ai/purebyte/blob/main/spec/FORMAT.md) (full coverage or it raises) |
| `training/pbtrain/release/`, `gpu/` | releases and public bundles; the device and GPU backends |
| `training/tests/` | the training contract (golden fingerprints, export -> reference/engine, factory smoke) |
| `specialists/<name>/` | `specialist.yaml`, `plugin.py`, the generator, `recipes/*.yaml`, README, model card template |
| `data/` | download / build scripts and manifests (URLs, commits, sha256, licenses); never data |

## Build and test

Requirements: Python 3.10 or newer and PyTorch 2.4 or newer (CPU, CUDA, ROCm or Apple silicon builds; only the CPU and
an AMD Radeon under Windows have been run by the maintainers: [Training on any hardware](training/docs/gpu.md)).
PyTorch 2.2 and 2.3 were built against NumPy 1.x: with them, install `numpy<2` (on an Intel Mac, where PyTorch stops at
2.2: `python -m pip install "torch==2.2.2" "numpy<2"` first). CI tests Python 3.10 and 3.12 with PyTorch 2.4.1 and
NumPy 2.2.1; the golden fingerprints were recorded with Python 3.10 (below); PyTorch 2.2 is not tested in CI. pbtrain
runs from a checkout of this repository (`pip install -e`): the specialists, the NumPy reference and the data scripts
live next to `training/`.

```bash
python3 -m venv .venv && . .venv/bin/activate   # Windows: py -m venv .venv, then .venv\Scripts\Activate.ps1
python -m pip install -e "training[test]"
python -m pbtrain doctor
python -m pytest -q training/tests
for suite in specialists/*/tests; do python -m pytest -q "$suite"; done
python -m pytest -q data/tests
```

- The export tests run every exported file in the NumPy reference and, when `PUREBYTE_MODEL_DUMP` points at the
  runtime's `model_dump` tool, in the C++ engine too. Build it from the runtime repository:

  ```bash
  git clone https://github.com/purebyte-ai/purebyte
  cmake -S purebyte -B purebyte/build -DCMAKE_BUILD_TYPE=Release -DPUREBYTE_BUILD_CLI=OFF -DPUREBYTE_BUILD_SHARED=OFF
  cmake --build purebyte/build --target model_dump --parallel --config Release
  export PUREBYTE_MODEL_DUMP=$PWD/purebyte/build/tests/model_dump
  ```

  On Windows, the Visual Studio generator (CMake's default when Visual Studio is installed) ignores
  `CMAKE_BUILD_TYPE` and puts the tool in a folder per configuration, hence `--config Release`; in PowerShell:
  `$env:PUREBYTE_MODEL_DUMP = "$PWD\purebyte\build\tests\Release\model_dump.exe"` (with Ninja or MinGW Makefiles:
  `...\build\tests\model_dump.exe`).

- The golden weight fingerprints are compared only where the machine matches their recording in every field of their
  `meta`: PyTorch 2.4.1+cpu, NumPy 2.2.1, Python 3.10, Windows on AMD64, CPU kernels AVX2
  (`torch.backends.cpu.get_cpu_capability()`), one thread. Elsewhere they skip and say why (`-rs`); the determinism
  tests run everywhere. **CI does not compare them**: every CI job sets `PBTRAIN_GOLDEN_SKIP=1`, because a hosted
  runner's processor is not guaranteed to match. The maintainers compare them on the recording platform before every
  release, and a pull request that touches `training/pbtrain/model/`, `train/`, `gpu/` or `runtime.py` must be
  compared there too (`requirements-ci.txt` pins the libraries, not Python 3.10 or the CPU):

  ```bash
  py -3.10 -m venv venv-golden && venv-golden\Scripts\activate          # cmd.exe on Windows, AMD64, an AVX2 processor
  pip install -r training/requirements-ci.txt
  python -m pytest -q -rs -p no:cacheprovider training/tests/test_golden_training.py specialists/pii/tests/test_pii_training_golden.py
  ```

  Every case must pass; a skipped case is not a pass.
- CI runs the tests above on CPU (not `pbtrain doctor`): `training/tests` with the engine check and every
  `specialists/*/tests` on Linux with Python 3.12; `training/tests` on Windows and macOS with Python 3.12 and on Linux
  with Python 3.10; `data/tests` in a job of their own (latest NumPy, PyYAML and pytest) on Python 3.10 and 3.12. The
  golden fingerprints are skipped everywhere, as above.
- Scratch and run directories go to `$PUREBYTE_WORK`, never inside the source tree.

## The golden path (Vibe Training)

1. In a virtual environment, `python -m pip install -e "training[test]"`, then `python -m pbtrain doctor`.
2. Data from `data/` (exams and their fingerprints FIRST: `fetch_exams.py`, `fetch_creddata.py` on Linux, macOS or
   WSL, `fingerprint_exams.py`, which fingerprints the sets without a public download incomplete; then corpora and
   catalogs: [data/README.md](data/README.md#quick-start)).
3. `python -m pbtrain new-vertical NAME` for a new specialist; write its generator, its classic rival and its tests.
4. `python -m pbtrain run specialists/NAME/recipes/smoke-cpu.yaml` (CPU, minutes; the smoke recipes of the released
   specialists need `python data/build_smoke_data.py` first, and on so few steps their gate may refuse the model: the
   run then stops at `eval` with exit status 2, a correct outcome).
5. Write the recipe WITH its pre-registered `criteria:` (they are frozen by the first `data` stage of that version);
   `python -m pbtrain data TASK`, `python -m pbtrain train TASK --device auto --parallel 3` (the CPU, or an NVIDIA,
   AMD or Apple silicon GPU through stock PyTorch; only the AMD Radeon path has been run by the maintainers:
   [Training on any hardware](training/docs/gpu.md)), `python -m pbtrain export TASK`.
6. Exams on real data: `python -m pbtrain eval TASK --record EXAM --metrics FILE`; six seeds decide.
7. `python -m pbtrain release TASK --public --semver X.Y.Z`: the report (with the exams recorded so far), the immutable
   internal release and the public bundle. When the internal release exists already (`pbtrain run` runs every stage,
   `release` included), it is checked and only the bundle is made.

## Invariants

Breaking one of these is a blocking defect, whatever else the change achieves.

### The training contract

1. **Bit-identity of the trainer.** On CPU, `pbtrain` reproduces bit for bit the weights of the code the released
   models were trained with (`training/tests/golden/training_fingerprints.json`, 27 trainer paths; the pii cases, one
   type with IGNORE bytes and the `nano_n15` preset, in `specialists/pii/tests/golden/`), on the platform that recorded
   them (compared there before every release, not in CI: [Build and test](#build-and-test)). These are fingerprints of
   tiny test runs; the released weights themselves were trained on a GPU and a retraining reproduces them only
   approximately. Refactors keep the parameter creation order, the RNG draws and the floating-point order. Never
   regenerate a golden file to make a test pass.
2. **The generator is locked** (`training/tests/golden/secrets_data_fingerprints.json`, and the digests of the pii
   tests): every generator option at its default draws no random number, so a new option never changes the data of an
   existing recipe.
3. **The exported file is the model**: it loads in the NumPy reference and in the engine and gives the same spans as
   Python (`training/tests/test_export_reference.py`; `PUREBYTE_MODEL_DUMP` enables the engine check).
4. **No leakage.** Splits are disjoint by construction (separate byte ranges of the corpora, or a hash of each
   document's id for the pii documents). What the `data` stage
   checks, exactly ([evaluate.md](training/docs/evaluate.md#what-the-leak-checks-check)):
   - between the three pairs of splits (train-select, train-test, select-test), what the vertical's `overlap_guard`
     enforces: `windows`, whole-window hashes of every window, at most 0.1 % of the second split identical (a
     generator can repeat a short window, a public dataset can hold one document under two ids); or `chunks`, 120-byte
     chunks at the fixed offsets 0, 120, 240... of the first 300 windows of each split, none shared. It also
     MEASURES, without enforcing, the windows of each split that share a 64-byte block with another split (every
     window, any offset: license headers and generator templates repeat legitimately); the report shows it;
   - train, select, the real-window stream and the guard windows against every registered exam: one shared 64-byte
     block with an exam of role `test` fails the stage. Exams of role `dev` (training material may come from them)
     are counted and reported, not enforced. The exams the task lists, with its specialist's when it lists any, must
     be registered; the release refuses a run that did not check every exam of its specialist;
   - `finetune` and `release` refuse a stream, or a set of registered exams, other than the ones `data` checked.
5. **Nothing silent**: unknown keys, keys that act on nothing, values of the wrong type or out of range and `data:`
   keys the generator did not read fail the run, in the task and in `specialist.yaml`.
6. **Pre-registered criteria**: written before training, frozen by hash by the first `data` stage of a version (a
   `data` stage re-run on other criteria under the same version is refused: change the version); results are PASS /
   FAIL / PENDING and are never re-tuned after looking at an exam. Three seeds inform, six decide; criteria read the
   max seed. The code blocks a release only on the delivery gate (control, recall floor, a flat bias sweep of the
   released seed, margin over the rival); FAIL criteria and the six seeds are checklist items that you must respect
   and show. A criterion on an exam that the task's `exams:` does not list is recorded with a warning.
7. **The done.json chain and immutable releases**: every stage checks that each earlier stage ran, in order, on the
   same task (its digest) and on the results on disk (the sha256 of each previous done.json), and deletes its own and
   every later done.json before it runs. A release records the task, the code commit, every sha256, the seeds and how
   the released one was chosen, where and with what they were trained, the verdicts and the warnings; it is never
   overwritten, and a public bundle refuses internal paths or run names in its card and metadata.
8. **One GPU job at a time** (`pbtrain.runtime.gpu_lock`, an operating-system lock released when its owner ends); CPU
   work at idle priority with capped threads.

### Privacy and security

1. **Never print, log or store a detected or labeled value in clear**: not in errors, debug output, test logs,
   assertions or reports. Mask it (first characters + length) or hash it with a local key.
2. **No network access** except the data scripts' downloads of their pinned public sources. No telemetry.
3. **Never verify a credential** against any service, not even in tests, not even from a public dataset.
4. **Tests use generated look-alike values**, preferably assembled at test time, so that the repository holds no
   literal that could pass for a live credential or a real person's data.

## Style

- **Python**, type hints on public functions, no network access in tests, CPU by default.
- **Docs.** American English, short sentences, sentence-case headings, one topic per page, links instead of
  repetition, conditions next to every number, TBD for what is not measured. The product terms are written **AI
  Specialists** and **Vibe Training**, always with these capitals.

## What not to do

- Do not commit data, weights, runs, credentials (even from a dataset) or personal data.
- Do not choose seeds, ensembles or operating points on an exam; do not report a number without its conditions.
  (`train.released_seed` overrides the choice on `select`; the release records the override, the seed `select` would
  have chosen and `released_seed_reason`. secrets-code 1.0.0 is a recorded exception:
  [release.md](training/docs/release.md#recorded-exceptions-of-the-released-specialists).)
- Do not add an option to the trainer without a golden case, nor a block or head type that the runtime does not
  implement first (spec, engine, NumPy reference in [purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte)).
- Do not change code inside a block without mirroring it in `training/pbtrain/gpu/radeon/`.

## Out of scope

- Dataset files, downloaded or generated (they go to `$PUREBYTE_DATA`; only the scripts and manifests that build them
  are in git), run artifacts and logs.
- Model weights of any kind (they are released separately, next to the runtime:
  [models/](https://github.com/purebyte-ai/purebyte/tree/main/models)).
- Changes to the runtime or the model format: they go to [purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte)
  first.
- Internal names of the maintainers' private tooling, experiments or model versions.

If a task seems to need one of these, stop and ask the maintainers instead.

## Before you finish

- [ ] The relevant tests ran (see [Build and test](#build-and-test)), with the engine check when you touched export.
- [ ] No golden file changed, or the change is explained and the goldens were regenerated on purpose.
- [ ] No value from a dataset can reach any output unmasked.
- [ ] No new network access, no data or weights in the diff.
- [ ] Documentation updated where users will look for it (README, [docs/training.md](docs/training.md), the playbooks).
- [ ] `CHANGELOG.md` updated for user-visible changes.
