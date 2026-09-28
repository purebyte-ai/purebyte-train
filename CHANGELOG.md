# Changelog

All notable changes to purebyte-train, the training stack of PureByte's AI Specialists, are recorded here. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions follow
[Semantic Versioning](https://semver.org/spec/v2.0.0.html). The runtime has its own
[changelog](https://github.com/purebyte-ai/purebyte/blob/main/CHANGELOG.md), and the specialists have their own
versions, recorded in their [model cards](https://github.com/purebyte-ai/purebyte/blob/main/models/README.md).

## [Unreleased]

## [0.1.0] - 2026-09-25

First public release.

### Added

- The training stack: the `pbtrain` package and CLI ([training/](training/README.md)), the recipes and data generators
  of the `secrets-code`, `secrets-bin` and `pii` specialists ([specialists/](specialists/README.md)), and the scripts
  that build their data from pinned public sources ([data/](data/README.md)).
- Exported model files declare their version and license (`general.version`, `general.license`,
  `general.license.name`).
- Training on any hardware ([training/docs/gpu.md](training/docs/gpu.md)): `--device cpu|cuda|mps|auto`, with Apple
  silicon (MPS) support next to the CPU, NVIDIA (CUDA) and AMD (ROCm). The CPU stays the default and the reference;
  `auto` takes CUDA, then MPS, then the CPU, and says which (the CPU included). On a GPU, pbtrain asks PyTorch for
  deterministic algorithms, without TF32. `training/tests/test_devices.py` checks every `--device` / `--backend`
  combination, the GPU lock and the process environment everywhere and, with `PBTRAIN_GPU_TESTS=1`, that every GPU it
  finds trains like the CPU through `train_spans` (losses within 1e-4 relative, log-probabilities within 1e-4) and
  gives the same bits in two runs and in parallel runs. Measured on an RX 6800 XT (ROCm on Windows) only.
- `data/build_creddata_stream.py`: the labeled CredData DEV windows of the `secrets-code` 1.0.0 recipe, bit for bit
  the released ones by default (Python 3.14), or without the windows that share content with CredData TEST and the
  windows of the five code-fp repositories that are CredData DEV repositories (`--exam-overlap drop`, what the training
  factory accepts; every build reports both). `data/fetch_creddata.py` downloads only the labeled files, by FileID
  (about 1 GB instead of tens of GB); `--method official` keeps CredData's own script, with its output captured so that
  no value is printed.
- `data/build_smoke_data.py`: made-up stand-ins (a code-like corpus, two credential formats, PII lists and look-alike
  documents) for the two CPU smoke recipes, offline, in seconds; the plugins refuse them for any other recipe. The
  Quick start trains and exports a tiny `pii` model on them in about two minutes on a CPU.
- `data/fetch_pii.py lists --skip-sentences`: the PII lists without the weekly-replaced Tatoeba exports, which the
  release recipe does not read.
- Pinned copies of the runtime's NumPy reference ([reference/](reference/README.md)) and of its two benchmark scorers
  ([benchmarks/](benchmarks/README.md)), so that exported files are checked without cloning the runtime. CI checks that
  they are identical to those of [purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte) at the release tag it
  pins (`RUNTIME_REF`, `v1.0.0`).
- CI also runs `training/tests` and `data/tests` on Python 3.10, the oldest version the package declares.

### Changed

After an adversarial review of the release candidate. Some of these refuse what the candidate accepted; each refusal
names the key or the stage and says what to do.

- The done.json chain is checked as a whole: every done.json records the task's digest (its parsed content, without
  `description`) and the sha256 of the previous done.json; before it runs, a stage checks every earlier link and
  deletes its own and every later done.json. A task edited after `data` (a real-window stream added to `train:`, for
  instance), a stage re-run out of order or an edited result is refused by every later stage and by the release;
  `finetune` and `release` also refuse a stream, or a set of registered exams, other than the ones `data` checked.
- The pre-registered criteria of a version are frozen by its first `data` stage (`criteria.frozen.json` next to the
  stages); a `data` stage re-run on other criteria under the same version is refused. The report and the release
  MANIFEST say when they were frozen.
- `eval` writes its done.json on every outcome, with `gate_passes: false` and the reasons when the control, the recall
  floor, a flat bias sweep of the released seed (new) or the margin over the rival refuses the model; a margin equal
  to the spread between seeds is now a tie (refused).
- `pbtrain release TASK --public --semver X.Y.Z` makes the public bundle from an existing internal release of the same
  task (the one `pbtrain run` or `pbtrain release TASK` wrote), after checking every file against its MANIFEST;
  `--public`, `--semver` and the version are checked before any stage runs. The internal release is built in a
  temporary folder and moved into place: a failure leaves nothing behind. It needs as many seeds as the specialist's
  `release.ensemble`, and every exam of its specialist checked by `data`. A bundle whose card names another version of
  the specialist than `--semver` is refused.
- The leak check between splits compares train-select and select-test too, the last whole 120-byte chunk of a window
  included, and measures (without enforcing) the windows that share a 64-byte block with another split; DEV exams'
  hits are shown in the report. [evaluate.md](training/docs/evaluate.md#what-the-leak-checks-check) says exactly what
  is checked.
- Validation: every value of `train:` is typed and range-checked (a repeated seed, a string where a number goes, `clip:
  0`, `span_weight` below 1, a `size` that does not exist, `batch` larger than `n_train`, an option declared without
  the one it needs); `name`, `version`, `specialist`, exam names and `--semver` must be safe to use in paths. The
  top-level `baselines:` key and `output_schema.reject_label` are refused (they acted on nothing); `engine: {threads:
  N}` now acts (the engine's `model_dump` threads in `verify`). `specialist.yaml` is validated the same way.
- A criterion on an exam that the task's `exams:` does not list, an exam of the specialist that is not registered, an
  exam the corpus build did not exclude and a `train.released_seed` that overrides the choice on `select` are recorded
  as warnings in done.json, the report and the release MANIFEST. `train.released_seed_reason` states why.
- Provenance: every checkpoint, `finetune/done.json` and the MANIFEST record the device, the backend, the PyTorch,
  NumPy and Python versions, the thread count, the CPU kernels, the digest of the trainer's code and the code commit;
  the MANIFEST separates the training machine from the release machine. The seed cache key includes the device, the
  backend, the trainer's digest, the library versions and the thread count.
- `--device`, `--parallel` and `--backend` exist only on `train` and `run` and are refused where they act on nothing;
  `--backend cpu` with a GPU device and the other mismatched pairs are refused; `--metrics` needs `--record`.
- Exam results are recorded append-only (`exams/<exam>.jsonl`), with the time, the metrics file's sha256 and the
  run's model files; the report stage computes every criterion, exams included, into the report and the MANIFEST.
- `calibrate` measures the document head's calibration error and its noise floor on `select`; `verify` compares
  windows where Python finds spans and windows where it finds none, and refuses a comparison without any span.
- The finetune checkpoints are named `final.pt` (they hold the last step, or the tail guard's choice).
- `gpu_lock` is an operating-system lock (released when its owner ends, however it ends); the doctor probes CUDA and
  ROCm devices in a child process and names the made-up smoke stand-ins of `data/build_smoke_data.py`.
- Checkpoints, `base_checkpoint` and `ngram_init` are loaded with `torch.load(weights_only=True)`.
- The radeon backend refuses windows longer than 512 bytes and platforms other than PyTorch ROCm on Windows, with a
  message; its reduction block sizes are constants.
- `out_lr_mult` and `real_mode: spans` are stable (released specialists train with them).
- `eval` labels a run with `decide: true` `DECIDES` only from six seeds; with three to five it is labeled `INFORMS`.
- `doctor` names every missing corpus with the script that builds it, and writes "catalog".
- The package requires PyTorch 2.4 or newer, except on Intel Macs (PyTorch 2.2 with NumPy below 2, the last build
  there); it declares its license as the SPDX expression `Apache-2.0` (built with setuptools 77 or newer).
- The pii golden fingerprints are compared under the same conditions as the training ones: `PBTRAIN_GOLDEN_SKIP`, and
  the Python minor version and the CPU kernel path of their recording.
- `data/fetch_exams.py` and `data/fingerprint_exams.py` go on past a set that fails and name every set in a summary;
  files that have no public download no longer stop a set: it is fingerprinted from its other files and marked
  incomplete.
- `data/exams/binary-fp.yaml`: two npm members of `bin-fp-c` now pin the files npm publishes; they had been pinned with
  the application-bundled builds of the same versions that the published figures of secrets-bin 1.0.0 were measured
  on (no published finding was in them). Fingerprints of `bin-fp-a/b/c` made before are stale, and so are those of
  `creddata-test` and `creddata-dev` (a comment of `data/exams/creddata.yaml` was corrected; manifests are recorded by
  the SHA-256 of their bytes).
- The documentation says where the three released examples depart from the factory's rules (README.md, "What the
  factory guarantees"), as the runtime's benchmarks/RESULTS.md publishes.

### Fixed

- With PyTorch 2.6 or newer, the seed cache could not be read back (`weights_only` refused the PyTorch version object
  of the provenance record).
- ROCm builds of PyTorch report `torch.cuda.is_available()` with every device hidden: pbtrain also requires a device
  count above zero.

[Unreleased]: https://github.com/purebyte-ai/purebyte-train/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/purebyte-ai/purebyte-train/releases/tag/v0.1.0
