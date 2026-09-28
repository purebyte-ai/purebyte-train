# Contributing to purebyte-train

Thank you for helping. This is the training stack of PureByte's AI Specialists, and it makes strict promises: the
trainer is reproducible bit for bit on the machine that recorded its golden fingerprints, success criteria are written
before training, no evaluation data reaches a training set, and no labeled value is ever printed. This page is mostly
about keeping them while you change things.
If you use an AI coding agent, point it to [AGENTS.md](AGENTS.md), which has the same rules in more detail.

## Ways to contribute

| Contribution | Where to start |
|---|---|
| Report a bug | An issue with the command, its output (values masked) and `python -m pbtrain doctor` |
| Improve a specialist, or add a new one | [docs/training.md](docs/training.md) and [Train a new specialist](training/docs/new-specialist.md), then an issue with the task and its success criteria |
| Add a data source | [data/README.md](data/README.md): pinned URL or commit, SHA-256, license, and a test |
| Support more hardware | [Training on any hardware](training/docs/gpu.md); a backend must keep the CPU reference and its determinism rules |
| Change the model format, a block or a head | In the runtime first: [purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte) |
| Report a vulnerability | Privately, as described in [SECURITY.md](SECURITY.md) |

## Scope

This repository holds the `pbtrain` package (`training/`), the specialists' recipes, plugins and data generators
(`specialists/`) and the scripts that download and build their data from pinned public sources (`data/`).

It never holds dataset files, model weights or training runs: pull requests that add them cannot be accepted. Data
goes to `$PUREBYTE_DATA`, runs and releases to `$PUREBYTE_WORK` (defaults: `~/.purebyte/data` and `~/.purebyte/work`), and released
weights are published next to the runtime under the
[PureByte Model License](https://github.com/purebyte-ai/purebyte/blob/main/MODEL_LICENSE.md).

## Set up

Python 3.10 or newer and PyTorch 2.4 or newer, any build: CPU, CUDA, ROCm or Apple silicon (PyTorch 2.2 and 2.3 need
NumPy 1.x: [requirements](README.md#quick-start)).

```bash
git clone https://github.com/purebyte-ai/purebyte-train.git
cd purebyte-train
python3 -m venv .venv && . .venv/bin/activate   # Windows: py -m venv .venv, then .venv\Scripts\Activate.ps1
python -m pip install -e "training[test]"
python -m pbtrain doctor
python -m pytest -q training/tests
```

CI installs the CPU wheels of `training/requirements-ci.txt` (PyTorch 2.4.1, NumPy 2.2.1) and runs `training/tests`
on Linux with Python 3.12 (with the export check against the C++ engine), on Windows and macOS with Python 3.12 and on
Linux with Python 3.10; every `specialists/*/tests` on Linux; and `data/tests` in a job of their own, with the latest
NumPy, PyYAML and pytest, on Python 3.10 and 3.12. It does not run `pbtrain doctor`, and it skips the golden weight
fingerprints (rule 1 below). The export check against the engine needs the runtime's `model_dump` tool:
[AGENTS.md](AGENTS.md#build-and-test) shows how to build it.

## The rules that keep the training honest

1. **Reproducibility.** The trainer reproduces its golden fingerprints bit for bit on the machine that recorded them
   (Windows x86-64 with AVX2 kernels, Python 3.10.14, PyTorch 2.4.1+cpu, NumPy 2.2.1, one thread). CPU kernels round
   differently elsewhere, so CI and your machine skip that comparison: keep the trainer's parameter order, random draws
   and floating-point order, or say in the pull request that you changed them. The maintainers compare the
   fingerprints on the recording machine before merging such a change and before each release. A change that alters
   them says why and regenerates them on purpose, never to make a test pass.
2. **Pre-registered criteria.** Success criteria are written in the task before training and frozen by hash by the
   first `data` stage of a version (new criteria need a new version); results are never re-tuned after looking at an
   exam.
3. **No leakage.** No training window may share a 64-byte block with a registered exam of role `test`; the `data`
   stage checks it ([exactly what it checks](training/docs/evaluate.md#what-the-leak-checks-check)).
4. **The exported file is the model.** It must load in the runtime and give the same spans as PyTorch.
5. **Nothing leaves the machine.** No telemetry; the only network access is the data scripts' downloads of their
   pinned sources. Never verify a credential against any service.
6. **Values are never printed.** Mask or hash every labeled value, in logs, errors, tests and reports. Tests use
   look-alike values generated for the test.
7. **English everywhere**: code, comments, messages, documentation.

## Pull requests

- Keep each pull request to one purpose, with tests and documentation in the same change.
- Describe what changes for users, and paste the relevant test output.
- Commit messages: an imperative summary line (for example `train: stream real windows in chunks`), then the why.
- Sign off your commits (`git commit -s`) to certify the
  [Developer Certificate of Origin](https://developercertificate.org/). Contributions are accepted under the
  Apache-2.0 license of the repository.
