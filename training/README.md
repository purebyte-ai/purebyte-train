# training: the `pbtrain` package

The training stack of PureByte's AI Specialists: from a task file to a released GGUF that the runtime's engine runs
(and the golden path an AI coding agent follows for Vibe Training). This is the `training/` folder of purebyte-train.
The runtime (engine, CLI, model format, benchmarks) is a separate repository,
[purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte). Start with [docs/training.md](../docs/training.md);
the playbooks are in [docs/](docs/).

From the root of a checkout, in a virtual environment ([Quick start](../README.md#quick-start)):

```bash
python -m pip install -e "training[test]"
python -m pbtrain doctor
python data/build_smoke_data.py                                           # made-up stand-ins for the smoke recipes (seconds, offline)
python -m pbtrain run specialists/pii/recipes/smoke-cpu.yaml --stages data,baselines,finetune,eval,calibrate,export,verify,report
python -m pbtrain run specialists/secrets-code/recipes/smoke-cpu.yaml     # minutes on a CPU; ends at the gate (GATE NOT PASSED, exit 2)
python -m pytest training/tests -q                                        # the contract (CPU, offline)
```

The first smoke run trains and exports a tiny model on made-up documents in about two minutes
(`$PUREBYTE_WORK/runs/pii-smoke/0.0.1/export/model.gguf`); the second proves the stages up to the gate, which a model
trained on so few steps does not pass, the documented outcome.

Comments in the code cite the model format as `spec/FORMAT.md` and `spec/OUTPUT.md`: both live in the runtime
repository, [purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte/tree/main/spec).

| Module | What it holds |
|---|---|
| `pbtrain/model/` | the byte model in PyTorch; block and head types named as in the runtime's engine (`ssm_v2`, `attention`, `tag`, `choice`...) |
| `pbtrain/train/` | the span trainer, recipes, real-window streams, the tail guard, pretrained bodies and adapters, multi-seed runs |
| `pbtrain/factory/` | the stages (`data` ... `release`), pre-registered criteria, the done.json chain, runtime verification, report |
| `pbtrain/verticals/` | the plugin contract of specialists and their discovery (`specialists/<name>/plugin.py`) |
| `pbtrain/data/` | byte corpora, content fingerprints and the exam-leak checks |
| `pbtrain/eval/` | span metrics decoded like the runtime's engine, the operating point, classic rivals and the gate, calibration |
| `pbtrain/export/` | the GGUF writer and the model -> engine format mapping (the runtime's [spec/FORMAT.md](https://github.com/purebyte-ai/purebyte/blob/main/spec/FORMAT.md)) |
| `pbtrain/release/` | immutable releases and public bundles (GGUF + sha256 + model card) |
| `pbtrain/gpu/` | GPU backends: stock PyTorch on CUDA, ROCm and Apple silicon, and the fused Radeon HIP kernels ([docs/gpu.md](docs/gpu.md)) |
| `tests/` | golden training fingerprints, generator golden, export -> pinned NumPy reference / runtime's engine, factory smoke, units, CLI |

Invariants (each one is a test):

- the trainer reproduces, bit for bit on CPU, the weights of the code the released models were trained with
  (`test_golden_training.py`, 27 trainer paths). These fingerprints are compared only on the platform that recorded
  them (PyTorch 2.4.1+cpu, NumPy 2.2.1, Python 3.10, Windows AMD64, AVX2, one thread): **CI does not compare them**
  (`PBTRAIN_GOLDEN_SKIP=1`), the maintainers do before every release, with the command in
  [AGENTS.md](../AGENTS.md#build-and-test). Anywhere else they skip and say why; `test_training_is_deterministic`
  runs everywhere;
- the secrets generator reproduces its data bit for bit (`test_secrets_data_golden.py`);
- every exported file loads in the NumPy reference (the pinned copy in [reference/](../reference/README.md)) and in the
  runtime's engine and computes what PyTorch computes, with identical spans (`test_export_reference.py`; set
  `PUREBYTE_MODEL_DUMP` to the engine's `model_dump` tool, built as in [AGENTS.md](../AGENTS.md#build-and-test));
- the factory refuses leaks (in the data, and in a stream or an exam set changed after the data stage), typos,
  changed criteria, a task edited after `data`, stages run out of order (`test_factory_smoke.py`); the release refuses
  a failed gate, too few members, an unchecked exam of the specialist and overwriting a release, and makes the public
  bundle in both orders of the documented procedure (`test_release.py`, on a real chain whose gate is forced either
  way, so that it does not depend on how good a smoke model is);
- the rules of `--device` / `--backend`, the GPU lock and, with `PBTRAIN_GPU_TESTS=1` on a GPU, parity with the CPU and
  run-to-run determinism (`test_devices.py`, `test_gpu_backend.py`).
