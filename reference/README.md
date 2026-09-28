# Reference implementation (a pinned copy)

`purebyte_ref/` is an exact copy of the NumPy reference implementation of the model format, whose source of truth is
[reference/ in purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte/tree/main/reference). It is here so that
the training stack can check every file it exports without cloning the runtime:

- the `verify` stage runs the exported model in it when `PUREBYTE_MODEL_DUMP` does not point at the engine's faster
  `model_dump` tool, and requires the same spans as PyTorch;
- `training/tests/test_export_reference.py` does the same on every exported layout.

Do not edit it here. A change to the reference is a change to the model format: it goes to the runtime repository
first (specification, engine, reference, golden tests), then this copy is refreshed from there. CI compares the copy
byte for byte with the runtime's `reference/purebyte_ref/` at the release tag named by `RUNTIME_REF` in
[.github/workflows/ci.yml](../.github/workflows/ci.yml) (`v1.0.0` today), and fails if they differ; the tag is moved
on purpose, in the same change that refreshes the copy.

It is slow on purpose (Python loops where the order of operations matters) and is not meant for production use:
the runtime is.
