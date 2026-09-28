# Benchmark scorers (pinned copies)

These two files are exact copies of the public scorers of the runtime repository, whose source of truth is
[benchmarks/ in purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte/tree/main/benchmarks), where the
evaluations of the released specialists and their results live:

| File | Used here by |
|---|---|
| `creddata/score.py` | the CredData split, units and categories of `data/fetch_creddata.py`, `data/build_creddata_stream.py` and `data/pbdata/creddata.py` (its SHA-256 is recorded in the split it produces) |
| `piimb/score.py` | the PII masking benchmark metrics of `specialists/pii/eval/piimb_mode.py` and its tests |

Training and evaluation must score exactly as the published benchmarks do, so these files are not edited here: a
change goes to the runtime repository first, then the copy is refreshed. CI compares both byte for byte with the
runtime's files at the release tag named by `RUNTIME_REF` in [.github/workflows/ci.yml](../.github/workflows/ci.yml)
(`v1.0.0` today), moved on purpose with the copies. They use the Python standard library only.
