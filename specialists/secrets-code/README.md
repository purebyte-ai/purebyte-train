# secrets-code

An AI Specialist that finds credentials (API keys and tokens, passwords, private keys, connection strings with passwords) in source code and
configuration files. A 1.9 M-parameter ternary byte model (`nano`: 256 x 4 `ssm_v2` blocks) with a span head and a
window gate. Three seeds are trained: the released one runs alone by default, and the other two ship as the optional
members of an ensemble with a 2-of-3 vote (`secrets-code-ensemble` in the runtime's [models.json](https://github.com/purebyte-ai/purebyte/blob/main/models/models.json)).

**Where 1.0.0 departs from the rules of this stack** (the runtime's
[benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md#the-pre-registered-criteria-and-the-choice-of-the-released-model)
publishes the verdicts): it trained three seeds, not six; its two false-alarm criteria on code-fp-b failed, and it was
released after a sample of such findings was read; and its released seed was chosen on code-fp-b, with the CredData
TEST scores of the three seeds in view, not by the factory's rule (the best seed on `select`, seed 2).

| File | What it is |
|---|---|
| `specialist.yaml` | export profile `secrets-code`, label `secret`, window 512 / stride 384, ensemble policy, exams |
| `plugin.py` | the `secrets_synthetic` vertical (also used by secrets-bin) and the classic rival |
| `secretsgen/` | the generator: credential lines (`credentials.py`), real formats from catalogs (`catalog.py`), windows (`task.py`), the smoke stand-ins (`standins.py`) |
| `recipes/secrets-code-1.0.0.yaml` | the recipe of the released 1.0.0 (three seeds, seed 1 released, labeled real windows in every step, its pre-registered criteria and their outcome) |
| `recipes/secrets-code-1.1.0-candidate.yaml` | an earlier candidate: the first candidate's recipe with lr 3e-4 and six seeds (superseded by 1.0.0, which already trains at lr 3e-4) |
| `recipes/smoke-cpu.yaml` | the whole pipeline on CPU in minutes, on the stand-ins of `data/build_smoke_data.py` |
| `tests/` | the Quick start on a fresh data folder (stand-ins, both smoke recipes), the refusal of stand-ins elsewhere |

## Data

Windows of REAL code and configuration (`$PUREBYTE_DATA/corpora/code-context.bin`, built by
`data/build_code_corpus.py` from public repositories pinned by commit, with the exam repositories excluded by identity
and by content) with one generated line injected on a line boundary, positive or negative at 50 %, sometimes inside a
block of up to five lines. Positive lines follow real provider formats (sampled from the gitleaks rules and validated
against the rule when this Python compiles it; trufflehog gives detector names for credential variable names, and a
verbatim example string of its tests in the rare draws, 0.035 %, where no rule yields a valid value;
`data/build_secret_catalogs.py`), connection strings, PEM blocks and generic secrets under credential names; negatives
are what looks like a credential and is not: placeholders, environment lookups, references, hashes, identifiers, public
keys, documentation examples, ordinary short values. The span mask marks the value bytes. The generated windows are the
same on Python 3.10 to 3.13 and differ on 3.14, which compiles three more sampled rules
([data/README.md](../../data/README.md#the-creddata-dev-stream), condition 1); the released weights were trained on
data generated with Python 3.10.14.

The 1.0.0 recipe adds real windows to every training step (`train.real_windows`, one window in five): lines labeled
per span in CredData repositories of the DEV half, only from a fixed 117 of its 169 repositories (the rest of DEV is
kept for measuring; no TEST repository is ever read), with half of the credential values replaced by generated values
of the same form. `python data/build_creddata_stream.py` builds it (`$PUREBYTE_DATA/streams/creddata-dev-spans.npz`):
by default, bit for bit the stream the released weights were trained with (with Python 3.14); with
`--exam-overlap drop`, without its 847 windows that share content with CredData TEST files, which the training
factory's leak check refuses ([data/README.md](../../data/README.md#the-creddata-dev-stream)), and without the windows
of the five repositories below.

**Overlap with the false-alarm exams.** Five repositories of the code-fp sets are CredData DEV repositories of that
`train` subset, so the released weights were trained on labeled windows of them (at CredData's commits, not at the
exams' ones): tornado (code-fp-a), starlette, got and devise (code-fp-b) and awesome-compose (code-fp-c), 641 of the
stream's 24,000 windows, 274 of them with a credential. The false-alarm counts of 1.0.0 on those three sets are
therefore partly measured on code the model has seen, and code-fp-b, on which the released seed was chosen (the fewest
findings of the three seeds, with the CredData TEST scores of the three in view), holds three of the five. Without them, counted again per repository on 2026-09-25 (the
released file, the CLI's default options; this count gives exactly the published totals of code-fp-a and code-fp-b):
code-fp-a 73 findings instead of 93 (tornado had 20), code-fp-b 339 instead of 408 (starlette 2, got 47, devise 20).
Four more code-fp-c repositories (nginx, helm-charts, dotfiles, home-manager), and awesome-compose again, were in the
earlier code corpus the released weights were trained with; without those five, code-fp-c gives 51 findings (this
count gives 146 for all twelve repositories where 138 were published, 63 of them in awesome-compose). The runtime's
[benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md#false-alarms-on-repositories-without-credentials)
publishes the same rows. Three code-fp-c repositories (ansible-for-devops, helm-charts, phoenix) are CredData TEST
repositories, never read for training; in all eight, CredData labels some lines as credentials (89 in total), so the
code-fp premise "no real credential" does not hold for every line of them
([data/README.md](../../data/README.md#overlap-with-the-code-fp-exams)).

## Train

```bash
python data/build_smoke_data.py          # made-up stand-ins for the smoke recipe (seconds, offline)
python -m pbtrain run specialists/secrets-code/recipes/smoke-cpu.yaml    # ends at the gate on a CPU (exit status 2)
python -m pbtrain run specialists/secrets-code/recipes/secrets-code-1.0.0.yaml --device auto --parallel 3
```

The smoke recipe runs on the stand-ins of `data/build_smoke_data.py` (a generated code-like corpus and two made-up
credential formats); the plugin refuses those stand-ins for any other recipe. It proves the data, the leak checks, the
rival, three seeds, the control and the gate, not a model: on a CPU it takes minutes (6.8 minutes with 2 threads on a
desktop Ryzen 9 5900X that was running other work) and ends with `GATE NOT PASSED: does not beat the classic rival`,
exit status 2, its documented outcome. The `pii` smoke recipe goes through export and
`verify` in about two minutes ([Quick start](../../README.md#quick-start)). The released recipe needs the data of
[data/README.md](../../data/README.md#quick-start): the code corpus, the catalogs, the CredData stream built with
`--exam-overlap drop` (the factory refuses the released stream), and the fingerprints of its four exams.

The released weights were trained on a GPU (25 minutes for three seeds and the control on an RX 6800 XT with the Radeon
backend), with a code corpus assembled before `data/` existed, which cannot be rebuilt, and with the released CredData
stream. A retraining from the data scripts differs in both (the rebuilt corpus, and the stream without its overlapping
windows): as the recipe says, it is expected to land close to 1.0.0, not bit for bit. Given the same windows, the
trainer computes bit for bit what the code the weights were trained with computes, on the CPU of the machine that
recorded the golden fingerprints (`training/tests`, compared there, not in CI).

## Evaluate

The published figures, with their conditions, are in the runtime's
[benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md#secrets-code-on-creddata) (CredData TEST, split by repository,
gitleaks and trufflehog in the same harness):

| Exam | 1.0.0 | Conditions |
|---|---|---|
| CredData TEST, per line | F1 0.797 (precision 0.879, recall 0.729) | the released model alone at its operating point; 164 repositories (TEST without the four that overlap the training corpus), files up to 4 MB, as in [benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md#secrets-code-on-creddata) |
| same, ensemble of three | F1 0.805 (precision 0.879, recall 0.741) | the three seeds, each at its own operating point, 2-of-3 vote; same harness |
| synthetic test (factory) | span-F1 0.880 (released seed); 0.891 mean of the three seeds (0.880-0.898) | 2,000 generated windows; control with shuffled spans 0.000; classic rival 0.330 |

Running the exams on a release (`$PUREBYTE_WORK/specialists/secrets-code/<version>/`, the ensemble's GGUF files):

```bash
# false alarms on repositories with no credential (one exam folder from data/fetch_exams.py); outputs outside the tree
W="${PUREBYTE_WORK:-$HOME/.purebyte/work}/secrets-code-eval" && mkdir -p "$W"
python specialists/secrets-code/eval/count_findings.py --models RELEASE_DIR --exam EXAM_DIR --out "$W/code-fp-b.json"
python -m pbtrain eval specialists/secrets-code/recipes/secrets-code-1.0.0.yaml --record code-fp-b --metrics "$W/code-fp-b.json"
# CredData TEST: benchmarks/creddata/ of the runtime repository (score.py is pinned here); see below for its metrics file
```

The CredData criteria read `f1_ensemble` and `precision_ensemble` on the complete TEST partition. The scorer's
`score.py score ... --json FILE` writes every number under its own names (`results`, keyed by partition, scope and
model), not under those: copy the two into a JSON object (`{"f1_ensemble": ..., "precision_ensemble": ...}`) and record
that file with `--record creddata-test`. A converter is TBD.

Known weaknesses, from the released model's errors on the DEV repositories never used in training:
[benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md#known-weak-spots). The synthetic battery does not predict the
exam results: judge a new recipe on the exams, with six seeds, against pre-registered criteria
([training/docs/evaluate.md](../../training/docs/evaluate.md)).

## Notes on the generator

- `secretsgen/credentials.py` is an export, with its comments removed, of the generator the maintainers trained the
  released models with; this file is now the source of truth, and `training/tests/test_secrets_data_golden.py` locks
  its output bit for bit. A change to it is a new generator version: a new recipe version and a new golden file.
- `data.version: v3` and `v4` are two labels of the same generator (`secretsgen/task.py`): they record which data a
  model saw and select nothing else. The key `hard` is accepted only as `true` (a generator without the hard families
  is not built).
- When no catalog rule yields a valid value (0.035 % of the catalog draws with the pinned catalog), the generator
  plants one of trufflehog's test strings verbatim as a positive (AGPL-3.0 test data; [data/README.md](../../data/README.md#sources-training-material)).
  The next generator version is to fall back to a generated look-alike instead; changing it now would change the
  locked data.
