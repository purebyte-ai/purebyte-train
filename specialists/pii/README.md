# pii

An AI Specialist that finds personal data in any text and masks it, for **redaction by copy**. It marks spans over
the input's bytes, all of one type, `PII`; the runtime's `redact` profile then copies the input byte for byte and
replaces only those spans with markers such as `[PII_1]` ([spec/OUTPUT.md](https://github.com/purebyte-ai/purebyte/blob/main/spec/OUTPUT.md), section 9). Outside
the spans, the output is the input. The only possible errors are measurable: something leaks (recall) or too much is
masked (precision).

**Status: version 1.0.0 is released.** It was trained with the structured-forms variant of the recipe
([recipes/pii-1.0.0.yaml](recipes/pii-1.0.0.yaml), as written), which shipped against the pre-registered tie rule (why:
[Train](#train)). Its criterion was written before any model of it was scored; it passes at the default operating
point, where the reading was moved after the criterion was written, and fails at the point first written
([Evaluate](#evaluate)). The model card is
[models/pii.md](https://github.com/purebyte-ai/purebyte/blob/main/models/pii.md), in the runtime repository.

## What it masks

One type, `PII`: every span that the public PII datasets it learns from label as personal data. That is the policy of
the [PII Masking Benchmark](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/piimb/README.md), and it is broad on purpose:

| Family | Examples of source labels |
|---|---|
| People | given names, surnames, full names, titles, gender, age, nationality, occupation, education, marital status |
| Contact | e-mail addresses, phone and fax numbers, street addresses, building numbers, postcodes, cities, states, countries, coordinates |
| Identifiers | national identity, passport, driver license, tax and social security numbers, customer, employee, account, medical record, device and vehicle identifiers, license plates |
| Finance | payment card numbers and codes, IBAN, BBAN, SWIFT/BIC, routing numbers, PINs |
| Online | usernames, passwords, API keys, URLs, IP and MAC addresses, cookies |
| Dates and organizations | dates and times, dates of birth, company names |

The sources call all of these personal data (OpenPII-1M 19 labels, Nemotron-PII 54, Gretel PII masking EN 42, Gretel
finance 29); `pii` makes no distinction between them: one type, one marker. What it does **not** do: tell an e-mail
address from a name (an experimental typed variant does, [below](#experimental-typed-spans)), check that a value is
valid or real, or guarantee that nothing leaks.

## Data

Windows of 512 bytes cut from the **release pool**: labeled documents of the TRAIN splits of four public datasets,
rebuilt by [data/build_pii_pool.py](../../data/build_pii_pool.py) from files pinned by revision and sha256
([data/sources/pii-pool.yaml](../../data/sources/pii-pool.yaml); details in
[data/README.md](../../data/README.md#the-pii-pool)):

| Source (TRAIN split) | License | Documents in the pool | Languages |
|---|---|---:|---|
| The sample of [data/sources/pii.yaml](../../data/sources/pii.yaml) (the four datasets below) | CC BY 4.0, Apache 2.0 | 32,500 | English, Spanish |
| Ai4Privacy OpenPII-1M | CC BY 4.0 | 199,326 | 23: bg, cs, da, de, el, en, es, et, fi, fr, hr, hu, it, lt, lv, nl, pl, pt, ro, sk, sl, sr, sv |
| NVIDIA Nemotron-PII | CC BY 4.0 | 78,305 | English |
| Gretel PII masking EN v1 | Apache 2.0 | 42,418 | English |
| Gretel synthetic PII finance multilingual | Apache 2.0 | 34,842 | 7: de, en, es, fr, it, nl, sv |
| **Total** | | **387,391** | |

Every document is synthetic personal data made by its dataset's generator: no value was taken from a real person,
although a generated value can coincide with a real one. Every label of the source is personal data: one type, `PII`,
nothing ignored. A document goes to train / select / test by a stable hash of its id (about 90 / 5 / 5 %); the release
recipe draws 192,000 training windows, one pass over them. 855 ids of the base sample are shared by two documents each
(its sampler's numbering: [data/README.md](../../data/README.md#the-pii-pool)); both documents of a pair always fall in
the same split.

**The hard overlap check.** A candidate document is never written when its normalized text, one of its sentences or
lines (40 characters or more), its masked template (every labeled span replaced by a placeholder), or one of its masked
sentences (60 characters or more) is also in an evaluation document: the PII Masking Benchmark (its 25,033 documents and
150,022 sentences), pii-bench-test and pii-bench-dev. It leaves out 29,149 of the candidates, mostly shared boilerplate
("thank you for your attention to this matter."), and also templates shared by a generator's train and test splits and
the same document with other values.

**What remains shared**, measured with the data scripts' 64-byte content blocks: about 3 % of the pool's documents
share at least one block with the benchmark or with pii-bench-test (phrases of one generator across its splits), and
1,018 of the benchmark's 150,022 sentences share one with the pool (none of privy and mapa-eur-lex). The training
factory refuses any training window that shares a block with a `test` exam, so `python -m pbtrain run` stops at the
`data` stage with the release pool; `data/build_pii_pool.py --exam-overlap drop` builds a pool without those documents,
which the factory accepts and which is not the release pool. The benchmark's figures are published with and without the
sentences that share a block (the runtime's [benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md#pii-on-the-pii-masking-benchmark)).

Not used for training, ever: the PII Masking Benchmark (CC BY-NC, evaluation only), the ai4privacy 200k / 300k / 400k /
open-500k sets (licenses), privy (the benchmark's privy test is a fork of it), any test or validation split.

## Train

Requirements: the `pbtrain` package ([training/](../../training/)) and the resources below, all under `$PUREBYTE_DATA`
(`python -m pbtrain doctor` lists what it finds, and names the missing lists, catalogs, pool and fingerprints; a
missing corpus is not named yet):

```bash
# the CPU smoke recipe first (about two minutes, offline; made-up stand-ins, never mixed into the real data); it stops
# before `release`, which refuses a run whose exams have no fingerprints
python data/build_smoke_data.py
python -m pbtrain run specialists/pii/recipes/smoke-cpu.yaml --stages data,baselines,finetune,eval,calibrate,export,verify,report

# the release recipe
python data/fetch_exams.py --set piimb --set pii-bench-test --set pii-bench-dev          # evaluation sets first
python data/fingerprint_exams.py --set piimb --set pii-bench-test --set pii-bench-dev
python data/fetch_pii.py lists --skip-sentences --allow-missing-exams   # pii/lists (the generator and the rules)
python data/build_secret_catalogs.py                # catalogs (the rules rival; the SECRET values of generated scenes)
python data/build_code_corpus.py --allow-missing-exams                   # corpora/code-context.bin (the code windows)
python data/build_pii_pool.py --exam-overlap drop --allow-missing-exams  # pii/public_train_xl (needs pyarrow)
python -m pbtrain run specialists/pii/recipes/pii-1.0.0.yaml --device auto --parallel 3
```

What each option gives up, and why it is there:

- `--allow-missing-exams`: every data build refuses to run until EVERY evaluation set of `data/exams/` has a
  fingerprint file, and the binary, code and CredData sets are large, partly unavailable to the public (15 + 46 files
  without a public download) and, for CredData, fetched on Linux, macOS or WSL only. With the option, the three PII
  sets fingerprinted above are still excluded from everything built (and the pool's four-level check runs against
  them); the other sets are not excluded from the lists, the code corpus or the pool, and each build manifest lists them.
  The training factory still requires the fingerprints of the recipe's exams (`piimb`, `pii-bench-test`). To give up
  nothing, fetch and fingerprint every set first ([data/README.md](../../data/README.md#quick-start)) and drop the option.
- `--exam-overlap drop`: the release pool itself (the default, `keep`) holds about 3 % of documents that share a
  64-byte block with a `test` exam, and the factory's `data` stage refuses it ([below](#data)); `drop` leaves them out.
  The pool is then not the release pool, and a model trained on it is expected to land close to 1.0.0, not bit for bit.
- `--skip-sentences`: Tatoeba replaces its exports every week, so the pinned copies stop being served within days;
  the sentence lists are read only by the prose and e-mail scenes, which the release recipe does not use. Without it,
  `fetch_pii.py lists` stops on the sha256 of a replaced export (or, with `--accept-upstream-change`, builds sentence
  lists that differ from the pinned ones and records it). The INE tables are replaced once a year; after that only
  `--accept-upstream-change` builds the lists, and the generated windows then differ
  ([data/README.md](../../data/README.md#known-provenance-gaps)).

| Recipe | What | Runs on | Status |
|---|---|---|---|
| [recipes/pii-1.0.0.yaml](recipes/pii-1.0.0.yaml) | the release recipe: 192,000 windows, 80 % from the release pool and 20 % structured forms, `nano_n15` (256 x 4 `ssm_v2` blocks, causal, n-gram tables 3 x 2^15 x 64; 1.9 M body parameters and 6.3 M in the tables), six seeds of 12,000 steps, lr 3e-4, batch 16; the seed best on `select` is released | GPU | released as `pii` 1.0.0 |
| [recipes/smoke-cpu.yaml](recipes/smoke-cpu.yaml) | the pool-only data block on minimal shapes: 128-byte windows, `mini64`, 1 seed of 300 steps, on `pii/public_train` (the look-alike documents of `data/build_smoke_data.py`, or the smaller pool of `data/fetch_pii.py train` once it is there) | CPU, about two minutes up to `report` | proves the pipeline, not the model |
| [recipes/experimental-typed.yaml](recipes/experimental-typed.yaml) | 12 typed span types ([below](#experimental-typed-spans)) | GPU | experimental, never trained here |

**The variants** the release recipe was measured in (nothing else changes, and the criteria stay):

| Variant | Lines | What changes |
|---|---|---|
| structured forms (released) | as written | about 20 % of the windows are the generator's records, logs and code ([piigen/](piigen/)), every typed span collapsed into `PII`, and its look-alikes (order ids, creation dates, postcodes, lone cities, company names, URLs...) taught neither way (label -100, [piigen/lookalikes.py](piigen/lookalikes.py)): personal data for the public datasets, negatives for the typed policy |
| release pool only | `data.mix: {code: 0, data: 0, prose: 0, email: 0, log: 0, public: 1.0}`, `data.lookalikes: text` | windows of public documents only. Write the whole `mix`: a scene left out keeps its default share, so `{public: 1.0}` alone would draw about half of the windows from generated scenes, prose and e-mail included (tested in `tests/test_pii_mask.py`) |
| more capacity | `train.size: tiny_n18` | 8 blocks and n-gram tables 3 x 2^18 x 64 (about 29 MiB at 4 bits instead of 4.5 MiB) |

**Why the structured forms ship** (three seeds of each variant, default operating point): on the PII Masking Benchmark
the pool-only variant scored a mean F2 of 0.786 and the structured-forms variant 0.779, a tie by the pre-registered
rule (a difference under 0.01), and that rule keeps the variant measured first, the pool-only one. On a test set of
2,000 generated documents (code, structured records, logs, prose and e-mails, from held-out value lists, formats and
code) the pool-only variant left 26 % of the personal data unmasked (38 % in code, 28 % in structured records, 21 % in
logs) and the structured-forms variant 11 % (1.3 %, 1.9 % and 0.04 %), so the structured forms shipped against the tie
rule: a product decision. That test set comes from the same generator as the structured forms, so it favors them; on
the benchmark they score lower, and the published figures of `pii` 1.0.0 are theirs. More capacity was a loss; a
bidirectional model, a tie.

The `code` windows of the structured-forms variant are cut from `corpora/code-context.bin`, which the data scripts
rebuild from public repositories; the released weights were trained with an earlier corpus, so those windows (4 % of
the total) differ. Everything else is rebuilt bit for bit ([Status](#status)) under three conditions: the release pool
(the default `--exam-overlap keep`, which the factory refuses: see [Data](#data)), the pinned INE lists (served only
until INE replaces its tables: [data/README.md](../../data/README.md#known-provenance-gaps)), and Python 3.10 to 3.13
(a build on Python 3.10.14 gives the released arrays bit for bit, checked at full scale; on Python 3.14 the SECRET
values of the structured forms differ, because the credential generator validates its samples with catalog rules that
only 3.14 compiles).

## Evaluate

**The gate** (factory stage `eval`, on the pool's own `test` documents): the model against the rules of
[piigen/rules.py](piigen/rules.py) (patterns with checksums, context words, the name lists, the credential catalog),
every rule span counted as `PII`, both scored exactly like the model (micro span-F1, exact boundaries). The model must
beat them by more than the spread between seeds, and the shuffled-spans control must collapse.

**The criterion** (pre-registered in the recipe): the [PII Masking Benchmark](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/piimb/README.md), its
`sentences` subset, character-level F2, label-agnostic, the mean of its six tasks (the leaderboard's number), measured
with the `purebyte` CLI on each seed's file at its default operating point. The mean of the six seeds must score a
mean-of-6 F2 above 0.6621 (openai/privacy-filter, 1.4 B parameters), an ai4privacy-en F2 above 0.8426 (its headline
number) and a mean precision of at least 0.70 (masking every character scores F2 0.4423 at precision 0.157). Every seed
and the minimum are reported, and so is the released file's own score.

**Where it is read was changed after it was written.** As first written on 2026-09-23, the criterion was read at the
"PIIMB mode" point below. On 2026-09-24, after the pool-only variant had met it at the default point and missed the
precision floor at "PIIMB mode", the reading was moved to the default operating point, before any model of the
released variant was scored. At the default point the verdict is pass; by the criterion as first written it is fail
(mean precision 0.686 over the six seeds at "PIIMB mode", below 0.70), as the runtime's
[benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md#pii-on-the-pii-masking-benchmark)
publishes.

`run.sh` is in the runtime repository only
([benchmarks/piimb/](https://github.com/purebyte-ai/purebyte/tree/main/benchmarks/piimb)) and needs the `purebyte`
CLI; `score.py` has a pinned copy here, in `benchmarks/piimb/`. From the root of this repository, with the runtime
repository cloned at `/path/to/purebyte`; every output goes to a folder of `$PUREBYTE_WORK`, never into the source tree
(`run.sh` writes `piimb-results/` into the folder it runs in):

```bash
W="${PUREBYTE_WORK:-$HOME/.purebyte/work}/pii-eval" && mkdir -p "$W"
python specialists/pii/eval/export_seeds.py specialists/pii/recipes/pii-1.0.0.yaml --out "$W/seeds"
(cd "$W" && for f in seeds/seed_*.gguf; do bash /path/to/purebyte/benchmarks/piimb/run.sh /path/to/piimb "$f"; done)
python benchmarks/piimb/score.py metrics --released "$W"/piimb-results/seed_<released>/scores.json \
    --seeds "$W"/piimb-results/seed_*/scores.json --out "$W/piimb-metrics.json"
python -m pbtrain eval specialists/pii/recipes/pii-1.0.0.yaml --record piimb --metrics "$W/piimb-metrics.json"
```

**Beside it, not judged:** the "PIIMB mode" operating point
([eval/piimb_mode.py](eval/piimb_mode.py)): per seed, the bias and window gate that maximize the byte F2 on the run's
own `select` documents cut into sentences (never on the benchmark), and the benchmark scored there. The benchmark ranks
by F2 and scans single sentences; the default operating point is chosen for F1 on 512-byte windows.

| Measure | Value | Conditions |
|---|---|---|
| Rules, span-F1 on the pool's `test` windows | 0.444 (precision 0.847, recall 0.301) | the gate, `PII` for every rule span; 2,000 windows |
| Model, span-F1 on the pool's `test` windows | 0.824 mean of six seeds (0.816 to 0.830; the released seed 0.825) | six seeds: the gate passes by +0.379, 73.7 times the spread between seeds; the shuffled-spans control scores 0.0 |
| PII Masking Benchmark, mean-of-6 F2 / ai4privacy-en F2 / mean precision | six seeds' mean 0.7774 / 0.9624 / 0.765 (criterion met at the default point, where the reading was moved); the released file 0.7689 / 0.9569 / 0.773 | the criterion above; every seed, the "PIIMB mode" point and the figures without the sentences that share content with the training data in the runtime's [benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md#pii-on-the-pii-masking-benchmark) |

## Status

Built and tested: the two verticals, the look-alike policy, the recipes and their variants, the data scripts of the
release pool, the per-seed export and the benchmark's scorer. Checked against the code the released model was trained
with, on the same resources: the release pool rebuilds byte for byte from the pinned downloads (its five files match
the digests in `data/sources/pii-pool.yaml`); the `data` stage of the release recipe gives the same arrays, bit for bit,
at full scale (192,000 training windows, every split) with the corpus the released weights were trained with, and so
does its pool-only variant; the trainer gives the same weights (a few steps of `nano_n15` and `tiny_n18` on release windows, with
and without IGNORE bytes, and the control); the finetune stage chooses the same operating point and the same seed on
the same checkpoints (the six of the release); and the exported file gives the same spans in the engine as in Python.
Those checks ran on Python 3.10.14. The generated windows are the same on Python 3.10 to 3.13 and differ on 3.14 in
their SECRET values: the credential generator refuses, on every version, a catalog pattern with a global inline flag
after its start (Python 3.10 alone would compile it), but only 3.14 compiles `\z`, which three sampled gitleaks rules
use ([data/README.md](../../data/README.md#the-creddata-dev-stream), condition 1). The public-document windows of the
release recipe (80 %) do not depend on the version. The golden cases lock the generator on the resources they use:
[tests/](tests/) (a digest of each data block on generated resources, whose made-up catalog has no `\z` rule, and the
weights of the pii training cases in `tests/golden/pii_training_fingerprints.json`, which match only on the machine
that recorded them: Windows x86-64, PyTorch 2.4.1+cpu, NumPy 2.2.1, one thread; elsewhere they skip, and CI does not
compare them). Offline tests: `python -m pytest specialists/pii/tests -q`.

## Experimental: typed spans

The same generator also trains a TYPED variant ([recipes/experimental-typed.yaml](recipes/experimental-typed.yaml),
vertical `pii_spans`): 12 types (`EMAIL`, `PHONE`, `IBAN`, `CREDIT_CARD`, `IP`, `DNI_NIE`, `PASSPORT`, `DATE_OF_BIRTH`,
`URL_CREDENTIALS`, `SECRET`, `PERSON_NAME`, `ADDRESS`; 49 labels) with typed markers such as `[EMAIL_1]`, trained on the
generator's scenes (real code with its own personal data re-planted as synthetic values, records in 16 formats, prose,
e-mails, logs; 18 % of the windows with nothing to redact) and 30 % public documents whose labels outside the taxonomy
are ignored. The labelling policy of each type is in [piigen/types.py](piigen/types.py). It is kept as a documented
alternative for inputs where the type matters; it is not released and has not been trained with this stack. Its
policy leaves alone what the benchmark counts (companies, occupations, generic dates, most identifiers), so it cannot
score well there, and its criteria need an exam runner for typed spans that does not exist yet. Those criteria also name
an evaluation set, `pii-contexts-test`, that nothing builds yet, and metrics (`exact_f1_minus_rules.*`,
`hybrid_leak_ratio`, `share_of_types_passing`...) that are defined nowhere yet: the recipe's header says "defined in
README.md", and they are not. Its data stage runs; its exam criteria stay PENDING until that set and its runner exist.

## Licenses and attribution

The code is licensed like the rest of this repository. A model trained with the release recipe contains information
derived from: Ai4Privacy / Ai Suisse SA OpenPII-1M (CC BY 4.0) · NVIDIA Nemotron-PII (CC BY 4.0) · Gretel.ai datasets
(Apache 2.0). The structured-forms variant and the typed recipe also draw on: INE, www.ine.es (CC BY 4.0) · Tatoeba,
tatoeba.org (CC BY 2.0 FR) · US Census Bureau (public domain) · Faker (MIT) · the credential formats of gitleaks and
trufflehog (see the secrets-code specialist). The model card carries the attribution
([MODEL_CARD.template.md](MODEL_CARD.template.md)).

## Files

| Path | What |
|---|---|
| `specialist.yaml` | name, plugin, `uses: [secrets-code]`, what the exported model declares, the release policy, the exams |
| `plugin.py` | the verticals `pii_mask` (released policy) and `pii_spans` (typed, experimental): data block, resources, windows, guards, manifest, the rules rival |
| `piigen/` | the generator (types, validators, values, names, forms, generator, public_map, rules, metrics) and `lookalikes.py` |
| `recipes/` | the release recipe, the CPU smoke recipe, the experimental typed recipe |
| `eval/` | `export_seeds.py` (every seed as a file, for the criteria) and `piimb_mode.py` (the second operating point) |
| `tests/` | offline tests on generated look-alike resources, and the golden training cases |
| `MODEL_CARD.template.md` | the model card the release step fills in |
