# Train a new AI Specialist

This is the golden path of **Vibe Training**: an AI coding agent (or you) follows it when someone describes the
AI Specialist they want. An AI Specialist here is a byte model that marks spans of one or more entity types in any input (credentials in code,
personal data in text, fields in a log line). This page takes one from nothing to a released GGUF in ten steps. Every
step names the command, what it writes and the check that must pass before the next one.

Prerequisites: `python -m pip install -e "training[test]"` in a virtual environment, from a checkout of this
repository ([Quick start](../../README.md#quick-start): Python 3.10+, PyTorch 2.4+, or PyTorch 2.2 and 2.3 with NumPy
below 2; PyYAML, scikit-learn), and `python -m pbtrain doctor` shows no missing piece you need.

## 1. Scaffold the specialist

```bash
python -m pbtrain new-vertical invoice-ids --entities invoice_id,iban
```

This writes `specialists/invoice-ids/` with `specialist.yaml`, `plugin.py` (a toy generator that already runs, and a
placeholder rival that returns None: the `baselines` stage refuses the run until you write the real one, step 3),
`recipes/smoke-cpu.yaml`, `README.md` and `tests/test_plugin.py`. Nothing is registered anywhere else: the factory
finds the plugin through the task's `specialist:` key. `specialist.yaml` is validated like a task: an unknown key or
a value of the wrong type is refused.

## 2. Write the generator

Replace the toy generator in `plugin.py`. The contract is in `training/pbtrain/verticals/base.py`:

- `build(corpus, cfgd, fracs)` returns three DISJOINT splits `(X, Y, S)`: `X` uint8 `[n, T]` window bytes, `S` uint8
  `[n, T]` span mask (0 outside; 1 for a single entity type, `1 + e` for type `e`; 255 = ignore), `Y` = the window
  holds a span.
- Cut real context from disjoint byte RANGES of the corpus (`corpus.tr_lo .. corpus.tr_hi`, split by `fracs`), with a
  guard equal to what a window reads. Never draw train and test windows from the same range with different seeds:
  different hashes, overlapping bytes.
- Read every key of `data:` you support from `cfgd`. The factory fails on a declared key nobody read.
- Draw every random number from the generator you are given, and none for an option at its default: then adding an
  option never changes the data of an existing recipe.
- Teach negatives that look like positives (placeholders, hashes, references, ordinary values under the same names).
  A model that only ever saw positives next to their names learns the names.

Run `python -m pytest specialists/invoice-ids/tests -q` after every change.

### Bring your own labeled data

The generator does not have to invent everything. There are two ways to train on labeled windows of your own:

1. **Read them in `build()`.** Cut windows from your files and return them with their span masks (`S`: 0 outside,
   `1 + e` inside a span of type `e`, 255 where the loss must ignore a byte), split into train, select and test by file
   or by document, never by window: two windows of one file in different splits share bytes. Files that will judge the
   model stay out of `build()` and are declared as an exam set in `data/exams/` (entries of `type: local` for files
   without a public download; [data/README.md](../../data/README.md#manifest-schema)), so that the `data` stage checks
   every training window against them.
2. **Add them to every batch** as a real-window stream: an `.npz` under `$PUREBYTE_DATA/streams/` with `X` (uint8
   `[N, seq_len]`, the windows) and `S` (0/1 of the same shape: one entity type), and optionally `Y` (equal to
   `S.any(1)`); in the recipe, `train.real_windows` (its path), `real_frac` (real windows per step, as a fraction of
   `batch`) and `real_mode: spans`. This is how `secrets-code` 1.0.0 learns from labeled CredData DEV lines. The
   generated windows of every step stay those of the same run without the stream.

Either way the `data` stage checks the windows against every registered exam, and it records the stream's sha256:
`finetune` and the release refuse a stream that changed afterwards.

## 3. Write the classic rival

`span_rival(select, test)` returns the span-F1 of the rule a competent human would write (a regex plus a check),
tuned on `select`, scored on `test` exactly like the model (spans as sets, exact boundaries). It is the delivery
gate: a model that does not beat it by more than the spread between seeds is not delivered.

## 4. Build or fetch the data

Context corpora, catalogs and exams come from the scripts in `data/` (public sources pinned by commit or sha256;
see `data/README.md`). Register the exams that judge your specialist (their fingerprints) before building any
training data, and list them in `specialist.yaml` (`exams:`) and in the recipe (`exams:`): the `data` stage refuses a
run whose windows share a 64-byte block with an exam, and a release refuses a run whose data stage did not check every
exam of its specialist.

## 5. Run the smoke recipe

```bash
python -m pbtrain run specialists/invoice-ids/recipes/smoke-cpu.yaml
```

Minutes on a CPU, `mini64` shapes. It proves the pipeline, not the model: the leak checks, the rival, three seeds and
the control, the operating point, the gate and, when the gate passes, the export, the runtime equivalence, the report
and the release. When a model this small does not beat the rival, the run stops at `eval` with exit status 2 ("GATE
NOT PASSED"): a correct outcome.

## 6. Write the recipe and its pre-registered criteria

Copy the smoke recipe to `recipes/invoice-ids-0.1.0.yaml`, set the real sizes (start from the defaults of
`training/docs/recipes.md`: `nano`, 12,000 steps, batch 16, lr 3e-4) and write the `criteria:` block NOW, before
training: what the model must reach on the synthetic test and on each exam, with the date. The first `data` stage of
the version freezes their sha256; changing them afterwards fails every later stage, and a `data` stage run again under
the same version: new criteria need a new version.

## 7. Train three seeds (they inform)

```bash
python -m pbtrain run specialists/invoice-ids/recipes/invoice-ids-0.1.0.yaml --device auto --parallel 3
```

On a GPU, one job at a time (the lock refuses a second one) and `--parallel` sized from the VRAM table in
[gpu.md](gpu.md). The factory writes `$PUREBYTE_WORK/runs/invoice-ids/0.1.0/<stage>/done.json` for every stage and
the report in `report/report.md`. `pbtrain run` runs every stage, the immutable internal release included, and that
release records the exams recorded before it: to have the exams of step 8 in it, stop after `verify` (`--stages
data,baselines,finetune,eval,calibrate,export,verify`) and let step 10 write the report and the release. The released
seed's file is then `export/model.gguf`; the other members of an ensemble are exported by the release itself.

## 8. Run the exams and record them

Exams are real data the model never saw (see [evaluate.md](evaluate.md)). Produce each exam's metrics with its tool
and record them:

```bash
python -m pbtrain eval specialists/invoice-ids/recipes/invoice-ids-0.1.0.yaml --record my-exam --metrics m.json
```

`pbtrain eval TASK` prints PASS / FAIL / PENDING for every pre-registered criterion. A FAIL is a result: write it
down, do not re-tune on the exam. Records are kept, never overwritten: recording an exam again adds a record, and the
report shows how many each exam has.

## 9. Six seeds decide

A release needs six seeds and the exams read on the MAX seed as well as the mean: about one seed in six can end far
worse than the others. Change ONE thing per version, and keep a control. `eval/done.json` and the report label a run
with `decide: true` `DECIDES` only from six seeds; with three to five it is labeled `INFORMS`. The release checklist
([release.md](release.md)) asks for a deciding run.

## 10. Release

`python -m pbtrain release TASK --public --semver X.Y.Z`, once the exams are recorded: the report, the immutable
internal release and the public bundle (GGUF files, sha256, model card, catalog fragment). An internal release that
`pbtrain run` or `pbtrain release TASK` already wrote is checked and bundled. See [release.md](release.md).

## When the architecture must change

The model's blocks and heads are the public types of the runtime's engine (`ssm_v2`, `attention`, n-gram memory,
lookahead; `tag`, `choice`, `ordinal`, `exit`: pbtrain trains `tag` and `choice` heads, and the exporter also writes
`ordinal` and `exit` heads). A new type goes into the runtime repository FIRST
([purebyte-ai/purebyte](https://github.com/purebyte-ai/purebyte): spec, engine, NumPy reference), then here: the
pinned copy of the reference in `reference/` is refreshed from there, then the module with its `TYPE`, one line in
`pbtrain/model/registry.py`, its exporter in `pbtrain/export/pbformat.py`, and tests (the golden training fingerprints
of the untouched presets must not move; export -> reference -> same spans). Anything added inside a block must also be
mirrored in the Radeon backend ([gpu.md](gpu.md#the-radeon-backend)).
