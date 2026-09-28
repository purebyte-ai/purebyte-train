# Evaluate honestly

The synthetic test of the factory tells you whether the model learned the generator. It does not tell you whether it
works on real data: one released model scored 0.98 recall on its synthetic battery and 0.31 on a labeled public
dataset of real code. Only exams on real data decide.

## The rules

1. **Pre-register.** Criteria, prediction and the date go into the task's `criteria:` block before training. The first
   `data` stage of a version freezes their hash (`criteria.frozen.json`, next to the stages): a later stage refuses
   changed criteria, and so does a `data` stage run again under the same version. New criteria need a new version.
   Read the result exactly as written.
2. **One variable per version**, and a control. When a version changes two things, it is not an attribution result.
3. **Three seeds inform, six decide.** About one training seed in six can end with many times the false alarms of the
   others, and warmup, clipping or quantization tweaks only move the problem to another seed. Criteria read the MAX
   seed (for example "max <= 2 x median"; a max over the failure line fails whatever the mean says). A run with
   `decide: true` is labeled `DECIDES` from six seeds, `INFORMS` with three to five.
4. **Choose on DEV, look at TEST once.** Operating points, seeds and ensembles are chosen on `select` or on a DEV
   split. An ensemble chosen on the exam once scored 0 false alarms in sample and 148 out of sample. (The released
   `secrets-code` 1.0.0 breaks this rule: its seed was chosen on an exam, with TEST in view;
   [release.md](release.md#recorded-exceptions-of-the-released-specialists).)
5. **Keep the exams out of training, and prove it.** Register every exam's fingerprints (`data/fingerprint_exams.py`);
   the `data` stage checks train, select, real streams and guard windows against all of them and fails on one shared
   64-byte block with an exam of role `test` ([what the leak checks check](#what-the-leak-checks-check)). Report CLEAN
   numbers.
6. **Re-measure the baseline when the instrument or the machine changes.** A comparison against a number measured
   elsewhere measures the difference between machines.
7. **A failing instrument must be loud.** Zero windows read, zero files scored, a flat bias sweep of the released
   seed, a runtime comparison without a single span: errors, never results (an out-of-memory once produced a "0 false
   alarms"). The flat sweeps of the other seeds are recorded and shown.
8. **Numbers travel with their conditions**: dataset and split, files and sizes, operating point, seeds, ensemble,
   hardware for speeds. Absolute speeds only from an idle machine.

## What the factory measures

| Stage | Measures | Blocks the release when |
|---|---|---|
| `data` | overlap between splits; exam fingerprints; unread keys; the vertical's guards; freezes the criteria | any leak, typo or failed guard; criteria other than the ones frozen for this version |
| `baselines` | the classic rival's span-F1 (tuned on select) | the vertical has no rival |
| `finetune` | per seed: operating point on select, span-F1 on test; the shuffled-spans control; where and with what it trained | a stream or a set of exams changed since `data` |
| `eval` | the delivery gate: control <= 0.10, recall floor of the released seed, no flat bias sweep of the released seed, mean model - rival > spread between seeds; the pre-registered criteria are recorded | any of the four fails (the done.json says which; a FAIL criterion is recorded and shown, and the release checklist asks for none; the code does not block on it) |
| `calibrate` | the document head's calibration error on select, next to the estimator's noise floor | - |
| `verify` | the runtime's engine (or the pinned NumPy reference, `reference/`) gives the SAME spans as Python, on 40 test windows: half where Python finds a span, the rest where it finds none | one window differs, or no compared window holds a span |
| `report` | every criterion, exams included, with the results recorded so far | - |
| `release` | - | the gate was not passed, too few seeds for the ensemble, an exam of the specialist was not checked, a stream or the exams changed since `data`, the version exists |

## What the leak checks check

Precisely what the `data` stage enforces and measures (`training/pbtrain/data/leakage.py`), and why:

- **Between the splits** (train-select, train-test, select-test), what the vertical's `overlap_guard` enforces:
  - `chunks` (verticals that cut windows from real bytes, such as `secrets_synthetic`): no 120-byte chunk of the first
    300 windows of one split may appear among those of the other, zero tolerance. The chunks are taken at the fixed
    offsets 0, 120, 240... of each window (the last whole chunk included), and chunks made of one repeated byte value
    are skipped. So the check sees two copies of the same bytes only when their offsets inside the windows differ by a
    multiple of 120. It is a cheap guard against a vertical that breaks the construction (separate byte ranges of the
    corpus for each split), not a proof;
  - `windows` (the `pii` verticals): whole-window hashes of every window; at most 0.1 % of the second split may be
    identical to a window of the first. A generator of short records can repeat a window by chance, and a public
    dataset can hold one document under two ids; a generator with no variety fails. The pii vertical declares this
    mode although 4 % of its release mix are code windows cut from the corpus, from disjoint ranges.
  - measured, for every vertical, not enforced: the windows of each split that share at least one 64-byte block with
    another split (the content-anchored fingerprints of the exam check, every window, any offset). Content that repeats
    legitimately in disjoint parts of a corpus (license headers, boilerplate, the generator's templates) makes this
    non-zero, so it is recorded in `data/done.json` (`split_overlap.measured`) and shown in the report.
- **Against the exams**: every registered exam's fingerprints against every window of train and select, of the
  real-window stream and of the guard windows. One shared 64-byte block with an exam of role `test` fails the stage.
  Exams of role `dev` are development sets (the labeled CredData DEV stream of secrets-code comes from one): their
  hits are counted per set and shown in the report, not enforced; the part of a DEV set used to choose seeds or biases
  must be fingerprinted as its own `test` set. The test split is not compared with the exams (it trains nothing).
- **Which exams must exist**: the ones in the task's `exams:`, plus its specialist's `exams:` when the task lists any
  (a missing fingerprint file is then an error). Without them the check runs on the registered exams and records a
  warning; the release refuses a run that did not check every exam of its specialist. An exam that a criterion reads
  and `exams:` does not list is recorded with a warning saying whether its content was checked.
- **Afterwards**: the data stage records the sha256 of the streams and of every registered fingerprint set;
  `finetune` and `release` refuse a stream or a set of exams that changed since.

## The exams of the secrets specialists

| Exam | What it is | Metric | Tool |
|---|---|---|---|
| `creddata-test` | the TEST half (by repository) of the public CredData dataset | per-line precision, recall, F1 with a 95 % CI by repository | the runtime's [public scorer](https://github.com/purebyte-ai/purebyte/tree/main/benchmarks/creddata) (pinned copy: `benchmarks/creddata/score.py`); its `--json` output uses its own names, so the metrics file of the criteria (`f1_ensemble`, `precision_ensemble`) is written by hand from it (a converter is TBD) |
| `code-fp-a/b/c` | public repositories with no known credential, pinned by commit | findings per member and for the ensemble of a release (false alarms) | `specialists/secrets-code/eval/count_findings.py` (`findings_ensemble`, `findings_mean_seed`, `findings_max_seed`, `findings_median_seed`) |
| `bin-fp-a` | real shared objects with no credential | findings (false alarms) | `count_findings.py` counts the files of one release; a `secrets-bin` release holds one model, so the per-seed metrics of its criteria come from tooling that is not in this repository yet (TBD) |
| `bin-fp-b`, `bin-fp-c` | ELF + PE collections, unseen binary families | findings (false alarms); the `secrets-bin` criteria split bin-fp-b into PE and ELF files | as `bin-fp-a`; the PE / ELF split is TBD too |

The `secrets-bin` recipe also names two private exams, `bin-pyext` and `bin-probe`, declared nowhere in `data/exams/`:
no leak check covers them, and nobody outside the maintainers can run them
([specialists/secrets-bin](../../specialists/secrets-bin/README.md#evaluate)).

Sources, pins and licenses: `data/exams/*.yaml`. Record each result with `pbtrain eval TASK --record EXAM --metrics
FILE`; the file is a JSON object whose keys are the metric names the criteria use (numbers). Records are append-only
(`exams/<exam>.jsonl` in the run's folder, with the time, the metrics file's sha256 and the run's model files at that
moment): recording an exam again adds a record and says so, the criteria read the latest one, and the report and the
release MANIFEST show how many records each exam has.

## Calibration

For span models the operating point is the Viterbi bias chosen on `select` (one value per type with
`per_type_bias`); it travels in the GGUF and callers can override it. The document head gates the spans by its argmax,
which a temperature cannot change, and the file carries no temperature (1.0). The `calibrate` stage measures the head's
expected calibration error on `select` and records it next to the estimator's noise floor
(`pbtrain/eval/calibration.py`: an ECE under the floor's p95 is not a measurement), for the probabilities the runtime
reports.
