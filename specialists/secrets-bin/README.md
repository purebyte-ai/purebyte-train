# secrets-bin

An AI Specialist that finds credentials embedded in compiled binaries (ELF and PE executables and shared objects, and other formats with
string tables). The same generator as [secrets-code](../secrets-code/README.md) in binary mode, on a corpus of real
binaries, with an n-gram memory (`tiny_n18`: 256 x 8 `ssm_v2` blocks + hashed tables of byte 2-, 3- and 4-grams,
2^18 buckets each; 28.8 MiB at 4 bits) and real negative windows in every batch: Windows executables and libraries,
files of other binary formats and compiled Python extension modules.

**What a public user can and cannot reproduce.** The negative windows come from binaries of the machine the model was
trained on, many of which cannot be redistributed, and their builder is not published (TBD): the released recipe cannot
be trained from this repository alone ([Train](#train)). Two of its pre-registered criteria are measured on private
exams ([Evaluate](#evaluate)), and the released weights were trained with an earlier corpus that overlapped two of the
public exams (below). It was released with one criterion failed: the probe recall, on two of its six seeds.

| File | What it is |
|---|---|
| `specialist.yaml` | uses the secrets-code plugin; export profile `secrets-binary`, n-gram tables at 4 bits, exams |
| `recipes/secrets-bin-1.0.0.yaml` | the recipe of the released 1.0.0 (six seeds, the best on `select` released, its pre-registered criteria and their outcome) |
| `recipes/secrets-bin-1.0.0-rc1.yaml` | the earlier candidate, with negatives from Windows PE files only (superseded) |

## Data

- Context: `$PUREBYTE_DATA/corpora/binary-context.bin` (`data/build_binary_corpus.py`: public packages pinned by version
  and sha256, the exam binaries excluded by identity and by content). The released weights were trained with an earlier
  corpus, assembled from the system files of one machine before `data/` existed: its bytes overlapped 8 of the 25 Linux
  files scanned for `bin-fp-b` and 2 files of `bin-fp-c` (the runtime's benchmarks/RESULTS.md reports the counts
  without those files), and it cannot be rebuilt; a corpus rebuilt here excludes every exam file.
- Binary mode of the generator: 70 % of the injected blocks between NUL bytes like a C string table
  (`nul_separator`), sometimes the value alone when it is decidable by itself (`bare_value`), long hex digests as
  negatives (`long_hex`).
- Real negatives (`real_mode: neg`, 45 % of the batch size in 1.0.0): 78,283 windows of 512 bytes from Windows
  executables and libraries (they cut false alarms on PE files by about fifty times), from files of other formats
  (JAR, APK, PYC, WebAssembly, Mach-O, Node addons, firmware, ar archives, object files, ELF and PE) and from compiled
  Python extension modules of 63 packages; no exam file. Many of those binaries cannot be redistributed: the builder of
  `$PUREBYTE_DATA/streams/binary-negatives.npz` is not in this repository yet (TBD).

## Train

```bash
python -m pbtrain run specialists/secrets-bin/recipes/secrets-bin-1.0.0.yaml --device auto --parallel 3
```

This command stops at its `data` stage on any machine but the maintainers': it reads
`$PUREBYTE_DATA/streams/binary-negatives.npz`, whose builder is not published yet (TBD, above). Without that stream the
recipe can be run with `train.real_windows`, `real_frac` and `real_mode` removed (a new recipe version, with its own
criteria: the negatives are what cut the false alarms, so expect many more). The superseded
`recipes/secrets-bin-1.0.0-rc1.yaml` reads `streams/pe-negatives.npz`, which nothing builds either (its comment says
"data/ scripts": there is no such script).

`tiny_n18` needs ~3.9 GB of VRAM per job: at most three in parallel on 16 GB ([gpu.md](../../training/docs/gpu.md)).

## Evaluate

The false alarms on real binaries are the hard part, and they have a TAIL: about one seed in six can end with many
times the false alarms of the others. Six seeds, and the criteria read the max seed as well as the mean. secrets-bin
is released as ONE model (`release: ensemble: 1` in `specialist.yaml`): a jury of three models did not generalize out
of sample. The numbers of 1.0.0, with their conditions (and the six seeds of its recipe, mean and max), are in the
runtime's [benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md#secrets-bin) and its [model card](https://github.com/purebyte-ai/purebyte/blob/main/models/secrets-bin.md).

What of that can be checked from here:

- `bin-fp-a`, `bin-fp-b` and `bin-fp-c` are declared in [data/exams/binary-fp.yaml](../../data/exams/binary-fp.yaml)
  (the three exams of `specialist.yaml`, fingerprinted for the leak check). 15 files of `bin-fp-b` and 46 of `bin-fp-c`
  have no public download: a public user scans and fingerprints those sets without them.
- Two criteria of the recipe use **private exams** that are not declared anywhere in `data/exams/`, so neither the data
  scripts nor the factory's leak check cover them, and nobody outside the maintainers can run them: `bin-pyext`, 60
  compiled extension modules of 18 Python packages built locally and set apart before training, and `bin-probe`, 2,000
  windows cut from binaries of the maintainers' internal corpus with 400 planted credentials. The published probe
  recall (360 of 360 decidable credentials found by the released file) rests on the second. That probe is synthetic
  and in distribution: its credentials, their names and its traps come from the training generator, in the layouts of
  its binary mode, so it tells whether the model learned its generator, not how it does on real leaks. Its
  pre-registered criterion (recall of the worst seed at least 0.99) FAILED: two of the six seeds found 356 of the 360;
  the model was released with that failure recorded. A public probe built from redistributable binaries is TBD.
- The per-seed metrics of the criteria (`findings_mean_seed` on six seeds, `pe_findings_mean_seed`,
  `elf_findings_mean_seed`, `recall_min_seed`) come from tooling that is not in this repository yet (TBD):
  `count_findings.py` counts the files of one release, and a release of this specialist holds one model.
- `bin-fp-c` was corrected on 2026-09-25: two npm members now pin the files npm publishes instead of the
  application-bundled builds the published figures were measured on; no published finding was in them, and the
  released model reports nothing on the published files either ([data/README.md](../../data/README.md#known-provenance-gaps)).
