# secrets-bin {{version}}

An AI Specialist that finds credentials embedded in compiled binaries. Byte-level ternary model with an n-gram memory (tables at 4 bits), run
by the PureByte engine on CPU; one model, no ensemble.

## Files

| File | Bytes | sha256 |
|---|---|---|
{{files}}

## Use

```bash
purebyte models pull secrets-bin
purebyte scan --model secrets-bin dist/
```

Operating point (Viterbi bias) {{operating_bias}}, stored in the file; `--bias` moves it (higher: more recall, more
false alarms).

## Evaluation

Every number with its binaries, sizes and conditions, including the false alarms of every seed of the recipe and
their maximum: the model card in the runtime repository,
[models/secrets-bin.md](https://github.com/purebyte-ai/purebyte/blob/main/models/secrets-bin.md), and
[benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md#secrets-bin). There is no
public, labelled benchmark of credentials inside binaries yet; the exams are partly built from binaries that cannot be
redistributed, and the recall figures of secrets-bin 1.0.0 come from a private probe (credentials planted in binaries
of the maintainers' internal corpus) that nobody else can run.

## Training data

- Generated credential lines injected into windows of real binaries: the binary corpus that `data/build_binary_corpus.py`
  of purebyte-train builds from public packages pinned by version and sha256 (licenses recorded per package).
  secrets-bin 1.0.0 was trained with an earlier corpus, from the system files of one machine, which cannot be rebuilt.
- Real negative windows (files that hold no credential) from Windows executables and libraries, other binary formats
  and compiled Python extension modules of the machine the model was trained on; many cannot be redistributed, and they
  are not published.
- Credential formats: the rule catalog of gitleaks (MIT), and detector names and example strings of trufflehog
  (AGPL-3.0), as for secrets-code.

## Limitations

- Credentials split across non-contiguous bytes, or inside compressed or encrypted sections, are not found; zip, gzip
  and tar containers are opened by the runtime, other compression is not.
- Trained on generated credentials injected into real binaries, with real binaries as negatives.

## License

Weights: see [MODEL_LICENSE.md](https://github.com/purebyte-ai/purebyte/blob/main/MODEL_LICENSE.md). Code: Apache-2.0.
