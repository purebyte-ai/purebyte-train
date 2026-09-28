# secrets-code {{version}}

An AI Specialist that finds credentials (API keys and tokens, passwords, private keys, connection strings with passwords) in source code and
configuration files. Byte-level ternary model (1.9 M parameters) run by the PureByte engine on CPU. The first file is
the model; the others are the optional members of an ensemble of three independently trained models ({{vote}}).

## Files

| File | Bytes | sha256 |
|---|---|---|
{{files}}

## Use

```bash
purebyte models pull secrets-code
purebyte scan path/to/repo
purebyte models pull secrets-code-ensemble      # optional: the ensemble, three times the work
purebyte scan --model secrets-code-ensemble path/to/repo
```

Operating point (Viterbi bias) of the model: {{operating_bias}}. Every file stores its own (the members of the
ensemble keep theirs); `--bias` moves it (higher: more recall, more false alarms).

## Evaluation

Every number with its dataset, split, size, operating point and ensemble: the model card in the runtime repository,
[models/secrets-code.md](https://github.com/purebyte-ai/purebyte/blob/main/models/secrets-code.md), and
[benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md#secrets-code-on-creddata).
What was not measured says TBD there.

## Training data

- Generated credential lines injected into windows of real public code: the code corpus that `data/build_code_corpus.py`
  of purebyte-train builds from 114 permissively licensed repositories pinned by commit (each file keeps its license;
  the build manifest records them). secrets-code 1.0.0 was trained with an earlier corpus of public repositories, which
  cannot be rebuilt.
- Credential formats: the rule catalog of gitleaks (MIT), and detector names and example strings of trufflehog
  (AGPL-3.0), used to sample generated values and names; in a small share of draws (0.035 %) one example string of
  trufflehog's tests is used verbatim.
- Labelled lines of CredData (Samsung/CredData, Apache-2.0; its files keep their upstream licenses): 117 repositories of
  its DEV half, half of the credential values replaced by generated ones; never TEST. Five of those repositories are also
  repositories of the code-fp false-alarm exams (see the training repository's data/README.md).

## Limitations

- secrets-code 1.0.0 misses about one labelled credential in four on CredData TEST (recall 0.729 at its operating
  point); another version has its own figures. Identifiers (UUIDs, salts, nonces) count only under an unequivocal
  credential name, and values with no name next to them are missed more often than the rest.
- Trained on generated credentials injected into real code, plus labelled lines of CredData repositories of the DEV
  half (never TEST); expect lower figures on data labelled differently, and much lower than on generated data.
- Never verify a finding by calling the provider's API.

## License

Weights: see [MODEL_LICENSE.md](https://github.com/purebyte-ai/purebyte/blob/main/MODEL_LICENSE.md). Code: Apache-2.0.
