# pii {{version}}

An AI Specialist: a byte-level model that finds personal data in any text and masks it, for **redaction by copy**. It
marks spans of one type, `PII`; the runtime's `redact` profile returns the input byte for byte with each span replaced
by a marker such as `[PII_1]` (the same value always gets the same marker). Nothing is generated, so nothing can be
invented; the only errors are data that leaks (a missed span) and text masked without need (a false span).

## Files

| File | Bytes | SHA-256 |
|---|---|---|
{{files}}

Ensemble policy: {{vote}}. Operating bias chosen on the selection split: {{operating_bias}}.

## What it masks

One type, `PII`: every span that its training data label as personal data. People (names, titles, gender, age,
nationality, occupation), contact details (e-mail addresses, phone numbers, street addresses, postcodes, cities,
countries), identifiers (identity, passport, license, tax, social security, customer, employee, account and device
numbers), finance (cards, IBAN, SWIFT/BIC, routing numbers), online (usernames, passwords, API keys, URLs, IP and MAC
addresses), dates and company names. It does not tell these kinds apart, check values, or guarantee that nothing leaks.

## How to use it

```bash
purebyte models pull pii
purebyte redact --model pii notes.txt > notes.redacted.txt
```

The redaction report lists every span (offsets, marker, confidence) and never holds a redacted value. The model reads
windows of 512 bytes with a stride of 384; inputs of any length are accepted, including very short ones.

## Training data

Documents of the TRAIN splits of four public datasets of synthetic personal data, rebuilt from pinned downloads and
checked against every evaluation set (the same text, a shared sentence or line, the same masked template, a shared
masked sentence); the release pool of pii 1.0.0 holds 387,391 of them, in 23 languages. No value was taken from a real
person, although a generated value can coincide with a real one. No test or validation split and nothing of the PII
masking benchmark was used for training. The benchmark chose neither the seed nor the operating point (both chosen on
the training pool's own selection split), but its scores were one input of the choice between recipe variants before
pii 1.0.0 was released (a tie under the pre-registered rule; the structured-forms variant shipped for its results on
code, records and logs).

Attribution. Contains information derived from: Ai4Privacy / Ai Suisse SA OpenPII-1M (CC BY 4.0) · NVIDIA Nemotron-PII
(CC BY 4.0) · Gretel.ai datasets (Apache 2.0). A model of the structured-forms variant of the recipe, such as pii
1.0.0, also contains information derived from: INE, www.ine.es (CC BY 4.0) · Tatoeba, tatoeba.org (CC BY 2.0 FR) · US
Census Bureau (public domain) · Faker (MIT) · credential formats of gitleaks and trufflehog.

## Evaluation

Every number comes with its conditions. Numbers that were not measured say TBD. This card is for pii {{version}}; the
figures below were measured on **pii 1.0.0**, the released example trained with `recipes/pii-1.0.0.yaml` on its release
pool, and they are not those of any other model. A model trained with this template's specialist, including a retraining
of the same recipe on a pool rebuilt with `--exam-overlap drop`, has its own figures (`pbtrain eval --record`, its
release's report): replace this table with them before publishing it.

| Measure | Value | Conditions |
|---|---|---|
| PII masking benchmark: mean F2 of the six tasks, ai4privacy-en F2, mean precision | 0.7689, 0.9569, 0.773 | `sentences` subset (150,022 sentences), character-level, label-agnostic (the leaderboard's metric), the released file at its default operating point, measured with the purebyte CLI |
| The same, mean and minimum of the recipe's six seeds | mean 0.7774, 0.9624, 0.765; minimum 0.7689, 0.9569, 0.759 | as above; the recipe's pre-registered criterion (above 0.6621 and 0.8426, precision at least 0.70): pass |
| The same, without the benchmark sentences that share a 64-byte block with the training data | 0.7686, 0.9568, 0.773 | as above, without 1,018 of the 150,022 sentences |
| Span-F1 on the training pool's own test documents, against the rules | 0.825 (six seeds: 0.816 to 0.830), rules 0.444 | the factory's gate: exact spans, 2,000 windows of 512 bytes |

## Limitations

- The training data are synthetic documents of four generators; real documents, other languages and other genres
  (protocol traces, legal texts, source code) are harder.
- The window head was trained on 512-byte windows: on a single short sentence it says "nothing here" more often.
- The model reads bytes left to right: a span's boundaries are decided without seeing much of what follows.
- Companies, occupations, dates and places are masked by design (the policy of its training data).
- The n-gram memory is stored at 4 bits, which changes the spans of a small share of windows.

## License

Weights: see [MODEL_LICENSE.md](https://github.com/purebyte-ai/purebyte/blob/main/MODEL_LICENSE.md). Code: Apache-2.0.
