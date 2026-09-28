# Recipes: the task file reference

A task file (`specialists/<name>/recipes/*.yaml`) is the whole recipe of one version of a specialist. The factory
refuses unknown keys, keys that would act on nothing, values of the wrong type or out of range (checked when the task is
loaded, before anything is built), and `data:` keys the vertical did not read. The task's DIGEST (its parsed content,
without `description`) is recorded by every stage: a task edited after `data` is run again from `data`, and the
pre-registered criteria of a version never change after its first `data` stage.

## Top level

| Key | Meaning |
|---|---|
| `name`, `version` | the specialist and the version of this recipe (lower-case name; semantic version); the work dir is `runs/<name>/<version>/` |
| `specialist` | the folder under `specialists/` whose plugin builds the data (defaults to `name`) |
| `description` | free text (left out of the digest) |
| `task_type` | `tag` (spans) |
| `output_schema` | `labels` (entity types), `per_byte: true`, `min_select_recall` (a collapse alarm, in (0, 1]), `bias_policy` (`conservative` or `f1`), `per_type_bias` (one bias per type, chosen on select; two labels or more) |
| `data` | the vertical's block: `format`, `path`, `n_train`, `n_eval`, `seq_len`, `split_policy`, `control`, and its own keys |
| `train` | the trainer (below) |
| `criteria` | pre-registered criteria ([evaluate.md](evaluate.md)); required when `train.decide` is true |
| `exams` | exams whose fingerprints must exist (the leak check refuses to run without them; with the specialist's `exams:` added) |
| `engine` | `{threads: N}`: the threads of the engine's `model_dump` in `verify` (with the NumPy reference, `verify` runs in the pbtrain process and records that the key did not apply) |
| `base_checkpoint` | a body to start from |

## `train:`

Required: `seeds`, `size`, `steps`, `lr`, `batch` (no silent defaults for the numbers that define a run). `seeds` is a
list of distinct whole numbers; numbers are numbers (YAML reads `3e-4` as text: write `0.0003`); `batch` is at most
`data.n_train`; an option declared without the one it needs (`real_frac` without `real_windows`, `guard_every` without
`guard_mode`, `lora_rank` without `adapter: lora`) is refused.

| Key | Default | Status | What it does |
|---|---|---|---|
| `size` | - | stable | a preset (below) |
| `mode` | causal | stable | `causal` or `bidirectional` (hydra); bidirectional did not help secrets |
| `steps`, `batch`, `lr` | - | stable | optimizer steps, windows per step, peak learning rate (WSD schedule) |
| `warmup` | 30 | stable | linear warmup steps |
| `wd` | 0.05 | stable | AdamW weight decay on matrices only |
| `clip` | 1.0 | stable | gradient norm clip |
| `lam` | 0.5 | stable | weight of the document head in the loss |
| `real_windows`, `real_frac`, `real_mode: neg` | off | stable | real negative windows appended to every batch |
| `real_mode: spans` | off | stable | hand-labeled real windows (secrets-code 1.0.0; the file format: [bring your own labeled data](new-specialist.md#bring-your-own-labeled-data)) |
| `real_mode: distill` | off | experimental | a teacher's probabilities |
| `span_weight` | 1 | experimental | weight (>= 1) of span bytes in the tag loss (extraction tasks) |
| `out_lr_mult` | 1 | stable | lr multiplier of every `out_proj` (secrets-bin 1.0.0: 0.25) |
| `beta2` | 0.95 | experimental | AdamW beta2 |
| `init_seed` | seed | experimental | init seed apart from the data order |
| `ngram_init`, `freeze_ngram` | off | experimental | an n-gram memory pretrained on the task's corpus, optionally frozen |
| `adapter`, `lora_rank` | none | experimental | frozen body + `heads` / `scales` / `lora` |
| `guard_*` | off | experimental | the tail guard (a proxy of false alarms on real windows during training) |
| `released_seed` | best on select | stable | force the released seed: an exception to the rule (never choose on an exam); recorded in done.json, the report and the MANIFEST with the seed `select` would have chosen, and a warning when it differs |
| `released_seed_reason` | none | stable | why `released_seed` overrides the rule (recorded with it) |
| `decide` | false | stable | this run is meant to decide: it needs >= 3 seeds and criteria, and is labeled `DECIDES` from six seeds (`INFORMS` with three to five) |

"Stable" means a released or recommended recipe uses the option; "experimental" means none does yet. The report lists
the experimental options a run uses.

Not implemented because they did not help on real evaluations (a task that declares them is refused): an EMA of the
weights, averaging the last steps (SWA), a lower ternary zeroing threshold, other poolings of the document head.

## Sizes

| Preset | Shape | Body params | Use |
|---|---|---|---|
| `nano` | 256 x 4 | 1.9 M | secrets-code |
| `tiny` | 256 x 8 | 3.8 M | measured level with `small` on secrets |
| `tiny_n18` | 256 x 8 + n-gram memory (orders 2-4, 2^18 buckets) | 3.8 M + 50 M tables | secrets-bin (28.8 MiB at 4 bits) |
| `nano_n15` | 256 x 4 + n-gram memory (orders 2-4, 2^15 buckets) | 1.9 M + 6.3 M tables | pii (4.5 MiB at 4 bits) |
| `nano_n`, `tiny_n` | + 2^16-bucket tables | | |
| `nano_a`, `nano_an` | one late attention layer | | extraction of several fields |
| `pico`, `small`, `medium`, `base` | 128 x 8 ... 768 x 16 | | |
| `base20`, `base50` (`_n18`) | 512 / 768 x 12 | 22 M / 50 M | base models for decisions (the factory trains `tag` tasks only today) |
| `mini64` | 64 x 2 | 0.1 M | CPU tests and smoke recipes only |

More capacity did not help when the labels are synthetic (larger presets raised the false alarms on real code).
The n-gram memory is the architecture change that helped most (binaries: false alarms divided by more than ten).

## The model

A byte embedding (+ the n-gram memory at a declared layer), `ssm_v2` blocks (Mamba-2 style, ternary projections,
paired log-spaced A / dt initialization), an optional attention block, a final RMSNorm and optional lookahead, then
two float heads: `tag` (BIOES per byte, k entity types) and `choice` (the window gate, max pooling). The exported file
declares all of it; the runtime's [spec/FORMAT.md](https://github.com/purebyte-ai/purebyte/blob/main/spec/FORMAT.md) is the normative description of the forward pass.

## The secrets generator (`data:` keys of `secrets_synthetic`)

| Key | Meaning |
|---|---|
| `version` | `v4` (the generator the released models were trained with) |
| `extra_lines` | blocks of up to N extra lines around the injected one, in half of the windows |
| `real_neighbors` | `true` / `no_pem`: half of the negative neighbors are real corpus lines |
| `nul_separator`, `bare_value`, `long_hex`, `binary_symbols` | binary mode |
| `web_negatives`, `shared_hosts`, `documented_examples`, `bodyless_pem` | negative families |
| `connection_config`, `short_passwords`, `ordinary_short_values`, `reinforced_forms`, `weak_passwords`, `no_equals_sign`, `non_assignment_weight`, `nested_weight`, `invented_prefixes` | family weights |
| `password_named_short`, `auth_header`, `url_short_password`, `unnamed_values`, `dotnet_resources` | families found missing on real labeled data |
| `mined_negatives` | `{path, fraction}`: real windows a released model flagged falsely, as negatives |

Every family at its default draws no random number: adding a key never changes the data of an existing recipe.
