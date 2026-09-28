# data/: download and build the training and evaluation data

This folder holds scripts and pinned manifests. It never holds data: everything they download or build goes to
`$PUREBYTE_DATA` (default `~/.purebyte/data`), outside git. Every input is public and pinned (git commit, package
version and sha256, dataset revision), every license is recorded, and a written guarantee, enforced in code, keeps the
evaluation sets out of the training data. Where a set cannot be fetched by everyone, the scripts say so and the
guarantee says how far it reaches ([Limits](#limits-stated-plainly)).

## Quick start

```bash
python3 -m venv .venv && . .venv/bin/activate  # a virtual environment (Windows: py -m venv .venv, then .venv\Scripts\Activate.ps1)
python -m pip install numpy pyyaml           # plus git on PATH; Python 3.10 or newer (CI runs 3.10 and 3.12)
export PUREBYTE_DATA=~/.purebyte/data         # optional (this is the default)

# 0. Only for the CPU smoke recipes: made-up stand-ins, seconds, offline (never mixed into real builds)
python data/build_smoke_data.py

# 1. Evaluation sets first: they are what the builds exclude
python data/fetch_exams.py                    # public evaluation sets; a failed set is reported, the others go on
python data/fetch_creddata.py                 # CredData (Linux, macOS or WSL; see its section)
python data/fingerprint_exams.py              # exams/fingerprints/<set>.npz for every set it can read

# 2. Training material
python data/build_code_corpus.py              # corpora/code-context.bin
python data/build_binary_corpus.py            # corpora/binary-context.bin
python data/fetch_pii.py all                  # pii/lists/*.tsv and pii/public_train/*.jsonl
python data/build_pii_pool.py --exam-overlap drop          # pii/public_train_xl/ (needs pyarrow; see below)
python data/build_secret_catalogs.py          # catalogs/gitleaks.json and catalogs/trufflehog.json
python data/build_creddata_stream.py --exam-overlap drop   # streams/creddata-dev-spans.npz (secrets-code; see below)

# 3. Check what is on disk against the manifests at any time
python data/build_code_corpus.py --verify     # (every script takes --verify and --dry-run)
python -m pytest data/tests -q                # offline tests
```

What to expect on a machine that is not the maintainers':

* `fetch_exams.py` downloads about 1.5 GB (0.7 GB of it the Android NDK archive of `bin-fp-c`). The Ubuntu packages of
  `bin-fp-b` are zstd-compressed: on Python 3.13 or older put the `zstd` command on PATH, or that set fails (and is
  named in the summary). 15 files of `bin-fp-b` and 46 of `bin-fp-c` (Windows system files, locally compiled or
  application-bundled files) have no public download: they come only from a local copy (`--from-dir`), and without one
  those two sets are fetched and fingerprinted without them.
* `fingerprint_exams.py` downloads nothing and fingerprints every set whose files are on disk. A set with files
  missing that have no public download is written anyway and marked incomplete; a set that cannot be read at all
  (not fetched, CredData on Windows) is skipped. The summary names both, and the exit status is 1 when a set was
  skipped.
* Every build refuses to run while a declared set has no fingerprint file. `--allow-missing-exams` builds anyway:
  the sets without fingerprints are **not** excluded from what it writes, and its manifest lists them. Use it knowingly
  (for example to build the PII pool with the PII sets only, [below](#the-pii-pool)); the training factory still checks
  every window against every fingerprint file present, and requires those of the exams a recipe declares.

`--dry-run` (print the plan, download nothing) and `--verify` (re-hash what is on disk against the recorded manifests;
the corpus builders also re-check the exclusion against the fingerprints on disk, and the CredData stream re-runs its
hard check) are accepted by every script. `--allow-missing-exams` is accepted by the scripts that check their output
against the exam fingerprints (`build_code_corpus.py`, `build_binary_corpus.py`, `build_pii_pool.py`,
`build_creddata_stream.py`, `fetch_pii.py`), and `--no-fetch` (fail instead of downloading a missing input) by the builds
that fetch their own inputs (the same minus `fetch_pii.py`); `python data/fetch.py` downloads every source up front. A
script never accepts an option it would ignore. Scripts run at idle priority with two numeric threads.

## What each script builds

| Script | Reads | Writes (under `$PUREBYTE_DATA`) | Size | Downloads |
|---|---|---|---|---|
| `build_code_corpus.py` | `sources/code-context.yaml` (114 repositories) | `corpora/code-context.bin` + `.manifest.json` + `.index.tsv.gz` | 300 MiB | a few GB of git objects (files up to 512 KiB only) |
| `build_binary_corpus.py` | `sources/binary-context.yaml` (280 artifacts) | `corpora/binary-context.bin` + `.manifest.json` + `.index.tsv.gz` | 300 MiB | 1.1 GiB (+ a member cache of up to a few GB, deletable) |
| `fetch_pii.py lists` | `sources/pii.yaml` (13 files) | `pii/lists/*.tsv`, `LICENSES.md`, `SHA256.json` | 27 MB | 110 MB |
| `fetch_pii.py train` | `sources/pii.yaml` (4 datasets, train splits) | `pii/public_train/*.jsonl`, `MANIFEST.json` | 37 MB | ~0.2 GB of API responses and byte ranges |
| `build_pii_pool.py` | `sources/pii-pool.yaml` (4 TRAIN files), `sources/pii.yaml`, the PII evaluation sets | `pii/public_train_xl/*.jsonl`, `pii/public_train_clean/`, `MANIFEST.json`, `LEAKAGE.json` | 330 MB | 3.9 GB (the OpenPII-1M TRAIN file is 3.7 GB) |
| `build_secret_catalogs.py` | `sources/secret-catalogs.yaml` | `catalogs/gitleaks.json`, `catalogs/trufflehog.json`, `catalogs.manifest.json` | < 1 MB | ~50 MB of git objects |
| `fetch_exams.py` | `exams/*.yaml` | `exams/<set>/...` + `exams/<set>/manifest.json` | ~0.5 GB | ~1.5 GB (0.7 GB is the Android NDK archive of `bin-fp-c`) |
| `fetch_creddata.py` | `exams/creddata.yaml` | `exams/creddata/CredData/`, `split.json`, `manifest.json` | ~1.2 GB | the labeled files of 337 repositories at pinned commits (~1 GB) |
| `build_creddata_stream.py` | CredData (fetched if missing), the catalogs | `streams/creddata-dev-spans.npz` + `.manifest.json` | 24 MB | CredData, when missing |
| `fingerprint_exams.py` | the fetched sets | `exams/fingerprints/<set>.npz` | ~0.5 GB | none |
| `build_smoke_data.py` | nothing (generated) | `corpora/smoke-context.bin`, and made-up `catalogs/`, `pii/lists/`, `pii/public_train/` where empty; `smoke-standins.json` | 1.3 MB | none |
| `fetch.py` | `sources/*.yaml` | `cache/` only | | everything the builds need |

The training package reads the paths in the third column; they are fixed.

Prerequisites: Python 3.10+ (the imported `training/` package needs it; CI runs the data tests on 3.10 and 3.12), NumPy, PyYAML
and git; nothing else is imported. Three exceptions: `build_pii_pool.py` reads three Parquet files with pyarrow; the
Ubuntu packages of `bin-fp-b` are zstd-compressed (Python 3.14+, or the `zstd` command on PATH); and CredData's own
obfuscation script needs its requirement `pybase62` and a POSIX system (Linux, macOS or WSL). The data the secrets
generator makes depend on the Python version in one way, 3.13 or older against 3.14
([below](#the-creddata-dev-stream), condition 1): the CredData stream is bit for bit the released one only with
Python 3.14.

**Smoke stand-ins.** `build_smoke_data.py` writes what the two CPU smoke recipes read, generated from random
syllables and digits (`pbdata/smoke.py`, the generators of the test suites): a code-like corpus, two made-up credential
formats, made-up PII lists and look-alike labeled documents. It never replaces real data (a folder that holds
catalogs, lists or documents is kept) and deletes nothing. `smoke-standins.json` records every stand-in with its sha256;
the secrets-code and pii plugins refuse a stand-in among the files a build reads unless the recipe's corpus is
`corpora/smoke-context.bin`, and `build_creddata_stream.py` refuses stand-in catalogs. The real builds overwrite the
stand-ins (`build_secret_catalogs.py`, `fetch_pii.py lists`); `pii/public_train/lookalike.jsonl` is the one they do not,
so `fetch_pii.py train` names it, and it is removed by hand before a real recipe reads that folder.

## The exclusion guarantee

**Nothing from an evaluation set enters training data.** It holds on two levels, and every build enforces both before
it writes a byte; the build manifest records how.

### By identity

Every entry of every manifest is reduced to identity keys: the normalized repository URL (any commit), package
name + version (Debian, PyPI, Maven, NuGet, npm, Go and others), dataset + split (Hugging Face), download URL, and the
sha256 of every pinned artifact and listed file. CredData's 337 upstream repositories and the upstream splits a
benchmark was built from are keys of their exam entries too. If any key of `data/sources/*.yaml` appears in
`data/exams/*.yaml`, every build stops (`pbdata/identity.py`; `tests/test_identity.py` runs the check offline on the
real manifests). While reading archives, the binary build also refuses any member whose sha256 is that of an
evaluation file.

### By content

`fingerprint_exams.py` fingerprints every evaluation set with the one algorithm the training package uses
(`training/pbtrain/data/fingerprint.py`, imported, never re-implemented): 64-bit hashes of 64-byte blocks at
content-defined anchors. A shared region of 512 bytes is missed with probability below 0.1 %, of 1 KB below 1e-6.
Each `.npz` records its set, its role (`test` or `dev`) and the sha256 of the manifest it was made from.

Every build then:

1. loads every `exams/fingerprints/*.npz` and **fails** if a set declared in `data/exams/*.yaml` has no fingerprint
   file, or has one made from another version of its manifest. `--allow-missing-exams` builds anyway and writes the
   missing sets into the manifest as a warning;
2. checks each candidate against each set, and removes what shares a block:
   * **text** (code corpus, PII lists and documents): the whole lines around each shared block (plus 128 bytes on
     each side) are cut out and the result is checked again; a file that loses more than half of its bytes, or keeps
     fewer than 64, is dropped. A list line or a PII document that shares a block is dropped;
   * **binaries**: a binary is never spliced. Each file contributes one contiguous span (at most 256 KiB), chosen
     inside the part of the file that shares no block with any set; a file without such a part of at least 4 KiB is
     dropped whole;
3. verifies the finished corpus as a whole (a junction between two files could in theory form a shared block),
   removes any piece that touches a shared block and writes it again, until it is clean;
4. records per evaluation set how many files were dropped and trimmed and how many bytes were removed.

After a build, the corpus shares no fingerprint block with any evaluation set, which is also what the training
factory checks on every window it builds. `--verify` re-runs the whole-corpus check against the fingerprints on disk
and flags a corpus built before an evaluation set changed.

Why binaries are not dropped whole by default: common structure is shared by almost every binary built by the same
toolchains (section-name tables, runtime stubs, standard symbol names). On a sample of 250 Linux system ELF files that
are not evaluation files, every one shared at least 51 blocks with the binary evaluation sets, and so did every one of
40 Windows DLLs and executables from a Python distribution; dropping every file that shares anything would leave no
binary corpus. The clean-span policy keeps the guarantee (no shared block reaches the corpus) and keeps the files.
`build_binary_corpus.py --binary-policy drop-file` applies the stricter rule when it is wanted.

`role: dev` sets (code-fp-dev, creddata-dev, pii-bench-dev) are excluded from the corpora like test sets. They are
development sets: whether training streams may use them is the training factory's decision (their fingerprints carry
the role).

### Limits, stated plainly

* **Short fragments.** A fragment shorter than 64 bytes holds no block, and one of a few hundred bytes may hold no
  anchor (a shared 512-byte region is missed with probability below 0.1 %), so short fragments can escape any block
  fingerprint; the margin around each cut makes surviving fragments rarer, not impossible.
* **Files only the maintainers have.** 15 files of `bin-fp-b` and 46 of `bin-fp-c` (Windows system files and driver
  firmware, locally compiled `.pyc` and objects, application-bundled Node.js addons and WebAssembly modules, Android
  emulator skins, JNA class files) have no public download. They are excluded by identity (sha256) everywhere, but by
  content only where their fingerprints were built from a local copy (`--from-dir`). Without one, `fingerprint_exams.py`
  writes those two sets from their other files and marks them incomplete, every build repeats the warning, and the
  training factory cannot see that content: a corpus built elsewhere may hold fragments of those files.
* **Sets a user cannot fetch.** CredData is fetched on Linux, macOS or WSL only; a set that failed to download has no
  fingerprint file. Builds then stop, or run with `--allow-missing-exams` and do not exclude it (recorded).
* **Evaluation sets outside this registry are excluded by nothing.** The secrets-bin recipe also names two private
  exams, `bin-pyext` (60 locally built Python extension modules) and `bin-probe` (credentials planted in binaries of the
  maintainers' internal corpus); neither is declared here, so no build or leak check covers them
  ([specialists/secrets-bin/README.md](../specialists/secrets-bin/README.md)).
* **Identity is checked between the source manifests and the exams, not between exams.** Five repositories of the
  code-fp sets are also CredData DEV repositories, and the secrets-code stream is built from CredData DEV:
  [Overlap with the code-fp exams](#overlap-with-the-code-fp-exams) says what that means and how
  `--exam-overlap drop` handles it.

## Manifest schema

Two kinds of YAML file, validated by `pbdata/manifest.py` (unknown keys are refused):

```yaml
# data/sources/<name>.yaml                          # data/exams/<name>.yaml
schema: 1                                           schema: 1
kind: source                                        kind: exam
name: code-context                                  name: code-fp
output: corpora/code-context.bin                    sets:
builder: build_code_corpus.py                         - name: code-fp-a          # -> exams/fingerprints/code-fp-a.npz
build: {...}              # builder parameters          role: test               # test | dev
entries: [<entry>, ...]                                 fingerprint: files       # files | jsonl-text | creddata
                                                        entries: [<entry>, ...]
```

An entry:

| key | meaning |
|---|---|
| `id` | unique in its manifest (or set): `[a-z0-9][a-z0-9._+-]*` |
| `type` | `git`, `url`, `deb`, `pypi`, `hf` or `local` (evaluation only: files without a public download) |
| `pin` | `git`: `{commit: <40 hex>}`; `url`: `{sha256, size?}` or `{verify: members}` when the container's digest is not published; `deb` / `pypi`: `{version, sha256, size?}`; `hf`: `{revision: <40 hex>}` |
| `url`, `mirrors` | https only; mirrors are tried in order, and every copy must match the pin |
| `package`, `arch`, `filename`, `dataset`, `splits` | what the artifact is (identity keys); `url` entries may carry `package: {ecosystem, name, version}` |
| `licence` | SPDX expression (`LicenseRef-*` when SPDX has no identifier) |
| `take` | what is used: `include` / `exclude` / `members` (path patterns: `*`, `**`, `?`, `[...]`; a pattern without `/` matches the file name), `nested` (archives inside the archive), `self` (the artifact itself), `max_file_bytes`, `blob_limit` |
| `files` | the exact files used, each `{member or path or name, sha256, size}` (evaluation sets list every file) |
| `group` | forces the corpus group of every member (firmware images have no magic number) |
| `identities` | extra identity keys `namespace:key`, e.g. `hf:dataset:split`, `repo:<url>` |
| `sampling`, `repos`, `partition`, `provenance`, `description`, `notes` | documentation and sampler parameters |

## Sources (training material)

**code-context** (`sources/code-context.yaml`): 114 permissively licensed repositories (MIT, BSD, Apache-2.0, ISC,
Zlib, PostgreSQL, SQLite's public-domain blessing, Artistic-2.0), pinned at the default-branch heads of 2026-09-23, none
of them an evaluation repository or a CredData repository. Files are read from git objects (exact committed bytes; no checkout, no
line-ending conversion); vendored directories are skipped; identical files are taken once. Twenty groups with declared
shares: Python, JavaScript/TypeScript, Java/Kotlin/Scala/Gradle, Go/Rust, Ruby/PHP, C/C++, C#, YAML/CI, configuration
(`.properties`, `.toml`, `.ini`, `.cfg`, `.conf`, `.env`), shell, infrastructure (Terraform, Dockerfiles), JSON/XML, SQL,
PowerShell, Swift, Lua, Perl, Dart, Elixir, Groovy. Guards: real share within 3 points of the declared one, at least 400
distinct files per group, every byte-decile keeps the global mix (the training stack cuts spans by position). The build
checks each repository's root license file and refuses one that is declared permissive but is not.

**binary-context** (`sources/binary-context.yaml`): 137 Debian 13 packages (executables and libraries for amd64, arm64
and armhf, static libraries, firmware images), 86 PyPI wheels (extension modules and bundled libraries for Linux,
Windows and macOS), Go for Windows and macOS and CPython for Windows (PE and Mach-O executables, DLLs and the `.pyc` of
the standard library), Git for
Windows, 7-Zip, tree-sitter and sql.js WebAssembly modules, and 26 Maven Central jars (class files, jars, and the native
libraries some jars carry). Groups: ELF, ELF spans that start in C++ symbol tables, ar archives, pyc, PE, Mach-O, Java,
WebAssembly, firmware. **Windows system files are not redistributable and are not used**; open-source Windows builds
stand in for them. Debian `copyright` licenses are recorded per package in the build manifest.

**pii** (`sources/pii.yaml`): frequency lists of given names and surnames, a street map with postcodes and a list of
municipalities (INE, CC BY 4.0); first names, surnames and places (US Census Bureau, public domain); sentences
(Tatoeba, CC BY 2.0 FR); street suffixes (Faker, MIT); and TRAIN documents of four synthetic PII datasets (OpenPII-1M
CC BY 4.0, Nemotron-PII CC BY 4.0, two Gretel datasets Apache-2.0) with their source labels. The lists rebuild byte for
byte from the pinned files (the `.xls` of surnames is read by `pbdata/xls.py`, standard library only) as long as the
publishers serve those files: INE and Tatoeba replace theirs in place ([Known provenance
gaps](#known-provenance-gaps) says what changes then, and what to do).

**pii-pool** (`sources/pii-pool.yaml`): the TRAIN files of the same four datasets, whole, for the release pool of the
`pii` specialist: 387,391 documents in 23 languages after the hard overlap check ([below](#the-pii-pool)).

**secret-catalogs** (`sources/secret-catalogs.yaml`): rule ids, regular expressions and keywords (gitleaks v8.30.1,
MIT), detector names, single-literal regular expressions and example strings (trufflehog v3.97.6, AGPL-3.0). The
catalogs hold credential-like example strings and only ever exist under `$PUREBYTE_DATA`. What the generator reads of
them: the gitleaks rules (sampled, and validated when the rule compiles), their keywords and documented false positives,
the trufflehog detector names (as credential variable names) and, when no rule gives a valid value (0.035 % of the draws
with the pinned catalog: 70 of 200,000 measured on Python 3.12), one trufflehog example string verbatim. The gitleaks
`tps` and the trufflehog `patterns` are extracted for inspection and read by nothing.

Attribution for model cards: "Contains data from the Instituto Nacional de Estadística (INE), www.ine.es, CC BY 4.0";
"Sentences from Tatoeba (tatoeba.org), CC BY 2.0 FR"; "Ai4Privacy / Ai Suisse SA OpenPII-1M, CC BY 4.0"; "NVIDIA
Nemotron-PII, CC BY 4.0"; "Gretel.ai datasets, Apache 2.0".

## Evaluation sets

| Set | Role | What | Source |
|---|---|---|---|
| `code-fp-a` | test | 6 repositories without real credentials (Python, Go, JavaScript, Ruby, Java) ¹ | git, pinned commits |
| `code-fp-b` | test | 18 repositories in 8 languages ¹ | git, pinned commits |
| `code-fp-c` | test | 12 repositories: Swift, Kotlin/Android, Ansible, Compose, Helm, Nix, Terraform, Elixir, nginx, dotfiles, notebooks ¹ | git, pinned commits |
| `code-fp-dev` | dev | 6 repositories whose findings may be inspected during development | git, pinned commits |
| `creddata-test` | test | CredData, 168 repositories | Samsung/CredData, pinned commit |
| `creddata-dev` | dev | CredData, the other 169 repositories | same |
| `bin-fp-a` | test | 24 Python extension modules | 2 PyPI wheels |
| `bin-fp-b` | test | 24 Linux system files and 15 Windows DLLs ² | 15 Ubuntu packages, 2 wheels; the DLLs are local-only |
| `bin-fp-c` | test | 69 files in 14 formats ³ | Maven Central, NuGet, PyPI, the Android NDK and Arduino downloads, npm; 46 files local-only |
| `pii-bench-test` | test | 4,122 sampled documents (Nemotron-PII test, OpenPII-1M validation, Gretel finance test) | Hugging Face, pinned revisions |
| `pii-bench-dev` | dev | 2,022 more, disjoint | same sampling run |
| `piimb` | test | PII Masking Benchmark (CC BY-NC 4.0: evaluation only, scores only are published) | Hugging Face, pinned revision + sha256 |

Every binary evaluation file is listed with its sha256 and size; the bytes of local-only files are never distributed.

1. Eight code-fp repositories are CredData repositories too, at other commits: tornado (code-fp-a); starlette, got
   and devise (code-fp-b); awesome-compose (code-fp-c) are CredData DEV repositories whose labeled files feed the
   secrets-code training stream ([Overlap with the code-fp exams](#overlap-with-the-code-fp-exams)); ansible-for-devops,
   helm-charts and phoenix (code-fp-c) are CredData TEST repositories. "Without real credentials" is the sets'
   premise, and CredData's labels contradict it in part: at CredData's commits it labels 89 lines of those eight
   repositories as credentials, so a finding there is not always a false alarm.
2. The runtime's published scans count 25 Linux files: they also read `/usr/bin/perl`, a hard link to the listed
   `perl5.40.1` (the same bytes and sha256; `protoc` and `protoc-3.21.12.0` are identical too, and both are listed).
3. Two npm members were corrected on 2026-09-25 ([Known provenance gaps](#known-provenance-gaps)).

**CredData split.** `fetch_creddata.py` imports `benchmarks/creddata/score.py`, the pinned copy of the runtime's
public scorer ([benchmarks/creddata/](https://github.com/purebyte-ai/purebyte/tree/main/benchmarks/creddata), checked
byte for byte against it in CI), and takes its `CredData(...).split`, so the DEV/TEST split is the public scorer's by
construction (seed 20260922, repositories sorted by positive units, consecutive pairs, a seeded coin per pair). The
lists in `exams/creddata.yaml` are what that gives for the pinned snapshot; the script stops if the two ever differ.

**CredData download.** CredData publishes line labels (`meta/`) and a script that clones each of the 337 upstream
repositories whole at its commit (tens of GB) to copy out the labeled files. `fetch_creddata.py` gets the same files
for about 1 GB (`pbdata/creddata.py`): per repository, its commit and trees only (`git fetch --filter=blob:none`), the
FileID of every path (sha256 of the path, first 8 hex digits, as CredData computes it), and then exactly the blobs
whose FileID and destination (`data/<repo>/<scope>/<FileID><ext>`) the meta lists, plus the license files CredData
copies; symbolic links, `.git/` paths and `.xml` files are skipped as CredData skips them. It resumes where it stopped
(`exams/creddata/fetch-status.json`) and stops if any labeled file did not arrive. Then CredData's own
`obfuscate_creds.py` runs, unmodified and once: it replaces every labeled credential value with a random one of the
same shape, seeded per row (the same result on every machine), and rewrites the files with LF line endings. It writes
in text mode, so it runs on Linux, macOS or WSL only. `--method official` runs CredData's `download_data.py` instead,
with its whole output captured: that script ends by running `obfuscate_creds.py`, which logs every labeled value it
leaves unchanged ("The same value: ..."); those lines are counted and never shown, only its progress lines are, and a
failure shows only its last lines that cannot hold a value (the default method does the same with the obfuscation).
The whole fetch took a few minutes on a fast connection, and gave, byte for byte, the 11,564 files of the copy the
released stream was built from (plus one file an antivirus had removed from that copy).

## The CredData DEV stream

`build_creddata_stream.py` builds `streams/creddata-dev-spans.npz`, the labeled real windows that the secrets-code
1.0.0 recipe adds to every training step (`train.real_windows`, `real_mode: spans`). The recipe is in
`pbdata/creddata_stream.py`:

* **Only CredData DEV, and only its `train` subset**: the 117 DEV repositories whose sha1(id) % 100 is below 70. The
  other 52 (`select` and `heldout`) are where operating points are chosen and errors studied, and CredData TEST never
  feeds training. A hard check runs before a meta row is used and before a file is opened (the repository is DEV, in
  `train`, and the file lives under its directory); the meta file of any other repository is never opened on that
  path, and the check runs again on the written file.
* **The TEST guard**: a DEV credential whose value is also a CredData TEST credential's (a file copied into repositories
  of both halves) is not labeled and its windows are dropped (6 credentials). The TEST values are compared as keyed
  hashes whose key exists only in the building process.
* **Labels per span**: the value; a multi-line PEM block as one span; a URL credential as its whole URL. A UUID, or a
  credential whose only rules are UUID, Salt or Nonce, is positive only under an unequivocal credential name;
  otherwise it is ignored, and every window that touches it, or touches an X unit, is dropped.
* **Windows**: 24,000 windows of 512 bytes in slots of four: two around a credential, one around a labeled false
  positive, one random background window. In half of the credential slots the value is replaced by a generated value
  of the same form (the secrets generator) and the same position gets its contrast (a placeholder, an environment
  lookup, an interpolation, an empty value): two windows. Seed 0.

The file holds `X` (uint8 [N, 512]), `S` (the 0/1 span mask), `Y` (`S.any(1)`), `kind` (0 original, 1 re-planted,
2 contrast, 3 hard negative, 4 background), `repo` (an index into the list of repositories) and `meta` (JSON:
parameters, counts, repositories, the split's digest). `creddata-dev-spans.manifest.json` records every input (CredData
commit and partition digests, the split, the generator's and the catalogs' sha256, Python and NumPy versions), the
sha256 of every array, the overlap with the exams and whether the arrays are the released ones. `--verify` re-hashes
the file and re-runs the hard check. Nothing prints or records a credential value; the stream holds CredData bytes, so
it is only ever written under `$PUREBYTE_DATA`.

**Reproducibility.** With the default parameters, from the pinned snapshot, on Linux with Python 3.14, the five arrays
are bit for bit those of the stream the released weights were trained with: the build compares the sha256 of every
array with the ones pinned in `pbdata/creddata_stream.py` and prints the result. The `.npz` file itself differs in one
member, `meta`: the maintainers' tool recorded a path of their private tooling and the digest of their own split file
there, and the public builder records public provenance instead (the public arrays with that member give the released
file, sha256 `0dae369fdef9b06a...`). The conditions:

1. **Python 3.14.** The generator validates every sampled provider value with its catalog rule when the rule compiles.
   Of the 120 rules it samples, Python 3.14 compiles 99 and Python 3.10 to 3.13 compile 96: only 3.14 accepts `\z`,
   which three sampled rules use (`curl-auth-user`, `openshift-user-token`, `sentry-org-token`), and the generator
   refuses a global inline flag after the start of a pattern on every version (plain `re` on Python 3.10 would compile
   20 more; measured with the pinned catalog on 3.10.14, 3.12.4 and 3.14.0). With another set of rules a sampled value
   that one version rejects the other keeps, so the re-planted provider values differ, and so does every window drawn
   after the first of them; the build says so. The same holds for every window the secrets generator makes (the
   secrets-code and secrets-bin data, and the SECRET values of the pii structured forms): they are the same on Python
   3.10 to 3.13 and differ on 3.14. NumPy's random generator gave the same numbers with NumPy 2.2 and 2.5.
2. **Every labeled file.** A file missing from the snapshot (an antivirus quarantine, for example) changes the stream:
   the build stops instead.
3. **The catalogs of `build_secret_catalogs.py`**: gitleaks v8.30.1 gives the same 222 rules, in the same order, as
   the catalog the released models were trained with.

**Overlap with CredData TEST: the training factory refuses the released stream.** 847 of its 24,000 windows (392 with a
credential) share at least one 64-byte block with files of 136 of the 168 CredData TEST repositories: five DEV files
are byte-for-byte copies of TEST files, and the rest are shared fragments (one TEST repository accounts for 405 of the
windows). No TEST credential value is in the stream (the TEST guard), but the leak check of the `data` stage refuses
any training data that shares a block with an exam of role `test`, so `pbtrain run secrets-code-1.0.0.yaml` stops
there with the released stream. `--exam-overlap drop` leaves out every window that shares a block with a fingerprinted
`test` exam (23,153 windows remain when only `creddata-test` is fingerprinted), and the windows of the repositories
below; then a missing test-exam fingerprint fails the build unless `--allow-missing-exams`. The factory accepts that
stream, and it is not the released one. Another 194 windows share a block with the DEV repositories kept for choosing
operating points; they belong to `creddata-dev` (role `dev`), which the factory reports and does not enforce.

### Overlap with the code-fp exams

Five repositories of the code-fp false-alarm exams are CredData DEV repositories of the `train` subset, so their
labeled files (at CredData's commits, not at the exams' ones) feed this stream: tornado (`f623c7b3`, code-fp-a),
starlette (`3b7cbdd3`), got (`be221a57`) and devise (`349ac2b1`) (code-fp-b), and awesome-compose (`6e2ed0e4`,
code-fp-c). **641 of the released stream's 24,000 windows (274 with a labeled credential) come from them**: tornado 182,
starlette 59, got 63, devise 249, awesome-compose 88. secrets-code 1.0.0 was trained with that stream, so it has seen
labeled code and hard negatives of those repositories, and its false-alarm counts there are not those of unseen code;
its released seed was the one with the fewest findings on code-fp-b, three of whose 18 repositories are among them
(the runtime's [benchmarks/RESULTS.md](https://github.com/purebyte-ai/purebyte/blob/main/benchmarks/RESULTS.md) gives
the counts with and without them). The identity rule of the source manifests ("another commit of an evaluation
repository is still that repository", `pbdata/identity.py`) did not reach the stream, which is built from an exam
(CredData) rather than from a source manifest. The builder now reports these windows every time (`exam_identity` in
the stream's manifest), and `--exam-overlap drop` leaves them out: 628 of them remain after the content drop above, so
22,525 windows (10,835 with a credential) remain when only `creddata-test` is fingerprinted, and fewer when the code-fp
sets are fingerprinted too (their content overlap is then dropped as well). Three more code-fp-c repositories
(ansible-for-devops, helm-charts, phoenix) are CredData TEST repositories, which no stream reads.

## The PII pool

`build_pii_pool.py` builds the pool the `pii` release recipe trains on (`sources/pii-pool.yaml`, `pbdata/pii_pool.py`):
labeled documents of the TRAIN splits of four public datasets of synthetic personal data, from four files downloaded
whole at pinned revisions and verified by sha256 (3.9 GB). Every line of `pii/public_train_xl/<source>.jsonl` is a
document `{id, source, lang, text, labels: [[char start, char end, source label]]}`:

| File | What | Documents |
|---|---|---:|
| `base_public_train.jsonl` | the sample of `sources/pii.yaml` (what `fetch_pii.py train` samples, drawn here from the same pinned files and checked against its reference digests) | 32,500 |
| `openpii1m_xl.jsonl` | OpenPII-1M TRAIN, 23 languages: 48,000 English and 7,000 of every other language, chosen by a hash of the document id | 199,326 |
| `nemotron_xl.jsonl` | every Nemotron-PII TRAIN document | 78,305 |
| `gretel_en_v1_xl.jsonl` | every Gretel PII masking EN v1 TRAIN document | 42,418 |
| `gretel_finance_xl.jsonl` | every Gretel synthetic PII finance TRAIN document (7 languages) | 34,842 |

A document whose normalized text is already in the pool is written once. `pii/public_train_clean/` holds the first file
alone.

Document ids are not unique in `base_public_train.jsonl`: its sampler numbers a document by the page it comes from
plus the count of documents taken so far, so two documents can get one id. In the release pool, 855 ids are shared by
two documents each (1,710 of the 32,500: 562 ids of Nemotron-PII, 178 of Gretel PII masking EN, 115 of Gretel finance).
The `pii` plugin splits documents by a hash of their id, so both documents of a pair always fall in the same split
(train, select or test: no leak between splits), and an `EXCLUDED.json` naming one id would drop both (no build writes
one; `--verify` flags one). The ids are part of the pool's pinned digests, so they are kept as they are; the next pool
version will number documents by their row.

**The hard overlap check.** Before a candidate is written it is compared with every document of the PII evaluation
sets (the PII Masking Benchmark's 25,033 documents and 150,022 sentences, pii-bench-test and pii-bench-dev, whose labels
are read through the `pii` specialist's `public_map`, as the evaluation reads them). It is left out when:

1. its normalized text (lower case, runs of white space as one space) is an evaluation document's;
2. one of its units (a sentence or a line, normalized, 40 characters or more) is an evaluation unit;
3. its masked template (every labeled span replaced by a placeholder) is an evaluation document's: the same document
   with other values;
4. one of its masked units (60 characters or more, holding a placeholder) is a masked evaluation unit: a template that
   a generator shares between its train and test splits.

Units are cut the same way on both sides (a sentence end or a line break), so an identical region gives identical
units. The check leaves out 29,149 candidates; `HARD_CHECK_EXCLUDED.json` lists them with their level, and
`LEAKAGE.json` counts them per source and level. `--extra-test FILE` adds more evaluation documents (JSON lines with a
`text`). The documents that the maintainers' own evaluation set (built from their code corpus, not public) excluded are
listed in the manifest (`build.listed_exclusions`, by id and a key of their text), so that the public build gives the
release pool without that set.

**Reproducibility.** With the default options, the five files are byte for byte those of the pool the released model
was trained with: the manifest pins their sha256 (`build.reference_sha256`), and the build's `MANIFEST.json` says
whether they match (`release_pool`). It needs the PII evaluation sets on disk (`fetch_exams.py --set piimb --set
pii-bench-test --set pii-bench-dev`; the sampled pii-bench files are compared with the samples the release pool was
checked against, `BENCH_REFERENCE_SHA256` in `pbdata/pii.py`, and the set manifest records the result) and, like every
build, the fingerprints of every declared set. To build it from the PII sets alone:

```bash
python data/fetch_exams.py --set piimb --set pii-bench-test --set pii-bench-dev
python data/fingerprint_exams.py --set piimb --set pii-bench-test --set pii-bench-dev
python data/build_pii_pool.py --exam-overlap drop --allow-missing-exams
```

`--allow-missing-exams` gives up the content exclusion of the binary, code and CredData sets from the pool (its
manifest lists them); the four-level check against the PII sets and the drop of documents sharing a block with them
still run, and the training factory still requires the fingerprints of the pii recipe's exams (`piimb`,
`pii-bench-test`).

**The content check, and the training factory.** The documents of one synthetic generator share phrases across its
splits, and some of them hold 64-byte blocks that the data scripts' content check finds (the four levels compare whole
sentences and templates): about 3 % of the release pool's documents share at least one block with the benchmark or with
pii-bench-test. The build reports them (`LEAKAGE.json`, `exam_blocks`) and keeps them, because the pool is then the
release pool. The training factory refuses any training window that shares a block with a `test` exam, so
`python -m pbtrain run specialists/pii/recipes/pii-1.0.0.yaml` stops at the `data` stage on the release pool.
`--exam-overlap drop` leaves out every document that shares a block with a `test` exam: the factory accepts that pool,
and it is not the release pool (a model trained on it is expected to land close, not bit for bit). `--piimb-overlap`
also writes `PIIMB_OVERLAP.json`, the benchmark sentences that share a block with the pool, for scoring the benchmark
without them (`--exclude` of `benchmarks/piimb/score.py`, the pinned copy of the runtime's
[scorer](https://github.com/purebyte-ai/purebyte/tree/main/benchmarks/piimb), or `EXCLUDE=` of its `run.sh` there); it
is derived from the benchmark, so it stays local.

## Determinism and records

Same inputs give the same bytes: candidates are visited in the order of a sha256 key (seed, group, content id), never
in directory or dictionary order and never through a random generator whose stream could change between versions;
pieces are interleaved by deficit with a fixed tie-break; archives are read member by member in sorted order; the index
is written with a fixed gzip timestamp. Each build writes `<name>.manifest.json`: the manifest's sha256, every input with
its pin (and git tree id), license and license check, the output sha256, bytes and pieces, the real share of every group
and the guards, and the exclusion report (identity result, every exam set's fingerprint digest, and files and bytes
removed per set). `tests/test_corpus_determinism.py` builds twice from local repositories and compares the bytes.

## Known provenance gaps

* **Local-only evaluation files.** 15 Windows DLLs of `bin-fp-b` and 46 files of `bin-fp-c` (Windows system
  executables and driver firmware, locally compiled `.pyc` and objects, application-bundled Node.js addons and
  WebAssembly modules, Android emulator skins, JNA class files from an Eclipse bundle) have no public download. They
  are identified by sha256 and size; without a local copy their fingerprints are incomplete (recorded, and repeated by
  every build: [Limits](#limits-stated-plainly)).
* **Unpublished container digests.** PyTorch's CPU wheel and the Android NDK archive publish no SHA-256: those entries
  use `pin.verify: members` (every member used is verified by sha256). The three npm tarballs of `bin-fp-c` are pinned
  by sha256; the npm registry publishes their SHA-512, which they matched when they were pinned (2026-09-25).
* **A correction of `bin-fp-c` (2026-09-25).** Two of its npm members had been pinned with the sha256 and size of the
  builds that desktop applications bundle under the same package name and version, not with those of the files npm
  publishes, so `fetch_exams.py` stopped on them. `data/exams/binary-fp.yaml` now pins the published files (its header
  records the change): `rollup.win32-x64-msvc.node` of `@rollup/rollup-win32-x64-msvc` 4.54.0, 2,576,896 bytes before
  and 2,575,872 now; `zeromq_node.napi.glibc.node` (`prebuilds/win32-x64/node.napi.glibc.node` of `zeromq`
  6.0.0-beta.16), 927,264 bytes before and 644,608 now. The bin-fp-c figures published for secrets-bin 1.0.0 (the
  runtime's benchmarks/RESULTS.md) were measured on the earlier, application-bundled copies. None of the findings
  published there is in these two files, and the released model reports nothing on the npm-published copies either
  (scanned on 2026-09-25 with every span reported). The manifest changed, so fingerprints of `bin-fp-a`, `bin-fp-b` and
  `bin-fp-c` made before that date are stale: run `fingerprint_exams.py` for them again. The same day a comment and a
  description of `data/exams/creddata.yaml` were corrected (one release choice was made with TEST scores in view);
  no pin changed, but a manifest is recorded by the SHA-256 of its bytes, so fingerprints of `creddata-test` and
  `creddata-dev` made before that date are stale too.
* **Files replaced in place upstream.** The INE tables (yearly) and the Tatoeba exports (weekly) are republished at the
  same URL, so their pins hold only while the pinned copies are served: the Tatoeba pins are the weekly exports served
  on 2026-09-23 (`Last-Modified` 2026-09-19), the INE ones the tables of April 2026. There is no mirror of those copies
  yet. When a pin no longer matches, `fetch_pii.py lists` stops, and:
  - `--skip-sentences` builds the ten lists that do not come from Tatoeba and leaves out `sentences_{es,en}.tsv`. Only
    the prose and e-mail scenes of the pii generator read sentences, and the pii release recipe has neither, so its
    data stay bit for bit the same (tested in `specialists/pii/tests`); recipes with prose or e-mail refuse to build.
  - `--accept-upstream-change` uses the current file and records the deviation in `pii/lists/SHA256.json`. A new
    Tatoeba export changes only the sentence lists (the same consequences as above); a new INE table changes the name,
    street and municipality lists, and with them every generated window of the pii structured forms and typed recipes.
  - `--raw-dir DIR` takes the pinned copies from a folder, when someone has them (matched by name and sha256).

  The Hugging Face rows API serves only the current revision of Parquet datasets: a newer revision stops the sampling
  unless `--allow-revision-drift` (recorded). `build_pii_pool.py` does not use it: it reads the four TRAIN files pinned
  by revision and sha256.
* **Earlier corpora.** The first models were trained with context corpora assembled before this folder existed, from
  repositories cloned at branch heads without recorded commits and from the system files of one machine. They cannot
  be rebuilt byte for byte; the corpora built here replace them, with every input pinned and the exclusion enforced.
* **Catalogs.** gitleaks v8.30.1 reproduces the rule ids and regular expressions of the catalog used for the first
  models exactly; that catalog lacked the keywords of 35 rules whose keyword list spans several lines, and its
  true/false-positive examples were extracted with a looser heuristic (the builder here takes whole literal elements:
  counts differ). trufflehog v3.97.6 reproduces its 876 detector names exactly; its patterns and examples come from a
  documented heuristic that was not compared with the original. The trufflehog examples are strings of its detectors'
  test files (AGPL-3.0); the generator plants one verbatim, as a positive, only when no catalog rule gives a valid value
  (0.035 % of the catalog draws, [above](#sources-training-material)).

## Tests

```bash
python -m pytest data/tests -q
```

Offline (no network): manifest schema and repository hygiene, identity exclusion on the real manifests and on
synthetic conflicts, content exclusion on synthetic bytes with the training package's fingerprint module (text cuts,
binary spans, junctions, missing and stale fingerprint files), determinism of the corpus writer and of the code
builder on small local git repositories, archive readers, the Go-literal, TOML and `.xls` parsers. The CredData stream
on a synthetic CredData of six repositories with values generated at test time (`tests/test_creddata_stream.py`): no
byte of TEST, of the other DEV subsets, of an X unit or of an ignored UUID reaches a window; meta files of other
repositories are never opened; the hard check (also on a tampered file), the TEST guard, the labels and the contrast,
determinism, the exam-overlap report and drop (by content, and by identity with the code-fp repositories), the
trainer's loader; the partial fetch from a local repository, the line forms, and CredData's own scripts (the
obfuscation, and `download_data.py` for `--method official`) run with their output captured, never showing a value.
The evaluation-set scripts on synthetic sets (`tests/test_exam_scripts.py`): a failing set never stops the others,
files without a public download never stop a set, fingerprinting downloads nothing, the pii-bench samples are cut once
per run, and each script accepts only the options it acts on. The smoke stand-ins (`tests/test_smoke_data.py`): the
same bytes as the test suites' generators, a fresh data folder gets everything, real data are never replaced.
