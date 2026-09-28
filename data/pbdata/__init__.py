"""pbdata: the library behind the data scripts in `data/`.

Every script in `data/` downloads or builds something from a pinned manifest (`data/sources/*.yaml`,
`data/exams/*.yaml`) into `$PUREBYTE_DATA` (default `~/.purebyte/data`). Nothing produced here is ever committed.

Modules:
    paths       where the scripts read and write; the one bridge to `training/` (fingerprints, data root)
    util        hashing, atomic writes, JSON, logging, polite process settings
    cli         what the scripts share: options, the identity preamble, verification helpers
    globs       the path patterns used by `take:` blocks
    manifest    YAML manifests: loading and schema validation
    identity    the identity half of the exclusion guarantee (sources vs exams)
    fetch       downloads and git objects, verified by sha256 / commit
    archives    reading members of .deb, wheels, zips and tarballs in memory
    exclusion   the content half of the exclusion guarantee (exam fingerprints)
    exams       evaluation sets on disk: fetching them and reading their content
    corpus      the deterministic corpus writer (quotas, interleaving, verification, guards, manifest)
    licences    a sanity check of declared licences against the licence files of a checkout
    pii         the PII lists, public TRAIN documents and sampled benchmark
    xls         a standard-library reader of legacy .xls workbooks
    tomlsub     TOML without tomllib (Python < 3.11)
"""
