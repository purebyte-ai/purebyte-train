"""Offline stand-ins for $PUREBYTE_DATA in tests: a generated code-like corpus and tiny catalogs of MADE-UP credential
formats (no real provider, nothing that could pass for a live credential; values are sampled at test time), and exam
fingerprints made of random bytes that share nothing with the corpus."""
import contextlib
import json
import os

import numpy as np

# the exams the secrets-code specialist lists (specialist.yaml): a release needs every one checked by the data stage
SECRETS_CODE_EXAMS = ("creddata-test", "code-fp-a", "code-fp-b", "code-fp-c")


def make_data(data, exams=("unrelated-exam",) + SECRETS_CODE_EXAMS):
    """A $PUREBYTE_DATA for the secrets-code smoke recipe: corpus, catalogs and unrelated exam fingerprints."""
    from pbtrain.data import fingerprint as fp
    code_corpus(os.path.join(data, "corpora", "smoke-context.bin"))
    catalogs(os.path.join(data, "catalogs"))
    os.makedirs(os.path.join(data, "exams", "fingerprints"), exist_ok=True)
    for i, name in enumerate(exams):
        other = np.random.default_rng(5 + i).integers(0, 256, 200_000, dtype=np.uint8)
        fp.save(os.path.join(data, "exams", "fingerprints", name + ".npz"), fp.fingerprints(other),
                dict(name=name, files=1, bytes=len(other)))
    return data


def fabricate_finetune_pass(runs, spec):
    """Rewrites the metrics of a finished `finetune` stage so that only the rival margin decides `eval` (control 0,
    recall 1 on select, no flat sweep), for tests of what comes after the gate; the checkpoints and their sha256 are
    untouched. The chain stays consistent: nothing downstream has recorded this done.json yet."""
    from pbtrain.factory.chain import done_path
    p = done_path(runs, spec, "finetune")
    with open(p, encoding="utf-8") as f:
        d = json.load(f)
    d["control_f1"] = 0.0
    for r in d["runs"]:
        r.update(recall_select=1.0, flat_sweep=False)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(d, f)


@contextlib.contextmanager
def data_and_work(data, work):
    """PUREBYTE_DATA / PUREBYTE_WORK set for the block, restored afterwards."""
    old = {k: os.environ.get(k) for k in ("PUREBYTE_DATA", "PUREBYTE_WORK")}
    os.environ["PUREBYTE_DATA"], os.environ["PUREBYTE_WORK"] = data, work
    try:
        yield data, work
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

WORDS = ("user", "total", "items", "config", "result", "value", "count", "index", "buffer", "request", "response",
         "client", "server", "handler", "cache", "timeout", "retries", "path", "name", "status", "payload", "session")


def code_corpus(path, n_bytes=400_000, seed=0):
    """Deterministic pseudo source code: functions, assignments, config lines, comments."""
    g = np.random.default_rng(seed)
    w = lambda: WORDS[int(g.integers(0, len(WORDS)))]
    out, size = [], 0
    hexd = "0123456789abcdef"
    rnd_hex = lambda n: "".join(hexd[int(i)] for i in g.integers(0, 16, n))
    while size < n_bytes:
        k = int(g.integers(0, 9))
        if k >= 6:        # high-entropy literals that are NOT credentials (digests, ids): what makes the rule stumble
            name = ("checksum", "etag", "commit_sha", "digest", "build_id", "revision")[int(g.integers(0, 6))]
            out.append(f'{name} = "{rnd_hex(int(g.choice([32, 40, 64])))}"\n')
            size += len(out[-1])
            continue
        if k == 0:
            s = f"def {w()}_{w()}({w()}, {w()}):\n    {w()} = {w()} * {int(g.integers(1, 99))}\n    return {w()}\n"
        elif k == 1:
            s = f'{w()}_{w()} = "{w()}-{w()}"\n'
        elif k == 2:
            s = f"# {w()} the {w()} before {w()}\n"
        elif k == 3:
            s = f"{w()}:\n  {w()}: {int(g.integers(0, 1000))}\n  {w()}: {w()}\n"
        elif k == 4:
            s = f"if {w()} > {int(g.integers(0, 50))}:\n    {w()}.append({w()})\n"
        else:
            s = f'{w().upper()}_{w().upper()}={w()}\n'
        out.append(s)
        size += len(s)
    data = "".join(out).encode()[:n_bytes]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        f.write(data)
    return path


def catalogs(folder, seed=0):
    """Two made-up formats (`demo_` + 24 lower-alnum, `ACMEX` + 20 upper-alnum) and one example value, generated."""
    g = np.random.default_rng(seed)
    alnum = "abcdefghijklmnopqrstuvwxyz0123456789"
    example = "demo_" + "".join(alnum[int(i)] for i in g.integers(0, len(alnum), 28))
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, "gitleaks.json"), "w", encoding="utf-8") as f:
        json.dump({"rules": [{"id": "demo-token", "regex": r"\bdemo_[a-z0-9]{24}\b", "kw": ["demotoken"]},
                             {"id": "acmex-key", "regex": r"ACMEX[A-Z0-9]{20}", "kw": ["acmexkey"]}],
                   "tps": [], "fps": ["demo_placeholder_value_for_docs"]}, f)
    with open(os.path.join(folder, "trufflehog.json"), "w", encoding="utf-8") as f:
        json.dump({"providers": ["demoprov", "acmexprov"], "patterns": [], "examples": [example]}, f)
    return folder
