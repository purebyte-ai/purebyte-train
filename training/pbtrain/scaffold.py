"""`pbtrain new-vertical NAME`: a new specialist folder that already runs end to end on CPU (with a toy generator that
you then replace). Everything it writes is a starting point with TODOs, never a silent default."""
import os
import re

from .paths import SPECIALISTS_SRC, require_source_tree

SPECIALIST_YAML = """# {name}: TODO one line on what it detects. See README.md.
name: {name}
public_name: {name}
description: TODO
plugin: plugin.py
export:
  profile: none                    # none | secrets-code | secrets-binary | redact (the runtime's spec/OUTPUT.md)
  labels: {labels}
  window: {{size: 512, stride: 384}}
release:
  ensemble: 3                      # members of a release: the released seed + the next ones (train at least as many seeds)
  vote: 2
exams: []                          # TODO: the real evaluation sets (data/exams/*.yaml); a release needs their fingerprints
"""

PLUGIN = '''"""Vertical `{fmt}` of the {name} specialist. TODO: describe the data (what is real, what is generated).

Contract: pbtrain/verticals/base.py. Read EVERY key of the task's `data:` block that you support: the factory fails on
declared keys that nobody read.
"""
import numpy as np
from pbtrain.verticals import Partitions, Vertical, register

ENTITIES = {entities!r}


def _toy_windows(n, seq_len, rng):
    """TODO: replace with the real generator. Printable noise with one planted ASCII token per positive window."""
    X = rng.integers(32, 127, (n, seq_len), dtype=np.uint8)
    S = np.zeros((n, seq_len), dtype=np.uint8)
    Y = np.zeros(n, dtype=np.uint8)
    for i in range(0, n, 2):
        e = int(rng.integers(0, len(ENTITIES)))
        L = int(rng.integers(6, 16))
        a = int(rng.integers(0, seq_len - L))
        X[i, a:a + L] = np.frombuffer((ENTITIES[e].upper() * 16)[:L].encode(), np.uint8)
        S[i, a:a + L] = 1 if len(ENTITIES) == 1 else 1 + e
        Y[i] = 1
    return X, Y, S


@register
class {cls}(Vertical):
    data_format = "{fmt}"
    needs_corpus = False              # TODO: True when windows are cut from a real corpus (data.path)
    overlap_guard = "windows"         # "chunks" when windows are cut from real bytes
    entities = tuple(ENTITIES)

    def build(self, corpus, cfgd, fracs):
        n_tr, n_ev, T = int(cfgd["n_train"]), int(cfgd["n_eval"]), int(cfgd["seq_len"])
        seed = int(cfgd["split_policy"]["seed"])
        return Partitions(*(_toy_windows(n, T, np.random.default_rng(seed + k))
                            for k, n in ((0, n_tr), (1000, n_ev), (2000, n_ev))))

    def manifest(self, cfgd, t=None):
        return dict(generator="toy (TODO)")

    def span_rival(self, select, test):
        """TODO: the rule a competent human would write, tuned on select, scored on test like the model: a dict with
        span_f1, precision, recall and `name` (see specialists/secrets-code/plugin.py). Until it exists this returns
        None, and the `baselines` stage refuses the run: a model is never delivered without a rival to beat."""
        return None
'''

SMOKE = """# CPU smoke run of {name}: proves the pipeline in minutes, not the model.
name: {name}-smoke
version: 0.0.1
specialist: {name}
task_type: tag
output_schema:
  labels: {labels}
  per_byte: true
  min_select_recall: 0.05
data:
  format: {fmt}
  split_policy: {{unit: corpus_range, seed: 0}}
  n_train: 512
  n_eval: 64
  seq_len: 128
  control: shuffled_spans
train:
  seeds: [0, 1, 2]
  mode: causal
  size: mini64
  steps: 200
  lr: 0.003
  batch: 16
engine:
  threads: 1                       # threads of the engine's model_dump in `verify` (PUREBYTE_MODEL_DUMP)
"""

README = """# {name}

TODO: what it detects, in one paragraph; what it does NOT detect.

## Data
TODO: which corpus (data/build_*.py), which generator families, which real sets; how the exams are kept out.

## Train
```bash
pbtrain run specialists/{name}/recipes/smoke-cpu.yaml          # the pipeline, on CPU, in minutes
pbtrain run specialists/{name}/recipes/{name}-0.1.0.yaml --device auto --parallel 3
```

## Evaluate
TODO: the exams, the commands that produce their metrics, and `pbtrain eval TASK --record EXAM --metrics FILE`.
"""

TEST = '''"""The plugin contract of {name} (offline, tiny)."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def test_plugin_builds_valid_partitions():
    from pbtrain.factory.spec import Watched
    from pbtrain.verticals import get, load_specialist
    load_specialist("{name}")
    v = get("{fmt}")
    cfgd = Watched(dict(n_train=16, n_eval=8, seq_len=64, split_policy=dict(seed=0)))
    t = v.build(None, cfgd, (0.8, 0.1, 0.1))
    t.check(64, v.n_entities)
'''


def new_vertical(name, entities, data_format=None, root=None):
    if not re.fullmatch(r"[a-z][a-z0-9-]{1,40}", name):
        raise SystemExit(f"specialist names are lower-case words with dashes: {name!r}")
    root = root or require_source_tree(SPECIALISTS_SRC)
    folder = os.path.join(root, name)
    if os.path.exists(folder):
        raise SystemExit(f"{folder} exists")
    fmt = data_format or name.replace("-", "_")
    cls = "".join(p.capitalize() for p in fmt.split("_"))
    labels = "[" + ", ".join(entities) + "]"
    files = {
        "specialist.yaml": SPECIALIST_YAML.format(name=name, labels=labels),
        "plugin.py": PLUGIN.format(name=name, fmt=fmt, cls=cls, entities=list(entities)),
        os.path.join("recipes", "smoke-cpu.yaml"): SMOKE.format(name=name, labels=labels, fmt=fmt),
        "README.md": README.format(name=name),
        os.path.join("tests", "test_plugin.py"): TEST.format(name=name, fmt=fmt),
    }
    for rel, text in files.items():
        p = os.path.join(folder, rel)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
    return f"created {folder}: " + ", ".join(sorted(files)) + ". Next: training/docs/new-specialist.md"
