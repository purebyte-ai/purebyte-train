"""task.yaml: loading, validation, and the rule that every declared key must act.

A task file is the whole recipe of one specialist version. Top-level keys:

    name          the specialist's name (work dir); `specialist` defaults to it
    version       the version of this recipe (semantic version, e.g. 1.0.0)
    specialist    folder under specialists/ whose plugin builds the data and whose specialist.yaml says how the model
                  is exported and released
    description   free text (the only key that the task digest leaves out)
    task_type     tag (spans; the only one the factory builds today)
    output_schema labels, per_byte (true for tag), min_select_recall (collapse alarm), bias_policy, per_type_bias
    data          format (the vertical), path (the context corpus), n_train, n_eval, seq_len, split_policy, control,
                  and the vertical's own keys
    train         seeds, mode, size, steps, lr, batch (required) and the trainer options (pbtrain.train.recipe)
    criteria      PRE-REGISTERED criteria (factory/criteria.py); required when the run decides (`train.decide`)
    exams         exam names whose fingerprints must be registered for the leak check (data stage)
    engine        {threads: N}: the threads of the engine's `model_dump` in `verify`
    base_checkpoint  a body checkpoint to start from (path, `$PUREBYTE_DATA` expanded)

Unknown keys are refused, and so are values of the wrong type or out of range. Keys of the `data:` block are WATCHED:
after the vertical builds the data, a declared key that nobody read fails the stage (a typo like `extra_line: 5` would
otherwise silently do nothing).

`load` also computes the task's DIGEST: the sha256 of its parsed content (key order and comments do not matter;
`description` is left out). Every stage records it and every later stage refuses a chain made from another digest
(factory/chain.py): a task edited after `data` is run again from `data`.
"""
import hashlib
import json
import os
import re

import yaml

from ..train.recipe import from_train_block, missing_required
from .chain import StageError

TOP_KEYS = {"name", "version", "specialist", "task_type", "description", "output_schema", "data", "train", "criteria",
            "exams", "engine", "base_checkpoint"}
OUTPUT_KEYS = {"labels", "per_byte", "min_select_recall", "bias_policy", "per_type_bias"}
SPLIT_KEYS = {"unit", "seed", "train_frac", "select_frac", "test_frac"}
ENGINE_KEYS = {"threads"}
DESCRIPTIVE_KEYS = {"description"}           # left out of the task digest
NAME = re.compile(r"[a-z][a-z0-9-]{0,62}")    # names and specialists: they become folder names
VERSION = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z.-]+)?")
EXAM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")


class Watched(dict):
    """A dict that records which leaves were read (nested dicts are watched too)."""

    def __init__(self, data, path=""):
        super().__init__(data)
        self._path, self._read = path, set()

    def _wrap(self, key, value):
        if isinstance(value, dict) and not isinstance(value, Watched):
            value = Watched(value, f"{self._path}{key}.")
            dict.__setitem__(self, key, value)
        return value

    def __getitem__(self, key):
        self._read.add(key)
        return self._wrap(key, dict.__getitem__(self, key))

    def get(self, key, default=None):
        if key in self:
            return self[key]
        return default

    def unread(self):
        out = []
        for k, v in dict.items(self):
            if isinstance(v, Watched):
                out += v.unread() if k in self._read else [f"{self._path}{k}"]
            elif k not in self._read:
                out.append(f"{self._path}{k}")
        return out


def digest(spec):
    """sha256 of the task's parsed content, `_` keys (added by `load`) and descriptive keys left out."""
    body = {k: v for k, v in spec.items() if not str(k).startswith("_") and k not in DESCRIPTIVE_KEYS}
    text = json.dumps(body, sort_keys=True, default=str, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def load(path):
    """The task as a dict with `_path`, `_sha256` (of the file's bytes, 16 hex) and `_digest`; validated."""
    try:
        with open(path, "rb") as f:
            raw = f.read()
    except OSError as e:
        raise StageError(f"cannot read the task {path}: {e}")
    try:
        spec = yaml.safe_load(raw)
    except yaml.YAMLError as e:
        raise StageError(f"{path}: not valid YAML ({e})")
    if not isinstance(spec, dict):
        raise StageError(f"{path}: not a YAML mapping")
    spec["_digest"] = digest(spec)
    spec["_path"] = os.path.abspath(path)
    spec["_sha256"] = hashlib.sha256(raw).hexdigest()[:16]
    spec.setdefault("specialist", spec.get("name"))
    validate(spec)
    return spec


def _number(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _positive_int(block, key, v):
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise StageError(f"`{block}.{key}: {v!r}` must be a whole number >= 1")


def validate(spec):
    extra = sorted(set(k for k in spec if not str(k).startswith("_")) - TOP_KEYS)
    if extra:
        raise StageError(f"unknown top-level keys {extra}; valid: {sorted(TOP_KEYS)}")
    for k in ("name", "version", "task_type", "output_schema", "data", "train"):
        if k not in spec:
            raise StageError(f"the task has no `{k}`")
    if not isinstance(spec["name"], str) or not NAME.fullmatch(spec["name"]):
        raise StageError(f"`name: {spec['name']!r}`: lower-case letters, digits and dashes (it names folders)")
    if not VERSION.fullmatch(str(spec["version"])):
        raise StageError(f"`version: {spec['version']!r}` must be a semantic version such as 1.0.0 or 1.0.0-rc1")
    if not isinstance(spec["specialist"], str) or not NAME.fullmatch(spec["specialist"]):
        raise StageError(f"`specialist: {spec['specialist']!r}` must name a folder of specialists/ (lower-case letters, "
                         f"digits and dashes)")
    if spec.get("description") is not None and not isinstance(spec["description"], str):
        raise StageError("`description` must be text")
    if spec["task_type"] != "tag":
        raise StageError(f"task_type {spec['task_type']!r}: the factory builds span (`tag`) specialists; document "
                         f"(`choice`) tasks are not implemented in pbtrain yet")
    _validate_output(spec)
    _validate_data(spec)
    tr = spec["train"]
    if not isinstance(tr, dict):
        raise StageError("`train:` must be a mapping")
    miss = missing_required(tr)
    if miss:
        raise StageError(f"`train:` must state {miss} explicitly (no silent defaults for the numbers that define a run)")
    try:
        from_train_block(tr)
    except ValueError as e:
        raise StageError(str(e))
    if tr["batch"] > spec["data"]["n_train"]:
        raise StageError(f"`train.batch: {tr['batch']}` is larger than `data.n_train: {spec['data']['n_train']}`: every "
                         f"step would train on fewer windows than declared")
    if tr.get("decide") and len(tr["seeds"]) < 3:
        raise StageError(f"`decide: true` with {len(tr['seeds'])} seed(s): three seeds inform, six decide; fewer "
                         f"than three measure nothing")
    if tr.get("decide") and not spec.get("criteria"):
        raise StageError("`decide: true` needs a PRE-REGISTERED `criteria:` block (written before training)")
    from .criteria import CriteriaError, validate as validate_criteria
    if spec.get("criteria") is not None:
        try:
            validate_criteria(spec["criteria"])
        except CriteriaError as e:
            raise StageError(str(e))
    ex = spec.get("exams")
    if ex is not None:
        if not isinstance(ex, list) or any(not isinstance(x, str) or not EXAM.fullmatch(x) for x in ex):
            raise StageError(f"`exams: {ex!r}` must be a list of exam names (letters, digits, `.`, `_`, `-`)")
        if len(set(ex)) != len(ex):
            raise StageError(f"`exams: {ex}` names an exam twice")
    eng = spec.get("engine")
    if eng is not None:
        if not isinstance(eng, dict) or set(eng) - ENGINE_KEYS or not eng:
            raise StageError(f"`engine: {eng!r}`: the only key is `threads` (the engine's model_dump threads in "
                             f"`verify`)")
        _positive_int("engine", "threads", eng["threads"])
    bc = spec.get("base_checkpoint")
    if bc is not None and (not isinstance(bc, str) or not bc):
        raise StageError("`base_checkpoint` must be a path")
    return spec


def _validate_output(spec):
    from ..eval.operating_point import POLICIES
    oe = spec["output_schema"]
    if not isinstance(oe, dict):
        raise StageError("`output_schema:` must be a mapping")
    if "reject_label" in oe:
        raise StageError("`output_schema.reject_label`: in a `tag` task abstention is per span, not a label; the key "
                         "is not supported")
    bad = sorted(set(oe) - OUTPUT_KEYS)
    if bad:
        raise StageError(f"unknown keys in output_schema: {bad}; valid: {sorted(OUTPUT_KEYS)}")
    if oe.get("per_byte") is not True:
        raise StageError("`output_schema.per_byte` must be true for a `tag` task")
    labels = oe.get("labels")
    if not labels or not isinstance(labels, list) or any(not isinstance(x, str) or not x for x in labels):
        raise StageError("`output_schema.labels` must name the entity type(s): a list of names")
    if len(set(labels)) != len(labels):
        raise StageError(f"`output_schema.labels: {labels}` names a type twice")
    fl = oe.get("min_select_recall")
    if fl is not None and (not _number(fl) or not 0 < fl <= 1):
        raise StageError(f"`output_schema.min_select_recall: {fl!r}` must be a number in (0, 1]")
    if "bias_policy" in oe and oe["bias_policy"] not in POLICIES:
        raise StageError(f"`output_schema.bias_policy: {oe['bias_policy']!r}`; known: {POLICIES}")
    if "per_type_bias" in oe:
        if oe["per_type_bias"] not in (True, False):
            raise StageError("`output_schema.per_type_bias` must be true or false")
        if oe["per_type_bias"] and len(labels) < 2:
            raise StageError("`output_schema.per_type_bias: true` with one label: there is one type, nothing to tune "
                             "per type (it would act on nothing)")


def _validate_data(spec):
    d = spec["data"]
    if not isinstance(d, dict):
        raise StageError("`data:` must be a mapping")
    for k in ("format", "n_train", "n_eval", "seq_len"):
        if k not in d:
            raise StageError(f"`data.{k}` is required")
    for k in ("n_train", "n_eval", "seq_len"):
        _positive_int("data", k, d[k])
    sp = d.get("split_policy") or {}
    if not isinstance(sp, dict):
        raise StageError("`data.split_policy` must be a mapping")
    if set(sp) - SPLIT_KEYS:
        raise StageError(f"unknown keys in split_policy: {sorted(set(sp) - SPLIT_KEYS)}")
    if sp.get("unit", "corpus_range") != "corpus_range":
        raise StageError(f"`split_policy.unit: {sp.get('unit')}` is not implemented (corpus_range)")
    if d.get("control", "shuffled_spans") != "shuffled_spans":
        raise StageError("the control of a span task is `shuffled_spans`")
    fracs(d)


def fracs(data):
    sp = data.get("split_policy") or {}
    try:
        tr, se, te = (float(sp.get("train_frac", 0.80)), float(sp.get("select_frac", 0.10)),
                      float(sp.get("test_frac", 0.10)))
    except (TypeError, ValueError):
        raise StageError(f"split fractions must be numbers: {dict(sp)}")
    if abs(tr + se + te - 1.0) > 1e-9 or min(tr, se, te) <= 0:
        raise StageError(f"split fractions {tr}, {se}, {te} must be positive and add up to 1")
    return tr, se, te


def n_entities(spec):
    return len(spec["output_schema"]["labels"])
