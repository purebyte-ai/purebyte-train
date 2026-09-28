"""Specialists and their vertical plugins.

A specialist is a folder `specialists/<name>/` with:
    specialist.yaml   name, description, the plugin, what the exported model declares (profile, labels, window, n-gram
                      precision, license), the release policy (ensemble, vote), the exams that judge it
    plugin.py         the Vertical subclasses it registers (see base.py)
    recipes/*.yaml    task files for the factory (the released recipe, the recommended next one, a CPU smoke recipe)
    README.md         what it detects, how it is trained and evaluated
`load_specialist("secrets-code")` imports its plugin (and the plugins it `uses:`) and returns its description.

specialist.yaml is VALIDATED (`validate_config`): unknown keys are refused at every level and every value is typed, so
a typo (`ngram_bit: 4`, `ensamble: 3`) cannot silently fall back to a default. Its `exams:` are the evaluation sets of
the specialist: a task that declares `exams:` must have the fingerprints of these too (data stage), and a release
refuses a run whose data stage did not check every one of them (release/package.py).
"""
import importlib.util
import os
import re
import sys

import yaml

from ..paths import SPECIALISTS_SRC, require_source_tree
from .base import REGISTRY, Partitions, Vertical, get, register

__all__ = ["REGISTRY", "Partitions", "Vertical", "get", "register", "load_specialist", "Specialist",
           "list_specialists", "validate_config", "SpecialistError"]
_LOADED = {}
NAME = re.compile(r"[a-z][a-z0-9-]{0,62}")
EXAM = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
TOP = {"name", "public_name", "description", "plugin", "uses", "export", "release", "exams"}
EXPORT = {"profile", "labels", "window", "ngram_bits", "license", "license_name"}
WINDOW = {"size", "stride", "min"}
RELEASE = {"ensemble", "vote"}


class SpecialistError(ValueError):
    pass


def _positive_int(where, v):
    if isinstance(v, bool) or not isinstance(v, int) or v < 1:
        raise SpecialistError(f"{where}: {v!r} must be a whole number >= 1")


def validate_config(name, config, where="specialist.yaml"):
    """Refuses unknown keys and values of the wrong type in a specialist.yaml. Returns the config."""
    if not isinstance(config, dict):
        raise SpecialistError(f"{where}: not a YAML mapping")
    bad = sorted(set(config) - TOP)
    if bad:
        raise SpecialistError(f"{where}: unknown keys {bad}; valid: {sorted(TOP)}")
    if config.get("name", name) != name:
        raise SpecialistError(f"{where}: `name: {config.get('name')}` differs from its folder `{name}`")
    for k in ("public_name", "description"):
        if config.get(k) is not None and (not isinstance(config[k], str) or not config[k]):
            raise SpecialistError(f"{where}: `{k}` must be text")
    if config.get("public_name") is not None and not NAME.fullmatch(config["public_name"]):
        raise SpecialistError(f"{where}: `public_name: {config['public_name']}`: lower-case letters, digits and dashes")
    plugin = config.get("plugin")
    if plugin is not None and (not isinstance(plugin, str) or not plugin.endswith(".py") or os.path.isabs(plugin)
                               or os.path.normpath(plugin).split(os.sep)[0] == ".."):
        raise SpecialistError(f"{where}: `plugin: {plugin!r}` must be a .py file inside the specialist's folder")
    uses = config.get("uses") or []
    if not isinstance(uses, list) or any(not isinstance(u, str) or not NAME.fullmatch(u) for u in uses):
        raise SpecialistError(f"{where}: `uses` must be a list of specialist names")
    ex = config.get("export") or {}
    if not isinstance(ex, dict) or set(ex) - EXPORT:
        raise SpecialistError(f"{where}: unknown keys in `export` {sorted(set(ex) - EXPORT) if isinstance(ex, dict) else ex}"
                              f"; valid: {sorted(EXPORT)}")
    for k in ("profile", "license", "license_name"):
        if ex.get(k) is not None and not isinstance(ex[k], str):
            raise SpecialistError(f"{where}: `export.{k}` must be text")
    if ex.get("labels") is not None and (not isinstance(ex["labels"], list)
                                         or any(not isinstance(x, str) or not x for x in ex["labels"])):
        raise SpecialistError(f"{where}: `export.labels` must be a list of names")
    if ex.get("ngram_bits") is not None and ex["ngram_bits"] not in (4, 32):
        raise SpecialistError(f"{where}: `export.ngram_bits: {ex['ngram_bits']!r}`: 4 or 32")
    win = ex.get("window")
    if win is not None:
        if not isinstance(win, dict) or set(win) - WINDOW:
            raise SpecialistError(f"{where}: `export.window` takes {sorted(WINDOW)}")
        for k, v in win.items():
            _positive_int(f"{where}: `export.window.{k}`", v)
    rel = config.get("release") or {}
    if not isinstance(rel, dict) or set(rel) - RELEASE:
        raise SpecialistError(f"{where}: `release` takes {sorted(RELEASE)}")
    for k in RELEASE & set(rel):
        _positive_int(f"{where}: `release.{k}`", rel[k])
    if rel.get("vote", 1) > rel.get("ensemble", 1):
        raise SpecialistError(f"{where}: `release.vote: {rel.get('vote')}` is larger than `release.ensemble: "
                              f"{rel.get('ensemble', 1)}`")
    exams = config.get("exams") or []
    if not isinstance(exams, list) or any(not isinstance(x, str) or not EXAM.fullmatch(x) for x in exams):
        raise SpecialistError(f"{where}: `exams` must be a list of exam names")
    return config


class Specialist:
    def __init__(self, name, folder, config):
        self.name, self.dir, self.config = name, folder, config

    @property
    def export(self):
        return self.config.get("export") or {}

    @property
    def release(self):
        return self.config.get("release") or {}

    @property
    def ensemble(self):
        return int(self.release.get("ensemble", 1))

    @property
    def vote(self):
        return int(self.release.get("vote", 1))

    @property
    def exams(self):
        return list(self.config.get("exams") or [])

    def recipe(self, name):
        p = os.path.join(self.dir, "recipes", name if name.endswith(".yaml") else name + ".yaml")
        if not os.path.exists(p):
            raise FileNotFoundError(f"no recipe {name!r} in {os.path.join(self.dir, 'recipes')}")
        return p


def _import_plugin(name, path):
    mod_name = "pbtrain_specialist_" + name.replace("-", "_")
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    folder = os.path.dirname(path)
    if folder not in sys.path:
        sys.path.insert(0, folder)            # the plugin's own packages (its generator) import by name
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def list_specialists(root=None):
    root = root or require_source_tree(SPECIALISTS_SRC)
    return sorted(d for d in os.listdir(root) if os.path.exists(os.path.join(root, d, "specialist.yaml")))


def load_specialist(name, root=None):
    """Imports the specialist's plugin (after the ones it `uses:`) and returns its description (validated)."""
    if name in _LOADED:
        return _LOADED[name]
    if not isinstance(name, str) or not NAME.fullmatch(name):
        raise SpecialistError(f"specialist {name!r}: lower-case letters, digits and dashes (a folder of specialists/)")
    root = root or require_source_tree(SPECIALISTS_SRC)
    folder = os.path.join(root, name)
    cfg_path = os.path.join(folder, "specialist.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"no specialist {name!r} (looked for {cfg_path}); known: {list_specialists(root)}")
    with open(cfg_path, encoding="utf-8") as f:
        config = validate_config(name, yaml.safe_load(f) or {}, cfg_path)
    for dep in config.get("uses") or []:
        load_specialist(dep, root)
    plugin = config.get("plugin")
    if plugin:
        _import_plugin(name, os.path.join(folder, plugin))
    _LOADED[name] = Specialist(name, folder, config)
    return _LOADED[name]
