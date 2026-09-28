"""SpanRecipe: every argument of the span trainer, in one place, with its status.

A recipe is DATA: the `train:` block of a task.yaml translates 1:1 into a SpanRecipe (`from_train_block`). The defaults
are the recommended secrets-code recipe (`nano`, 12,000 steps, batch 16, lr 3e-4, warmup 30, wd 0.05, clip 1.0,
lam 0.5, beta2 0.95). A task must still state `steps`, `lr`, `batch`, `size` and `seeds` explicitly (the factory refuses
it otherwise), so a changed default can never silently change an existing recipe.

Status of each key:
  stable        part of a released or recommended recipe (secrets-bin 1.0.0 trains with `out_lr_mult`, secrets-code
                1.0.0 with `real_mode: spans`)
  experimental  implemented and OFF by default; no released specialist uses it. Using one is recorded in the run's
                done.json and report.
Options that were tried and did not help are not implemented (weight EMA, SWA, a ternary-threshold multiplier, other
poolings of the document head); a task that declares one is refused with the reason.

`from_train_block` also checks the TYPE and the range of every value (a string where a number is expected, a zero
clip, a duplicated seed... are refused before anything is built).
"""
import math
from dataclasses import dataclass, fields

GUARD_MODES = ("select", "rollback")
REAL_MODES = ("neg", "distill", "spans")
ADAPTERS = ("none", "heads", "scales", "lora")
DIRECTIONS = {None: "causal", "causal": "causal", "bidirectional": "hydra"}

# key in the task's `train:` block -> (SpanRecipe field, status, what it does)
TRAIN_KEYS = {
    "steps":         ("steps", "stable", "optimizer steps"),
    "batch":         ("batch", "stable", "synthetic windows per step"),
    "lr":            ("lr", "stable", "peak learning rate of the WSD schedule"),
    "warmup":        ("warmup", "stable", "linear warmup steps"),
    "wd":            ("wd", "stable", "AdamW weight decay, matrices only"),
    "clip":          ("clip", "stable", "gradient norm clip"),
    "lam":           ("lam", "stable", "weight of the document (choice) head in the loss"),
    "shuffle_spans": ("shuffle_spans", "stable", "the control: spans shuffled between windows (span-F1 must be ~0)"),
    "span_weight":   ("span_weight", "experimental", "weight (>= 1) of span bytes in the tag loss (1 = flat CE); "
                                                     "extraction"),
    "init_seed":     ("init_seed", "experimental", "seed of the init, apart from the data order (`seeds`)"),
    "out_lr_mult":   ("out_lr_mult", "stable", "lr multiplier of every out_proj matrix (secrets-bin 1.0.0: 0.25)"),
    "beta2":         ("beta2", "experimental", "AdamW beta2"),
    "real_windows":  ("real_windows", "stable", ".npz of real windows appended to every batch"),
    "real_frac":     ("real_frac", "stable", "real windows per step, as a fraction of `batch`"),
    "real_mode":     ("real_mode", "stable", "neg | spans (stable) | distill (experimental)"),
    "distill_alpha": ("distill_alpha", "experimental", "weight of KL(teacher || student) in `real_mode: distill`"),
    "ngram_init":    ("ngram_init", "experimental", "checkpoint whose n-gram tables and projection seed the model"),
    "freeze_ngram":  ("freeze_ngram", "experimental", "the n-gram tables and projection get no gradient"),
    "adapter":       ("adapter", "experimental", "frozen body + adapter: heads | scales | lora (needs a base)"),
    "lora_rank":     ("lora_rank", "experimental", "rank of `adapter: lora`"),
    "guard_windows": ("guard_windows", "experimental", "tail guard: .npz of real negative windows"),
    "guard_every":   ("guard_every", "experimental", "tail guard: steps between two guard points"),
    "guard_mode":    ("guard_mode", "experimental", "tail guard: select | rollback"),
    "guard_k":       ("guard_k", "experimental", "tail guard: rollback when proxy > k x max(lowest seen, 1)"),
}
# policy keys: read by the factory, not arguments of the trainer
POLICY_KEYS = {"seeds", "mode", "size", "decide", "released_seed", "released_seed_reason"}
REQUIRED_KEYS = ("steps", "lr", "batch", "size", "seeds")
# the type and range of every value: (kind, low, low included, high, high included); kinds int | float | bool | str
VALUE_RULES = {
    "steps": ("int", 1, True, None, False), "batch": ("int", 1, True, None, False),
    "warmup": ("int", 0, True, None, False), "init_seed": ("int", None, False, None, False),
    "lora_rank": ("int", 1, True, None, False), "guard_every": ("int", 1, True, None, False),
    "lr": ("float", 0.0, False, None, False), "wd": ("float", 0.0, True, None, False),
    "clip": ("float", 0.0, False, None, False), "lam": ("float", 0.0, True, None, False),
    "beta2": ("float", 0.0, False, 1.0, False), "out_lr_mult": ("float", 0.0, False, None, False),
    "span_weight": ("float", 1.0, True, None, False), "real_frac": ("float", 0.0, False, 1.0, True),
    "distill_alpha": ("float", 0.0, False, None, False), "guard_k": ("float", 1.0, True, None, False),
    "shuffle_spans": ("bool",), "freeze_ngram": ("bool",), "decide": ("bool",),
    "real_windows": ("str",), "ngram_init": ("str",), "guard_windows": ("str",), "real_mode": ("str",),
    "adapter": ("str",), "guard_mode": ("str",), "size": ("str",), "released_seed_reason": ("str",),
}
NOT_SUPPORTED = {
    "ema": "an EMA of the weights did not reduce false alarms on real data; not implemented",
    "swa_frac": "averaging the last steps (SWA) did not reduce false alarms on real data; not implemented",
    "tern_thr": "a lower ternary threshold did not hold at six seeds; not implemented as a training option",
    "pooling": "the span trainer's document head always pools with `max` (mean pooling dilutes short spans)",
}


@dataclass
class SpanRecipe:
    """Arguments of `pbtrain.train.loop.train_spans`. Defaults = the recommended secrets-code recipe."""
    size: str = "nano"
    direction: str = "causal"          # causal | hydra | bimamba
    ternary: bool = True
    n_entities: int = 1
    steps: int = 12000
    batch: int = 16
    lr: float = 3e-4
    warmup: int = 30
    wd: float = 0.05
    clip: float = 1.0
    lam: float = 0.5
    beta2: float = 0.95
    out_lr_mult: float = 1.0
    span_weight: float = 1.0
    shuffle_spans: bool = False
    init_seed: int = None
    base: str = None                   # body checkpoint (a pretraining or a factory final.pt); resolved path
    real_windows: str = None           # .npz of real windows; resolved path
    real_frac: float = 0.0
    real_mode: str = None
    distill_alpha: float = 1.0
    ngram_init: str = None
    freeze_ngram: bool = False
    adapter: str = "none"
    lora_rank: int = None
    guard_windows: str = None
    guard_every: int = 0
    guard_mode: str = None
    guard_k: float = 3.0

    def replace(self, **kw):
        d = self.as_dict()
        d.update(kw)
        return SpanRecipe(**d)

    def as_dict(self):
        return {f.name: getattr(self, f.name) for f in fields(self)}

    def experimental_in_use(self):
        """The experimental options this recipe switches on (recorded in the run's done.json and report)."""
        default = SpanRecipe()
        on = []
        for key, (name, status, _) in TRAIN_KEYS.items():
            if status == "experimental" and getattr(self, name) != getattr(default, name):
                on.append(key)
        if self.real_mode == "distill":
            on.append(f"real_mode: {self.real_mode}")
        return on


class RecipeError(ValueError):
    pass


def _check_value(key, v):
    """Refuses a value of the wrong type or out of range (a string where a number is expected, a zero clip...)."""
    rule = VALUE_RULES.get(key)
    if rule is None:
        return
    field = TRAIN_KEYS[key][0] if key in TRAIN_KEYS else None
    if v is None and field is not None and getattr(SpanRecipe, field, 0) is None:
        return                                            # an option that is off by default may be written as null
    kind = rule[0]
    if kind == "bool":
        if v not in (True, False):
            raise RecipeError(f"`train.{key}: {v!r}` must be true or false")
        return
    if kind == "str":
        if not isinstance(v, str) or not v:
            raise RecipeError(f"`train.{key}: {v!r}` must be a non-empty string")
        return
    if isinstance(v, str):
        hint = " (YAML reads a number without a dot such as 3e-4 as text: write 0.0003)" if kind == "float" else ""
        raise RecipeError(f"`train.{key}: {v!r}` is a string; it must be a number{hint}")
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (kind == "int" and not isinstance(v, int)):
        raise RecipeError(f"`train.{key}: {v!r}` must be {'a whole number' if kind == 'int' else 'a number'}")
    if not math.isfinite(v):
        raise RecipeError(f"`train.{key}: {v!r}` must be finite")
    _, lo, lo_in, hi, hi_in = rule
    if lo is not None and (v < lo or (v == lo and not lo_in)):
        raise RecipeError(f"`train.{key}: {v}` must be {'>=' if lo_in else '>'} {lo}")
    if hi is not None and (v > hi or (v == hi and not hi_in)):
        raise RecipeError(f"`train.{key}: {v}` must be {'<=' if hi_in else '<'} {hi}")


def check_seeds(tr):
    """`seeds`: a non-empty list of DISTINCT whole numbers; `released_seed` one of them; `released_seed_reason` only
    next to a `released_seed`."""
    seeds = tr.get("seeds")
    if not isinstance(seeds, list) or not seeds:
        raise RecipeError(f"`train.seeds: {seeds!r}` must be a non-empty list of whole numbers")
    if any(isinstance(s, bool) or not isinstance(s, int) for s in seeds):
        raise RecipeError(f"`train.seeds: {seeds}` must hold whole numbers only")
    dup = sorted({s for s in seeds if seeds.count(s) > 1})
    if dup:
        raise RecipeError(f"`train.seeds: {seeds}` repeats {dup}: a repeated seed trains the same run twice, counts it "
                          f"twice and shrinks the spread between seeds that the gate reads")
    forced = tr.get("released_seed")
    if forced is not None and (isinstance(forced, bool) or not isinstance(forced, int) or forced not in seeds):
        raise RecipeError(f"`train.released_seed: {forced!r}` is not one of the trained seeds {seeds}")
    if tr.get("released_seed_reason") is not None and forced is None:
        raise RecipeError("`train.released_seed_reason` without `released_seed`: it would act on nothing")
    return seeds


def from_train_block(tr, base=None, n_entities=1):
    """`train:` block of a task (a dict) -> SpanRecipe. Unknown and unsupported keys are refused with the reason; so
    are values of the wrong type or out of range, repeated seeds and an unknown `size`."""
    if not isinstance(tr, dict):
        raise RecipeError("`train:` must be a mapping")
    bad = sorted(set(tr) & set(NOT_SUPPORTED))
    if bad:
        raise RecipeError("; ".join(f"`{k}`: {NOT_SUPPORTED[k]}" for k in bad))
    extra = sorted(set(tr) - set(TRAIN_KEYS) - POLICY_KEYS)
    if extra:
        raise RecipeError(f"unknown keys in `train:`: {extra}. Valid: {sorted(set(TRAIN_KEYS) | POLICY_KEYS)}. "
                          f"(A typo must not fall back to a default value.)")
    mode = tr.get("mode")
    if mode not in DIRECTIONS:
        raise RecipeError(f"unknown train.mode: {mode!r}; valid: causal, bidirectional")
    for k, v in tr.items():
        _check_value(k, v)
    if "seeds" in tr:
        check_seeds(tr)
    from ..model.config import PRESETS
    size = tr.get("size", "nano")
    if size not in PRESETS:
        raise RecipeError(f"unknown `train.size: {size}`; known: {sorted(PRESETS)}")
    for key, known in (("real_mode", REAL_MODES), ("adapter", ADAPTERS), ("guard_mode", GUARD_MODES)):
        if tr.get(key) is not None and tr[key] not in known:
            raise RecipeError(f"`train.{key}: {tr[key]!r}`; the valid ones are {known}")
    for needs, keys in (("real_windows", ("real_frac", "real_mode", "distill_alpha")),
                        ("guard_mode", ("guard_windows", "guard_every", "guard_k"))):
        declared = [k for k in keys if tr.get(k) is not None]
        if declared and tr.get(needs) is None:
            raise RecipeError(f"`train.{', '.join(declared)}` without `{needs}`: declared with no effect")
    if tr.get("lora_rank") is not None and tr.get("adapter") != "lora":
        raise RecipeError("`train.lora_rank` without `adapter: lora`: declared with no effect")
    kw = {TRAIN_KEYS[k][0]: tr[k] for k in tr if k in TRAIN_KEYS}
    return SpanRecipe(size=size, direction=DIRECTIONS[mode], base=base, n_entities=n_entities, **kw)


def missing_required(tr):
    return [k for k in REQUIRED_KEYS if k not in tr]
