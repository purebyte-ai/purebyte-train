"""The factory: one task.yaml -> a released specialist, stage by stage.

    data -> baselines -> finetune -> eval -> calibrate -> export -> verify -> report -> release

Each stage reads only what earlier stages wrote, writes `<work>/<name>/<version>/<stage>/`, and leaves a done.json with
the hashes of its inputs and outputs, the task's digest and the sha256 of the previous done.json. Before it runs, a
stage checks that whole chain and deletes its own done.json and every later one (factory/chain.py): a task edited after
`data`, a stage re-run with other inputs or a stage that failed never leaves a stale result behind. Each stage fails
LOUDLY when it cannot prove what it claims. The rules each stage enforces are in its docstring.
"""
import json
import os
import time

import numpy as np

from .. import paths, provenance
from ..data import leakage
from ..data.corpus import ByteCorpus
from ..eval import baselines as rivals
from ..eval import calibration
from ..eval.operating_point import BIAS_GRID, OperatingPointError, choose, per_type
from ..eval.spans import forward_windows, score
from ..gpu import RADEON_MAX_WINDOW
from ..runtime import gpu_determinism
from ..train import checkpoint
from ..train.recipe import from_train_block
from ..train.seeds import train_seeds
from ..verticals import SpecialistError, get as get_vertical, load_specialist
from . import criteria as crit
from . import exams as exam_records
from .chain import ORDER, StageError, begin, hash_array, mark_done, read_done, sha_file, stage_dir
from .spec import Watched, fracs, n_entities

SELECT_WINDOWS = 800            # windows of `select` used to choose the operating point
CONTROL_MAX_F1 = 0.10           # the shuffled-spans control must collapse below this span-F1
DECIDE_SEEDS = 6                # three seeds inform, six decide: a run is labeled DECIDES from six seeds


class Context:
    """How the factory runs: device, parallel seeds, GPU backend, logger, windows compared by `verify`."""

    def __init__(self, device="cpu", parallel=1, backend="cpu", log=print, verify_windows=40):
        self.device, self.parallel, self.backend, self.log = device, int(parallel), backend, log
        self.verify_windows = int(verify_windows)


def _check_criteria_frozen(work, spec):
    frozen = read_done(work, spec, "data").get("criteria_sha256")
    now = crit.sha(spec.get("criteria"))
    if frozen != now:
        raise StageError(f"the task's `criteria:` changed after the data stage froze them ({frozen} -> {now}). "
                         f"Pre-registered means written BEFORE: restore them, or give the task a new version")


def criteria_record_path(work, spec):
    return os.path.join(work, spec["name"], str(spec["version"]), "criteria.frozen.json")


def _criteria_record(work, spec):
    """The criteria of a version are frozen by its FIRST successful `data` stage, in a record that no stage deletes or
    rewrites (`criteria.frozen.json` next to the stages). A `data` stage run again on other criteria under the same
    version is refused. Returns the record (None when this version has none yet)."""
    p = criteria_record_path(work, spec)
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as f:
        rec = json.load(f)
    now = crit.sha(spec.get("criteria"))
    if rec["criteria_sha256"] != now:
        raise StageError(f"the pre-registered criteria of {spec['name']} {spec['version']} were frozen on "
                         f"{rec['frozen_at']} (sha256 {rec['criteria_sha256']}) and the task now holds others ({now}): "
                         f"pre-registered means written before; restore them, or give the task a new version")
    return rec


def _freeze_criteria(work, spec):
    """Writes the record of `_criteria_record` if this version has none (once, atomically). Returns it."""
    rec = _criteria_record(work, spec)
    if rec is not None:
        return rec
    rec = dict(name=spec["name"], version=str(spec["version"]), criteria_sha256=crit.sha(spec.get("criteria")),
               criteria=spec.get("criteria"), frozen_at=time.strftime("%Y-%m-%dT%H:%M:%S"),
               task_digest=spec["_digest"])
    p = criteria_record_path(work, spec)
    with open(p + ".tmp", "w", encoding="utf-8") as f:
        json.dump(rec, f, indent=2, default=str)
    os.replace(p + ".tmp", p)
    return rec


def _begin(work, spec, stage):
    """The chain check of every stage (chain.begin), with the clearer message for changed criteria first."""
    if stage != "data" and os.path.exists(os.path.join(work, spec["name"], str(spec["version"]), "data", "done.json")):
        _check_criteria_frozen(work, spec)
    begin(work, spec, stage)


def _partitions(work, spec):
    return np.load(os.path.join(stage_dir(work, spec, "data"), "partitions.npz"))


def _specialist(spec):
    try:
        return load_specialist(spec["specialist"])
    except (SpecialistError, FileNotFoundError) as e:
        raise StageError(str(e))


def _vertical(spec):
    sp = _specialist(spec)
    try:
        return get_vertical(spec["data"]["format"]), sp
    except KeyError as e:
        raise StageError(str(e).strip("'\""))


def _check_data_inputs(work, spec):
    """What the `data` stage checked against the exams must still be what the run uses: the real-window streams (by
    content) and the set of registered exams (by fingerprint content). Otherwise: re-run from `data`."""
    dd = read_done(work, spec, "data")
    rec = dd.get("streams") or {}
    recipe = from_train_block(spec["train"])
    for label in ("real_windows", "guard_windows"):
        p = getattr(recipe, label)
        if not p:
            continue
        try:
            path = paths.expand(p)
        except FileNotFoundError as e:
            raise StageError(str(e))
        if label not in rec or sha_file(path) != rec[label]["sha256"]:
            raise StageError(f"`train.{label}` ({p}) is not the file the data stage checked against the exams (its "
                             f"content changed since): run the stages again from `data`")
    now = leakage.registry(paths.exams_dir())
    before = dd.get("exam_registry") or {}
    if now != before:
        changed = sorted(set(now) ^ set(before) | {k for k in set(now) & set(before) if now[k] != before[k]})
        raise StageError(f"the registered exams changed since the data stage checked this run's data ({changed}): "
                         f"run the stages again from `data`")


def _checkpoint_of(ft, seed):
    """(checkpoint dict, run record) of a seed of `finetune`, refused if its file changed since finetune wrote it."""
    r = next(x for x in ft["runs"] if x["seed"] == seed)
    if sha_file(r["ckpt"]) != r["ckpt_sha256"]:
        raise StageError(f"{r['ckpt']} changed since the finetune stage wrote it: run the stages again from `finetune`")
    return checkpoint.load(r["ckpt"], require_complete=True), r


# ---------------------------------------------------------------------------------------------------------- data
def stage_data(work, spec, ctx):
    """Builds the three disjoint splits with the vertical, and proves they are clean:
    - the core contract on shapes / masks / Y (unless the vertical checks its own);
    - between the splits (train-select, train-test, select-test): what the vertical's `overlap_guard` enforces (whole
      windows, or 120-byte chunks of real content; pbtrain/data/leakage.py says exactly what), plus a recorded
      measurement of shared 64-byte blocks;
    - no shared content block between train / select / real streams / guard windows and ANY registered exam whose role
      is not `dev`; the exams the task declares (with those of its specialist) must be registered. The content hash of
      every stream is recorded: later stages refuse a stream that changed;
    - every declared `data:` key was read by the vertical (a typo cannot silently do nothing);
    - the vertical's own guards.
    Records as warnings: exams of the specialist or of the criteria that were not checked, and exams the corpus build
    did not exclude. Freezes the sha256 of the pre-registered criteria."""
    _criteria_record(work, spec)                  # criteria frozen by an earlier data stage of this version: unchanged
    _begin(work, spec, "data")                    # (checked first: a refused re-freeze deletes nothing)
    v, sp = _vertical(spec)
    d = stage_dir(work, spec, "data")
    cfgd = Watched(spec["data"], "data.")
    cfgd["format"]
    corpus = corpus_path = man = None
    if v.needs_corpus:
        try:
            corpus_path = paths.expand(cfgd["path"])
        except FileNotFoundError as e:
            raise StageError(str(e))
        man = leakage.corpus_manifest(corpus_path)
        if man is not None and (man.get("guard") or {}).get("ok") is False:
            raise StageError(f"the corpus {cfgd['path']} failed its own build guard ({man['guard'].get('failures')}): "
                             f"rebuild it (data/README.md) before training on it")
        corpus = ByteCorpus(corpus_path)
    for k in ("control", "split_policy"):         # keys the core itself acts on (validated in spec.py)
        if k in cfgd:
            cfgd[k]
    if "unit" in (cfgd.get("split_policy") or {}):
        cfgd["split_policy"]["unit"]
    try:
        t = v.build(corpus, cfgd, fracs(cfgd))
        if v.checks_contract:
            t.check(int(cfgd["seq_len"]), v.n_entities)
        v.guards(cfgd, t)
    except ValueError as e:
        raise StageError(f"{type(v).__name__}: {e}")
    unread = cfgd.unread()
    if unread:
        raise StageError(f"declared and never read by the vertical `{v.data_format}`: {unread}. Remove them or fix "
                         f"the typo: a key that does nothing must not look like it does")
    if n_entities(spec) != v.n_entities:
        raise StageError(f"output_schema.labels has {n_entities(spec)} type(s), the vertical builds {v.n_entities}")
    arrays = dict(Xtr=t.train[0], Ytr=t.train[1], Str=t.train[2], Xse=t.select[0], Yse=t.select[1],
                  Sse=t.select[2], Xte=t.test[0], Yte=t.test[1], Ste=t.test[2])
    try:
        overlap = leakage.check_splits(t, v.overlap_guard)
    except leakage.LeakError as e:
        raise StageError(str(e))
    sets = {"train": t.train[0], "select": t.select[0]}
    streams = {}
    recipe = from_train_block(spec["train"])
    for label, p in (("real_windows", recipe.real_windows), ("guard_windows", recipe.guard_windows)):
        if p:
            try:
                path = paths.expand(p)
            except FileNotFoundError as e:
                raise StageError(str(e))
            with np.load(path) as z:
                if "X" not in z.files:
                    raise StageError(f"`train.{label}: {p}` holds no `X` array of windows")
                sets[label] = z["X"]
            streams[label] = dict(path=path, sha256=sha_file(path))
    task_exams = list(spec.get("exams") or [])
    required = sorted(set(task_exams) | (set(sp.exams) if task_exams else set()))
    try:
        exams = leakage.check_against_exams(sets, paths.exams_dir(), required=required)
    except (leakage.LeakError, FileNotFoundError) as e:
        raise StageError(str(e))
    registered = set(exams["exams"])
    warnings = [exams["warning"]] if exams.get("warning") else []
    unchecked = [x for x in sp.exams if x not in registered]
    if unchecked:
        warnings.append(f"the specialist `{spec['specialist']}` lists the exam(s) {unchecked}, whose fingerprints are "
                        f"not registered: their content was NOT compared with this data, and a release refuses this run")
    outside = [x for x in crit.exams_named(spec.get("criteria")) if x not in task_exams]
    if outside:
        chk = [x for x in outside if x in registered]
        nochk = [x for x in outside if x not in registered]
        warnings.append(f"the pre-registered criteria read the exam(s) {outside}, which the task's `exams:` does not "
                        f"require" + (f"; registered, so checked: {chk}" if chk else "")
                        + (f"; NOT registered, so their content was never compared with the training data: {nochk}"
                           if nochk else ""))
    gaps = leakage.corpus_exclusion_gaps(man, exams)
    if gaps:
        warnings.append("the corpus build did not exclude every exam registered now (" + "; ".join(gaps) + "); every "
                        "window was still compared with every registered exam")
    for w in warnings:
        ctx.log(f"    WARNING: {w}")
    probe = rivals.unigram_probe(t.train[0], t.train[1], t.test[0], t.test[1])
    S = t.train[2].astype(np.int16)
    inside = (S > 0) & (S != 255)
    if v.n_entities > 1:                           # a span starts where a typed run starts
        per = (inside & np.concatenate([np.ones((len(S), 1), bool), S[:, 1:] != S[:, :-1]], axis=1)).sum(axis=1)
    else:
        per = (np.diff(inside.astype(np.int8), axis=1) == 1).sum(axis=1) + inside[:, 0]
    np.savez(os.path.join(d, "partitions.npz"), **arrays)
    return mark_done(work, spec, "data", dict(
        vertical=v.data_format, specialist=spec["specialist"], manifest=v.manifest(cfgd, t),
        corpus=(dict(path=cfgd.get("path"), sha256=sha_file(corpus_path, 16), build_manifest=man)
                if corpus_path else None),
        fracs=fracs(cfgd), n_train=len(t.train[0]), n_eval=len(t.test[0]), seq_len=int(cfgd["seq_len"]),
        positives=float(np.mean(t.train[1])), spans_per_window={str(int(k)): int((per == k).sum())
                                                                for k in np.unique(per)},
        unigram_probe=probe, split_overlap=overlap, exam_leak_check=exams, required_exams=required, streams=streams,
        exam_registry={k: dict(role=x["role"], fp_sha256=x["fp_sha256"]) for k, x in exams["exams"].items()},
        warnings=warnings, criteria_sha256=crit.sha(spec.get("criteria")),
        criteria_frozen_at=_freeze_criteria(work, spec)["frozen_at"],
        partitions_sha256=sha_file(os.path.join(d, "partitions.npz")),
        hash_train=hash_array(t.train[0]), hash_test=hash_array(t.test[0])))


# ----------------------------------------------------------------------------------------------------- baselines
def stage_baselines(work, spec, ctx):
    """The classic rival of the vertical, tuned on `select`, scored on `test` exactly like the model: the gate."""
    _begin(work, spec, "baselines")
    v, _ = _vertical(spec)
    z = _partitions(work, spec)
    r = v.span_rival((z["Xse"], z["Yse"], z["Sse"]), (z["Xte"], z["Yte"], z["Ste"]))
    if r is None:
        raise StageError(f"the vertical `{v.data_format}` provides no classic rival: the gate could not be passed")
    with open(os.path.join(stage_dir(work, spec, "baselines"), "baselines.json"), "w", encoding="utf-8") as f:
        json.dump(r, f, indent=2)
    return mark_done(work, spec, "baselines", dict(rival_f1=r["span_f1"], rival=r.get("name", "classic"), detail=r))


# ------------------------------------------------------------------------------------------------------ finetune
def stage_finetune(work, spec, ctx):
    """One run per seed + the shuffled-spans control, trained (or reused from the content cache); for each seed the
    operating bias is chosen on `select` (first 800 windows) and the model is scored on `test`. First it checks that the
    real-window streams and the registered exams are still those the data stage checked. The checkpoint is saved
    BEFORE the bias is chosen (a policy error must never destroy a training). The released seed is the best on
    `select` unless `train.released_seed` forces one: then the rule, the seed the rule would have chosen and the
    task's `released_seed_reason` (or a warning that it gives none) are recorded. Records where and with what the seeds
    were trained (device, backend, software, trainer digest, code commit)."""
    _begin(work, spec, "finetune")
    _check_data_inputs(work, spec)
    d = stage_dir(work, spec, "finetune")
    npz = os.path.join(stage_dir(work, spec, "data"), "partitions.npz")
    tr = spec["train"]
    try:
        base = paths.expand(spec["base_checkpoint"]) if spec.get("base_checkpoint") else None
        recipe = from_train_block(tr, base=base, n_entities=n_entities(spec))
        resolved = {k: paths.expand(getattr(recipe, k)) for k in ("real_windows", "ngram_init", "guard_windows")
                    if getattr(recipe, k)}
    except (FileNotFoundError, ValueError) as e:
        raise StageError(str(e))
    recipe = recipe.replace(**resolved)
    if ctx.backend == "radeon" and int(spec["data"]["seq_len"]) > RADEON_MAX_WINDOW:
        raise StageError(f"the radeon backend is validated up to {RADEON_MAX_WINDOW}-byte windows (1,024 crashed) and "
                         f"this task's `seq_len` is {spec['data']['seq_len']}: use the stock backend")
    if ctx.device != "cpu":
        gpu_determinism()
    if recipe.experimental_in_use():
        ctx.log(f"    experimental options in use (recorded in done.json): {recipe.experimental_in_use()}")
    seeds = [int(s) for s in tr["seeds"]]
    try:
        runs = train_seeds(npz, recipe, seeds, control_seed=seeds[0], device=ctx.device, parallel=ctx.parallel,
                           cache_dir=os.path.join(d, "_seed_cache"), backend=ctx.backend, log=ctx.log)
    except ValueError as e:
        raise StageError(f"training refused: {e}")
    z = _partitions(work, spec)
    oe = spec["output_schema"]
    k = n_entities(spec)
    out = []
    for s in seeds:
        ck = checkpoint.load(runs[(s, False)])
        m, th, ch = checkpoint.rebuild(ck, ctx.device)
        dest = os.path.join(d, f"seed_{s}", "final.pt")
        meta = dict(seed=s, init_seed=recipe.init_seed, provenance=ck.get("provenance"),
                    **({"guard": ck["guard"]} if ck.get("guard") else {}))
        checkpoint.save(dest, m, th, ch, bias=None, complete=False, **meta)
        lp, doc = forward_windows(m, th, ch, z["Xse"][:SELECT_WINDOWS], device=ctx.device)
        try:
            bias, best, sweep = choose(lambda b: score(lp, doc, z["Sse"][:SELECT_WINDOWS], z["Yse"][:SELECT_WINDOWS],
                                                       b, k)["gated"],
                                       BIAS_GRID, oe.get("min_select_recall"), oe.get("bias_policy", "conservative"))
        except OperatingPointError as e:
            raise StageError(str(e))
        bias_vec = None
        if oe.get("per_type_bias") and k > 1:          # one coordinate-ascent pass around the scalar bias
            bias_vec, _ = per_type(lambda vec: score(lp, doc, z["Sse"][:SELECT_WINDOWS], z["Yse"][:SELECT_WINDOWS],
                                                     vec, k)["gated"]["span_f1"], bias, k)
        lpt, doct = forward_windows(m, th, ch, z["Xte"], device=ctx.device)
        r = score(lpt, doct, z["Ste"], z["Yte"], bias_vec if bias_vec is not None else bias, k)
        checkpoint.save(dest, m, th, ch, bias=bias, bias_vec=bias_vec, complete=True, **meta)
        point = next(x for x in sweep if x.get("bias") == bias)
        out.append(dict(seed=s, select_f1=round(best, 4), sweep_bias=sweep, recall_select=point["recall"],
                        precision_select=point["precision"],
                        floor_met=not any(x.get("floor_met") is False for x in sweep),
                        flat_sweep=any("flat sweep" in str(x.get("warning", "")) for x in sweep),
                        test_f1=r["gated"]["span_f1"], test_recall=r["gated"]["recall"],
                        test_precision=r["gated"]["precision"], tag_only_f1=r["tag_only"]["span_f1"],
                        doc_acc=r["doc_acc"], bias=bias, bias_vec=bias_vec, ckpt=dest, ckpt_sha256=sha_file(dest),
                        params=m.n_params(), trained=ck.get("provenance"),
                        guard=ck.get("guard") and {kk: ck["guard"].get(kk) for kk in ("mode", "chosen_step",
                                                                                         "rollbacks")}))
    ckc = checkpoint.load(runs[(seeds[0], True)])
    mc, thc, chc = checkpoint.rebuild(ckc, ctx.device)
    lpc, docc = forward_windows(mc, thc, chc, z["Xte"], device=ctx.device)
    # the control is read at the more favorable of two points (0 and +3): a leak must not hide behind the bias
    control = max(score(lpc, docc, z["Ste"], z["Yte"], b, k)["gated"]["span_f1"] for b in (0.0, 3.0))
    on_select = max(out, key=lambda r: r["select_f1"])
    forced, reason, warnings = tr.get("released_seed"), tr.get("released_seed_reason"), []
    best = on_select if forced is None else next(r for r in out if r["seed"] == int(forced))
    if forced is not None and best["seed"] != on_select["seed"]:
        warnings.append(f"`train.released_seed: {forced}` releases seed {forced}; the factory's rule (the best on "
                        f"`select`) would release seed {on_select['seed']}; "
                        + (f"the task's reason: {reason}" if reason else "the task gives no `released_seed_reason`"))
    for w in warnings:
        ctx.log(f"    WARNING: {w}")
    return mark_done(work, spec, "finetune", dict(
        runs=out, control_f1=control, control_ckpt=runs[(seeds[0], True)], released_seed=best["seed"],
        released_ckpt=best["ckpt"], forced_seed=forced is not None,
        released_seed_rule="forced by train.released_seed" if forced is not None else "the best on select",
        released_seed_reason=reason, select_best_seed=on_select["seed"], warnings=warnings,
        spread=float(np.std([r["test_f1"] for r in out])) if len(out) > 1 else 0.0,
        recipe=recipe.as_dict(), experimental=recipe.experimental_in_use(),
        training=dict(provenance.environment(ctx.device, ctx.backend), code_commit=provenance.code_commit(),
                      parallel=ctx.parallel),
        inputs={k: dict(path=v, sha256=sha_file(v, 16)) for k, v in dict(resolved, base=base).items() if v}))


# ---------------------------------------------------------------------------------------------------------- eval
def synthetic_metrics(ft, bl):
    f1 = [r["test_f1"] for r in ft["runs"]]
    return dict(test_f1_mean=float(np.mean(f1)), test_f1_min=float(np.min(f1)), test_f1_max=float(np.max(f1)),
                spread=ft["spread"], control_f1=ft["control_f1"], n_seeds=len(f1),
                recall_select_min=float(min(r["recall_select"] for r in ft["runs"])),
                doc_acc_mean=float(np.mean([r["doc_acc"] for r in ft["runs"]])),
                gate_margin=float(np.mean(f1) - bl["rival_f1"]))


def stage_eval(work, spec, ctx):
    """The delivery gate on the synthetic test. A model is NOT delivered when: the control does not collapse (span-F1
    above 0.10: the model finds positions without the label, a positional leak); the released seed misses the
    declared recall floor on `select`; the released seed's bias sweep is flat (it has no operating point); or the mean
    of the seeds does not beat the classic rival by MORE than the spread between seeds. The done.json is written in
    every case (`gate_passes` false with the reasons in `refused`), so a failed re-run can never leave an earlier pass
    behind. The run is labeled DECIDES with `decide: true` and six seeds or more, INFORMS otherwise (validation accepts
    `decide: true` from three seeds: three inform, six decide). Records the verdict of every pre-registered criterion
    (exams: the results recorded so far)."""
    _begin(work, spec, "eval")
    ft, bl = read_done(work, spec, "finetune"), read_done(work, spec, "baselines")
    m = synthetic_metrics(ft, bl)
    refused = []
    if ft["control_f1"] > CONTROL_MAX_F1:
        refused.append(dict(by="control", reason=f"the control with shuffled spans gives {ft['control_f1']:.4f}; it "
                                                 f"must be ~0: a positional leak"))
    floor = spec["output_schema"].get("min_select_recall")
    rel = next(r for r in ft["runs"] if r["seed"] == ft["released_seed"])
    if floor is not None and rel["recall_select"] < floor:
        refused.append(dict(by="recall_floor", reason=f"the released seed {rel['seed']} has recall "
                                                      f"{rel['recall_select']:.4f} on `select`, below "
                                                      f"`min_select_recall: {floor}`"))
    if rel.get("flat_sweep"):
        refused.append(dict(by="flat_sweep", reason=f"the bias sweep of the released seed {rel['seed']} on `select` "
                                                    f"is flat: the bias does not move its output, so it has no "
                                                    f"operating point (an instrument failure, not a result)"))
    passes, reason = rivals.gate(m["test_f1_mean"], bl["rival_f1"], max(m["spread"], 1e-6))
    if not passes:
        refused.append(dict(by="gate", reason=reason))
    flat = [r["seed"] for r in ft["runs"] if r.get("flat_sweep") and r["seed"] != rel["seed"]]
    notes = [f"flat bias sweep on `select` for seed(s) {flat} (recorded; their test F1 enters the mean and the "
             f"spread)"] if flat else []
    declared = bool(spec["train"].get("decide"))
    decide = declared and m["n_seeds"] >= DECIDE_SEEDS
    verdicts = crit.evaluate(spec.get("criteria"), synthetic=m, exams=exam_records.results(work, spec))
    r = mark_done(work, spec, "eval", dict(metrics=m, gate_passes=not refused, gate_reason=reason,
                                           margin_passes=bool(passes), refused=refused, notes=notes, decide=decide,
                                           scope=(f"DECIDES ({m['n_seeds']} seeds)" if decide else
                                                  f"INFORMS ({m['n_seeds']} seed(s): three seeds inform, six decide)"
                                                  if declared else
                                                  f"INFORMS ({m['n_seeds']} seed(s), `decide` not declared)"),
                                           criteria=verdicts, criteria_summary=crit.summary(verdicts)))
    if refused:
        first = refused[0]
        raise StageError(("GATE NOT PASSED: " if first["by"] == "gate" else "NOT DELIVERED: ") + first["reason"]
                         + ". Not delivered.")
    return r


# ----------------------------------------------------------------------------------------------------- calibrate
def stage_calibrate(work, spec, ctx):
    """Span models: the operating point is the Viterbi bias chosen on `select` in finetune (it travels in the GGUF,
    per type when `per_type_bias` gave a vector). The document head gates the spans by its argmax, which a temperature
    cannot change, and the file carries no temperature: it stays 1.0. Measured and recorded on `select` (first 800
    windows): the document head's expected calibration error and the estimator's noise floor (median, p95), for the
    probabilities the runtime reports."""
    _begin(work, spec, "calibrate")
    ft = read_done(work, spec, "finetune")
    ck, _ = _checkpoint_of(ft, ft["released_seed"])
    if ctx.device != "cpu":
        gpu_determinism()
    m, th, ch = checkpoint.rebuild(ck, ctx.device)
    z = _partitions(work, spec)
    doc = calibration.measure(m, ch, z["Xse"][:SELECT_WINDOWS], z["Yse"][:SELECT_WINDOWS], device=ctx.device)
    cal = dict(temperature=1.0, operating_bias=float(ck["bias"]), operating_bias_vec=ck.get("bias_vec"),
               fitted_on="select", labels=list(spec["output_schema"]["labels"]), document_head_ece=doc,
               note="the document head gates by its argmax, which a temperature does not change; the file carries none")
    with open(os.path.join(stage_dir(work, spec, "calibrate"), "calibration.json"), "w", encoding="utf-8") as f:
        json.dump(cal, f, indent=2)
    return mark_done(work, spec, "calibrate", cal)


# -------------------------------------------------------------------------------------------------------- export
def export_checkpoint(ck_path, out_path, spec, bias=None):
    """One checkpoint -> one GGUF declaring what the specialist says (profile, window, labels; the n-gram precision
    applies to a model that has n-gram tables). A profile or window that specialist.yaml does not set comes from the
    vertical's `export_meta`."""
    from ..export.pbformat import export
    v, sp = _vertical(spec)
    ex = sp.export
    defaults = v.export_meta(spec) or {}
    ck = checkpoint.load(ck_path, require_complete=True)
    m, th, ch = checkpoint.rebuild(ck)
    m.eval()
    bits = ex.get("ngram_bits") if m.ngram is not None else None
    op = bias if bias is not None else (ck.get("bias_vec") or float(ck["bias"]))
    info = dict(version=str(spec["version"]), license=ex.get("license"), **{"license.name": ex.get("license_name")})
    return export(out_path, m, [th, ch], gating={0: 1}, labels=ex.get("labels") or spec["output_schema"]["labels"],
                  operating_bias=op, profile=ex.get("profile", defaults.get("profile")),
                  window=ex.get("window", defaults.get("window")), ngram_bits=bits,
                  name=sp.config.get("public_name", spec["name"]), info=info)


def stage_export(work, spec, ctx):
    """The released seed -> GGUF (spec/FORMAT.md) with full coverage; the release stage exports ensemble members."""
    _begin(work, spec, "export")
    ft = read_done(work, spec, "finetune")
    _, run = _checkpoint_of(ft, ft["released_seed"])
    gp = os.path.join(stage_dir(work, spec, "export"), "model.gguf")
    try:
        info = export_checkpoint(run["ckpt"], gp, spec)
    except ValueError as e:
        raise StageError(str(e))
    return mark_done(work, spec, "export", dict(gguf=gp, sha256=sha_file(gp), seed=run["seed"], ckpt=run["ckpt"],
                                                ckpt_sha256=run["ckpt_sha256"], mb=round(info["bytes"] / 2 ** 20, 3),
                                                **info))


# -------------------------------------------------------------------------------------------------------- verify
def stage_verify(work, spec, ctx):
    """The runtime must give the SAME spans as Python, gating included, on `ctx.verify_windows` test windows: half of
    them windows where Python finds a span, the rest windows where it finds none (a comparison of empty outputs only
    would prove nothing, so no span at all is an error). Uses the engine's `model_dump` when PUREBYTE_MODEL_DUMP points
    to it, with the task's `engine.threads`, else the NumPy reference (slow). Spans are compared exactly."""
    _begin(work, spec, "verify")
    from .verify import compare_runtime
    ex, ft = read_done(work, spec, "export"), read_done(work, spec, "finetune")
    if sha_file(ex["gguf"]) != ex["sha256"]:
        raise StageError(f"{ex['gguf']} changed since the export stage wrote it: run the stages again from `export`")
    z = _partitions(work, spec)
    ck, _ = _checkpoint_of(ft, ft["released_seed"])
    threads = (spec.get("engine") or {}).get("threads")
    res = compare_runtime(ex["gguf"], ck, z["Xte"], n=ctx.verify_windows, threads=threads)
    if threads and res["engine_threads"] is None:
        ctx.log(f"    note: `engine.threads: {threads}` applies to the engine's model_dump; this verify used the NumPy "
                f"reference in this process (recorded)")
    if res["windows_with_spans"] == 0:
        raise StageError(f"none of the {res['n']} compared test windows holds a span in Python: the comparison would "
                         f"prove nothing (a model that emits no span at its operating point)")
    if res["equal"] != res["n"]:
        raise StageError(f"the runtime does not reproduce the spans of Python: {res['equal']}/{res['n']} windows equal "
                         f"({res['runtime']})")
    return mark_done(work, spec, "verify", res)


# -------------------------------------------------------------------------------------------------------- report
def stage_report(work, spec, ctx):
    """The run report, from the done.json files and the exam results recorded so far: every pre-registered criterion,
    exams included, gets its verdict here (recorded in this done.json and copied into the release MANIFEST)."""
    _begin(work, spec, "report")
    from .report import write_report
    p, verdicts = write_report(work, spec)
    return mark_done(work, spec, "report", dict(report=p, report_sha256=sha_file(p),
                                                gate_passes=read_done(work, spec, "eval")["gate_passes"],
                                                criteria=verdicts, criteria_summary=crit.summary(verdicts),
                                                exam_records=exam_records.summary(work, spec)))


# ------------------------------------------------------------------------------------------------------- release
def stage_release(work, spec, ctx):
    """An IMMUTABLE internal release (release/package.py): refused unless the chain holds, the gate passed, the streams
    and registered exams are those the data stage checked, and every exam of the specialist was checked. The GGUF of
    every ensemble member, the report, the task, every done.json and a MANIFEST with full sha256s, the code commit,
    seeds, where and with what the seeds were trained, the verdicts and the warnings. Never overwritten. The public
    bundle is made from it by `pbtrain release TASK --public --semver X.Y.Z`."""
    _begin(work, spec, "release")
    _check_data_inputs(work, spec)
    from ..release.package import internal_release
    return mark_done(work, spec, "release", internal_release(work, spec))


STAGES = {"data": stage_data, "baselines": stage_baselines, "finetune": stage_finetune, "eval": stage_eval,
          "calibrate": stage_calibrate, "export": stage_export, "verify": stage_verify, "report": stage_report,
          "release": stage_release}


def run(spec, stages, work=None, ctx=None):
    """Runs `stages` (names, or ["all"]) in order. Returns {stage: done.json}."""
    ctx = ctx or Context()
    work = work or paths.runs_dir()
    todo = ORDER if stages == ["all"] else stages
    for s in todo:
        if s not in STAGES:
            raise StageError(f"unknown stage {s!r}; stages: {ORDER}")
    out = {}
    for s in todo:
        t0 = time.time()
        ctx.log(f"[{spec['name']} {spec['version']}] {s} ...")
        out[s] = STAGES[s](work, spec, ctx)
        ctx.log(f"[{spec['name']} {spec['version']}] {s} done in {time.time() - t0:.0f}s")
    return out
