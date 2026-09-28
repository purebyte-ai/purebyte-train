"""Releases: the immutable internal release of a factory run, and the PUBLIC bundle made from it.

internal_release (factory stage `release`):
    $PUREBYTE_WORK/specialists/<name>/<version>/
        model.gguf, member_<i>.gguf ...     the released seed and the other ensemble members
        report.md, task.yaml, done_<stage>.json
        MANIFEST.json                       the task's sha256 and digest, the code commit at release, every file's
                                            sha256, members, vote, seeds and how the released one was chosen, the
                                            operating point (vector when per type), the synthetic metrics and the gate,
                                            every criterion's verdict (exams included), the exam records, the warnings
                                            of the run, where and with what the seeds were trained, the release machine
    Refused unless the gate passed and every exam the specialist lists was checked by the data stage; the number of
    members must equal the specialist's `release.ensemble`. Built in a temporary folder and moved into place in one
    step: a failure leaves nothing behind. Never overwritten; files are made read-only.

public_bundle (`pbtrain release TASK --public --semver X.Y.Z`):
    $PUREBYTE_WORK/releases/<public name>-<semver>/
        purebyte-<name>-<semver>.gguf (+ -e1.gguf, -e2.gguf for ensemble members), one .sha256 per file
        MODEL_CARD.md                        from the specialist's card template + the recorded numbers WITH conditions
        models.json.fragment                 the catalog entry (name, version, urls, sha256, size) for models/models.json
    plus one line appended to $PUREBYTE_WORK/releases/MAPPING.jsonl (private, outside git): public name/version ->
    internal run, task sha, seeds, sha256s. Every file of the internal release is checked against its MANIFEST sha256
    first. The bundle is checked for hygiene: no internal path, run name or user-specific string may appear in the card
    or in any GGUF metadata string.
"""
import contextlib
import json
import os
import platform
import re
import shutil
import stat
import time

from .. import paths, provenance
from ..factory.chain import ORDER, StageError, done_path, read_done, sha_file

SEMVER = re.compile(r"(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(-[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?"
                    r"(\+[0-9A-Za-z-]+(\.[0-9A-Za-z-]+)*)?")


def _machine():
    """The machine that runs the release (the training machine is recorded apart, from the finetune stage)."""
    import torch
    return dict(cpu=platform.processor() or platform.machine(), os=f"{platform.system()} {platform.release()}",
                python=platform.python_version(), cpu_count=os.cpu_count(), torch_threads=torch.get_num_threads())


def _rmtree(path):
    """shutil.rmtree that also removes read-only files (Windows)."""
    def onerror(func, p, _):
        os.chmod(p, stat.S_IWRITE)
        func(p)
    shutil.rmtree(path, onerror=onerror)


def release_dir(spec):
    return os.path.join(paths.specialists_dir(), spec["name"], str(spec["version"]))


def internal_release(work, spec):
    from ..factory.stages import _checkpoint_of, export_checkpoint
    from ..verticals import load_specialist
    ev = read_done(work, spec, "eval")
    if not ev["gate_passes"]:
        raise StageError("the gate was not passed: not released ("
                         + "; ".join(r["reason"] for r in ev.get("refused") or []) + ")")
    dd, ft = read_done(work, spec, "data"), read_done(work, spec, "finetune")
    sp = load_specialist(spec["specialist"])
    checked = set((dd.get("exam_leak_check") or {}).get("exams") or {})
    missing = [x for x in sp.exams if x not in checked]
    if missing:
        raise StageError(f"the specialist `{spec['specialist']}` lists the exam(s) {missing} and the data stage of this "
                         f"run did not check them (their fingerprints were not registered): a release needs every one "
                         f"(data/fingerprint_exams.py, then run the stages again from `data`)")
    dest = release_dir(spec)
    if os.path.exists(dest):
        raise StageError(f"{dest} already exists: versions are never overwritten")
    released = ft["released_seed"]
    n_members = sp.ensemble
    others = [r for r in ft["runs"] if r["seed"] != released][: n_members - 1]
    if 1 + len(others) != n_members:
        raise StageError(f"specialist.yaml asks for an ensemble of {n_members} (release.ensemble) and this run trained "
                         f"{len(ft['runs'])} seed(s): train enough seeds or change the specialist's release policy")
    ex, cal, rp = (read_done(work, spec, s) for s in ("export", "calibrate", "report"))
    if sha_file(ex["gguf"]) != ex["sha256"]:
        raise StageError(f"{ex['gguf']} changed since the export stage wrote it: run the stages again from `export`")
    tmp = f"{dest}.partial-{os.getpid()}"
    if os.path.exists(tmp):
        _rmtree(tmp)
    os.makedirs(tmp)
    try:
        shutil.copy(ex["gguf"], os.path.join(tmp, "model.gguf"))
        members = [dict(file="model.gguf", seed=released, sha256=sha_file(os.path.join(tmp, "model.gguf")),
                        ckpt_sha256=ex["ckpt_sha256"])]
        for i, r in enumerate(others, 1):
            _checkpoint_of(ft, r["seed"])                      # refuses a checkpoint that changed since finetune
            p = os.path.join(tmp, f"member_{i}.gguf")
            export_checkpoint(r["ckpt"], p, spec)
            members.append(dict(file=os.path.basename(p), seed=r["seed"], sha256=sha_file(p),
                                ckpt_sha256=r["ckpt_sha256"]))
        if sha_file(rp["report"]) != rp["report_sha256"]:
            raise StageError(f"{rp['report']} changed since the report stage wrote it: run `report` again")
        shutil.copy(rp["report"], os.path.join(tmp, "report.md"))
        shutil.copy(spec["_path"], os.path.join(tmp, "task.yaml"))
        chain = {}
        for s in ORDER[:-1]:
            shutil.copy(done_path(work, spec, s), os.path.join(tmp, f"done_{s}.json"))
            chain[s] = sha_file(os.path.join(tmp, f"done_{s}.json"))
        files = {fn: sha_file(os.path.join(tmp, fn)) for fn in sorted(os.listdir(tmp))}
        man = dict(name=spec["name"], version=str(spec["version"]), specialist=spec["specialist"],
                   created=time.strftime("%Y-%m-%dT%H:%M:%S"), code_commit=provenance.code_commit(),
                   task_file=os.path.basename(spec["_path"]), task_sha256=sha_file(spec["_path"]),
                   task_digest=spec["_digest"], files=files, done_chain_sha256=chain,
                   members=members, ensemble=n_members, vote=sp.vote, seeds=[r["seed"] for r in ft["runs"]],
                   released_seed=released, released_seed_rule=ft.get("released_seed_rule"),
                   released_seed_reason=ft.get("released_seed_reason"), select_best_seed=ft.get("select_best_seed"),
                   operating_bias=cal["operating_bias"], operating_bias_vec=cal.get("operating_bias_vec"),
                   labels=cal.get("labels"), synthetic=ev["metrics"],
                   gate=dict(passes=ev["gate_passes"], reason=ev["gate_reason"]), decide=ev.get("scope"),
                   criteria=rp["criteria_summary"], criteria_verdicts=rp["criteria"],
                   criteria_sha256=dd.get("criteria_sha256"), criteria_frozen_at=dd.get("criteria_frozen_at"),
                   exam_records=rp.get("exam_records"),
                   warnings=list(dd.get("warnings") or []) + list(ft.get("warnings") or [])
                   + list(ev.get("notes") or []),
                   experimental=ft.get("experimental"), training=ft.get("training"), release_machine=_machine())
        with open(os.path.join(tmp, "MANIFEST.json"), "w", encoding="utf-8") as f:
            json.dump(man, f, indent=2, default=str)
        for fn in os.listdir(tmp):
            os.chmod(os.path.join(tmp, fn), 0o444)
        if os.path.exists(dest):
            raise StageError(f"{dest} already exists: versions are never overwritten")
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        os.replace(tmp, dest)
    except BaseException:
        with contextlib.suppress(OSError):
            _rmtree(tmp)
        raise
    return dict(dest=dest, **man)


def read_manifest(release):
    with open(os.path.join(release, "MANIFEST.json"), encoding="utf-8") as f:
        return json.load(f)


def verify_release(release):
    """Every file the MANIFEST lists (every member at least, for a MANIFEST of the earlier format) still has its
    sha256. Returns the MANIFEST; raises StageError naming the first file that differs."""
    man = read_manifest(release)
    listed = man.get("files") or {m["file"]: m["sha256"] for m in man["members"]}
    for fn, digest in sorted(listed.items()):
        p = os.path.join(release, fn)
        if not os.path.exists(p) or sha_file(p) != digest:
            raise StageError(f"{p} is missing or differs from the sha256 its MANIFEST records: the internal release "
                             f"was altered, it cannot be published")
    return man


def release_matches(release, spec):
    """True when the internal release was made from THIS task (its digest; the file's sha256 for a MANIFEST of the
    earlier format)."""
    man = read_manifest(release)
    if man.get("task_digest"):
        return man["task_digest"] == spec["_digest"]
    return man.get("task_sha256") == sha_file(spec["_path"])


# Strings that must never reach a public artefact: absolute paths (Windows drive letters, UNC shares, the usual POSIX
# roots, home directories), and internal version tags such as `v12`.
FORBIDDEN = [r"(?<![A-Za-z0-9])[A-Za-z]:[\\/]", r"\\\\[A-Za-z0-9._$-]+\\", r"(?:^|(?<=[\s\"'`(\[=,;:]))~[\\/]",
             r"(?:^|(?<=[\s\"'`(\[=,;]))/(?:home|Users|root|tmp|mnt|opt|srv|var|data|media|scratch|private|Volumes)/",
             r"\bv\d{1,3}\b(?![.\d])"]


def _patterns():
    """The generic patterns + the maintainers' own (one regex per line in $PUREBYTE_WORK/releases/forbidden.txt,
    private: internal run names, host names...)."""
    extra = os.path.join(paths.releases_dir(), "forbidden.txt")
    pats = list(FORBIDDEN)
    if os.path.exists(extra):
        with open(extra, encoding="utf-8") as f:
            pats += [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]
    for pat in pats:                                     # a broken pattern must fail BEFORE a bundle is started
        try:
            re.compile(pat)
        except re.error as e:
            raise StageError(f"forbidden pattern {pat!r} ({extra}): {e}")
    return pats


def hygiene(texts):
    """Strings that must never reach a public artefact (paths, user directories, run tags). Returns the offences."""
    bad = []
    pats = _patterns()
    for label, text in texts.items():
        for pat in pats:
            for mt in re.finditer(pat, text):
                bad.append(f"{label}: `{mt.group(0)}` ({pat})")
    return bad


def _operating_point(man):
    vec = man.get("operating_bias_vec")
    if vec:
        labels = man.get("labels") or [str(i) for i in range(len(vec))]
        return "per type " + ", ".join(f"{a} {b}" for a, b in zip(labels, vec))
    return str(man["operating_bias"])


def public_bundle(release_dir, semver, card_template=None, url_base="https://github.com/purebyte-ai/purebyte/releases/download"):
    from ..export.gguf import read_metadata
    if not SEMVER.fullmatch(str(semver)):
        raise StageError(f"--semver {semver!r} is not a semantic version (X.Y.Z, optionally -prerelease)")
    man = verify_release(release_dir)
    from ..verticals import load_specialist
    sp = load_specialist(man["specialist"])
    name = sp.config.get("public_name", man["specialist"])
    out = os.path.join(paths.releases_dir(), f"{name}-{semver}")
    _patterns()
    if os.path.exists(out):
        raise StageError(f"{out} exists: bundles are never overwritten")
    os.makedirs(out)
    files = []
    for i, mem in enumerate(man["members"]):
        fn = f"purebyte-{name}-{semver}" + ("" if i == 0 else f"-e{i}") + ".gguf"
        shutil.copy(os.path.join(release_dir, mem["file"]), os.path.join(out, fn))
        digest = sha_file(os.path.join(out, fn))
        with open(os.path.join(out, fn + ".sha256"), "w", encoding="utf-8", newline="\n") as f:
            f.write(f"{digest}  {fn}\n")
        files.append(dict(file=fn, sha256=digest, size=os.path.getsize(os.path.join(out, fn)),
                          url=f"{url_base}/{name}-{semver}/{fn}"))
    card_template = card_template or os.path.join(sp.dir, "MODEL_CARD.template.md")
    if os.path.exists(card_template):
        with open(card_template, encoding="utf-8") as f:
            card = f.read()
    else:
        card = f"# {name} {semver}\n"
    card = card.replace("{{version}}", semver).replace("{{files}}", "\n".join(
        f"| `{x['file']}` | {x['size']:,} | `{x['sha256']}` |" for x in files)).replace(
        "{{operating_bias}}", _operating_point(man)).replace("{{vote}}", f"{man['vote']} of {len(files)}")
    with open(os.path.join(out, "MODEL_CARD.md"), "w", encoding="utf-8", newline="\n") as f:
        f.write(card)
    metas = {x["file"]: read_metadata(os.path.join(out, x["file"])) for x in files}
    meta_strings = {fn: json.dumps({k: v for k, v in m.items() if isinstance(v, (str, list))}) for fn, m in metas.items()}
    bad = hygiene(dict(meta_strings, card=card))
    bad += [f"{fn}: general.version {m['general.version']!r} is not the bundle's version {semver!r}"
            for fn, m in metas.items() if m.get("general.version") not in (None, semver)]
    other = sorted({v for v in re.findall(rf"\b{re.escape(name)} v?(\d+\.\d+\.\d+(?:-[0-9A-Za-z.-]+)?)", card)
                    if v != semver})
    bad += [f"card: it names {name} {v}, and this bundle is {name} {semver} (numbers measured on another version: "
            f"update the card template)" for v in other]
    if bad:
        shutil.rmtree(out, ignore_errors=True)            # nothing half-made stays behind
        raise StageError("the public bundle carries internal strings or another version: " + "; ".join(bad[:10]))
    frag = dict(name=name, version=semver, files=files, ensemble=dict(members=len(files), vote=man["vote"]))
    with open(os.path.join(out, "models.json.fragment"), "w", encoding="utf-8", newline="\n") as f:
        json.dump(frag, f, indent=2)
    with open(os.path.join(paths.releases_dir(), "MAPPING.jsonl"), "a", encoding="utf-8") as f:
        f.write(json.dumps(dict(public=f"{name} {semver}", internal=f"{man['name']} {man['version']}",
                                release_dir=release_dir, task_sha256=man["task_sha256"],
                                task_digest=man.get("task_digest"), seeds=man["seeds"],
                                members=[(m["seed"], m["sha256"]) for m in man["members"]],
                                code_commit=man["code_commit"], ts=time.strftime("%Y-%m-%dT%H:%M:%S"))) + "\n")
    return dict(bundle=out, files=files)
