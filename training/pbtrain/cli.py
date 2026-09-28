"""pbtrain: the command line of the training stack.

    pbtrain doctor                                   what is installed, found and missing (GPUs probed in a child
                                                     process)
    pbtrain new-vertical NAME [--entities a,b]       scaffold specialists/NAME (plugin, specialist.yaml, recipes, tests)
    pbtrain data TASK                                build the splits + leak checks (stage `data`, CPU)
    pbtrain train TASK [--device cpu|cuda|mps|auto] [--parallel N] [--backend cuda|mps|radeon]
                                                     baselines -> finetune -> eval -> calibrate
    pbtrain eval TASK [--record EXAM --metrics FILE] record an exam's metrics; print every pre-registered criterion
    pbtrain export TASK                              export -> verify (runtime equivalence, CPU)
    pbtrain release TASK [--public --semver X.Y.Z]   report -> immutable internal release (-> public bundle); when the
                                                     internal release of TASK exists, `--public` only makes the bundle
    pbtrain run TASK [--stages all|a,b,...] [--device ...] [--parallel N] [--backend ...]
                                                     the whole chain (or the listed stages)
    pbtrain status TASK                              which stages are done; criteria verdicts
    pbtrain exams                                    registered exam fingerprints (leak check)

Every command runs at idle priority with --threads (default 2) CPU threads. CUDA and ROCm devices are hidden from the
process unless --device asks for a GPU: cuda (NVIDIA CUDA or AMD ROCm), mps (Apple silicon) or auto (the first of cuda,
mps and cpu found here). Apple's MPS cannot be hidden by an environment variable; nothing is placed on it unless
--device mps or auto chooses it. --device, --parallel and --backend exist only where they act (`train`, `run`), and an
option that would act on nothing is refused.
"""
import argparse
import json
import os
import sys

GPU_STAGES = ("finetune", "calibrate")     # the stages that compute on --device (finetune trains, calibrate measures)


def _args(argv):
    ap = argparse.ArgumentParser(prog="pbtrain", description="PureByte training stack")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("PBTRAIN_THREADS", "2")))
    ap.add_argument("--work", default=None, help="factory work dir (default $PUREBYTE_WORK/runs)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    nv = sub.add_parser("new-vertical")
    nv.add_argument("name")
    nv.add_argument("--entities", default="span")
    nv.add_argument("--data-format", default=None)
    for name in ("data", "train", "export", "run", "status"):
        p = sub.add_parser(name)
        p.add_argument("task")
        if name in ("train", "run"):
            p.add_argument("--device", default="cpu", choices=("cpu", "cuda", "mps", "auto"),
                           help="cpu (the reference), cuda (NVIDIA or AMD ROCm), mps (Apple silicon) or auto")
            p.add_argument("--parallel", type=int, default=1, help="seeds trained at once (worker processes)")
            p.add_argument("--backend", default=None, choices=("cpu", "cuda", "mps", "radeon"),
                           help="default: the stock PyTorch code of the device; radeon: fused HIP kernels (ROCm on "
                                "Windows, with --device cuda)")
        if name == "run":
            p.add_argument("--stages", default="all")
    ev = sub.add_parser("eval")
    ev.add_argument("task")
    ev.add_argument("--record", default=None, help="exam name whose metrics are recorded")
    ev.add_argument("--metrics", default=None, help="JSON file with the exam's metrics {metric: value}")
    rl = sub.add_parser("release")
    rl.add_argument("task")
    rl.add_argument("--public", action="store_true")
    rl.add_argument("--semver", default=None)
    sub.add_parser("exams")
    return ap.parse_args(argv)


def _ctx(a, spec=None):
    """The factory Context of a command. Refuses a device / backend combination that would not do what it says, and
    the radeon backend where its kernels do not run."""
    from .factory.stages import Context
    from .gpu import RADEON_MAX_WINDOW, backend_for, enable
    device = getattr(a, "device", "cpu")
    backend = backend_for(device, getattr(a, "backend", None))
    if backend == "radeon":
        if spec is not None and int(spec["data"]["seq_len"]) > RADEON_MAX_WINDOW:
            raise SystemExit(f"--backend radeon: the kernels are validated up to {RADEON_MAX_WINDOW}-byte windows "
                             f"(1,024 crashed) and this task's seq_len is {spec['data']['seq_len']}")
        enable("radeon")
    return Context(device=device, parallel=getattr(a, "parallel", 1), backend=backend)


def _run(a, stages):
    from .factory import spec as S
    from .factory.chain import ORDER
    from .factory.stages import run
    from .gpu import backend_for
    from .runtime import gpu_determinism, gpu_lock, resolve_device
    spec = S.load(a.task)
    names = ORDER if stages == ["all"] else stages
    requested = getattr(a, "device", "cpu")
    parallel = getattr(a, "parallel", 1)
    if parallel < 1:
        raise SystemExit("--parallel must be 1 or more")
    uses_device = any(s in GPU_STAGES for s in names)
    if not uses_device and (requested != "cpu" or getattr(a, "backend", None) or parallel != 1):
        raise SystemExit(f"--device, --backend and --parallel act on the stages {', '.join(GPU_STAGES)}; the stages "
                         f"{', '.join(names)} run on the CPU")
    if parallel != 1 and "finetune" not in names:
        raise SystemExit("--parallel acts on `finetune` (it trains seeds at once), which is not among the stages")
    if requested != "auto":
        backend_for(requested, getattr(a, "backend", None))   # a combination that would not do what it says: refused
    a.device = resolve_device(requested)                 # "auto" becomes the device found; a missing GPU is an error
    if requested == "auto":
        print(f"pbtrain: --device auto chose {a.device}", file=sys.stderr)
    elif a.device != "cpu":
        print(f"pbtrain: training on {a.device}", file=sys.stderr)
    if a.device != "cpu":
        with gpu_lock(f"pbtrain {a.cmd} {spec['name']} {spec['version']}"):
            ctx = _ctx(a, spec)
            gpu_determinism()
            return run(spec, stages, a.work, ctx)
    return run(spec, stages, a.work, _ctx(a, spec))


def cmd_eval(a):
    from . import paths
    from .factory import criteria as C
    from .factory import exams as E
    from .factory import spec as S
    from .factory.chain import read_done, StageError
    if a.metrics and not a.record:
        raise SystemExit("--metrics acts only with --record EXAM (the exam whose metrics the file holds)")
    if a.record and not a.metrics:
        raise SystemExit("--record needs --metrics FILE (a JSON object {metric: value})")
    spec = S.load(a.task)
    work = a.work or paths.runs_dir()
    if a.record:
        rec, before = E.record(work, spec, a.record, a.metrics)
        if before:
            print(f"WARNING: exam {a.record} already had {before} record(s); this one is added to its history (the "
                  f"criteria read the latest; the report and the release MANIFEST show every record)", file=sys.stderr)
        print(f"recorded {a.record} ({rec['recorded']}): {rec['metrics']}")
    try:
        synthetic = read_done(work, spec, "eval")["metrics"]
    except StageError:
        synthetic = None
    verdicts = C.evaluate(spec.get("criteria"), synthetic, E.results(work, spec))
    for v in verdicts:
        c = v["criterion"]
        print(f"{v['verdict']:8s} {c.get('exam', 'synthetic')}: {c['metric']} {c['op']} {c['value']}  "
              f"(observed {v['observed']})")
    print(C.summary(verdicts))


def cmd_status(a):
    from . import paths
    from .factory import spec as S
    from .factory.chain import ORDER, done_path
    spec = S.load(a.task)
    work = a.work or paths.runs_dir()
    for s in ORDER:
        print(f"{'done' if os.path.exists(done_path(work, spec, s)) else '----'}  {s}")
    cmd_eval(argparse.Namespace(task=a.task, work=a.work, record=None, metrics=None))


def cmd_release(a):
    """`pbtrain release TASK` makes the immutable internal release (after `report`). With `--public --semver X.Y.Z`
    it also makes the public bundle; when the internal release of TASK already exists (made by an earlier `pbtrain
    release TASK`), it is checked (same task digest, every file's sha256) and only the bundle is made."""
    from .factory import spec as S
    from .factory.chain import StageError
    from .release.package import SEMVER, public_bundle, release_dir, release_matches, verify_release
    if a.public and not a.semver:
        raise SystemExit("--public needs --semver X.Y.Z")
    if a.semver and not a.public:
        raise SystemExit("--semver acts only with --public (it names the public bundle)")
    if a.semver and not SEMVER.fullmatch(a.semver):
        raise SystemExit(f"--semver {a.semver!r} is not a semantic version (X.Y.Z, optionally -prerelease)")
    spec = S.load(a.task)
    rel = release_dir(spec)
    if os.path.exists(rel):
        if not a.public:
            raise StageError(f"{rel} already exists: versions are never overwritten (`--public --semver X.Y.Z` makes "
                             f"the public bundle from it)")
        if not release_matches(rel, spec):
            raise StageError(f"{rel} was released from another version of the task {spec['_path']}: it is not this "
                             f"task's release")
        verify_release(rel)
        print(f"pbtrain: the internal release {rel} exists and matches the task: making the public bundle from it",
              file=sys.stderr)
    else:
        _run(argparse.Namespace(**dict(vars(a), device="cpu", parallel=1, backend=None, cmd="release")),
             ["report", "release"])
    if a.public:
        print(json.dumps(public_bundle(rel, a.semver), indent=2))


def cmd_exams(a):
    from . import paths
    from .data import leakage
    from .data.fingerprint import load
    found = leakage.registered_exams(paths.exams_dir())
    if not found:
        print(f"no exam fingerprints in {os.path.join(paths.exams_dir(), 'fingerprints')} (data/fingerprint_exams.py)")
    for name, p in found.items():
        fp, meta = load(p)
        print(f"{name:24s} {len(fp):>10,} blocks  {meta.get('files', '?')} files  {meta.get('bytes', '?')} bytes  "
              f"role {meta.get('role', 'test')}")


def main(argv=None):
    a = _args(argv if argv is not None else sys.argv[1:])
    from . import runtime
    runtime.setup(threads=a.threads, lowprio=True, gpu=getattr(a, "device", "cpu") != "cpu")
    if a.cmd == "doctor":
        from .doctor import doctor
        return doctor()
    if a.cmd == "new-vertical":
        from .scaffold import new_vertical
        print(new_vertical(a.name, [e.strip() for e in a.entities.split(",") if e.strip()], a.data_format))
        return 0
    from .factory.chain import StageError
    from .train.seeds import WorkerError
    try:
        if a.cmd == "data":
            _run(a, ["data"])
        elif a.cmd == "train":
            _run(a, ["baselines", "finetune", "eval", "calibrate"])
        elif a.cmd == "export":
            _run(a, ["export", "verify"])
        elif a.cmd == "run":
            _run(a, ["all"] if a.stages == "all" else a.stages.split(","))
        elif a.cmd == "eval":
            cmd_eval(a)
        elif a.cmd == "status":
            cmd_status(a)
        elif a.cmd == "release":
            cmd_release(a)
        elif a.cmd == "exams":
            cmd_exams(a)
    except (StageError, runtime.DeviceError, runtime.GpuBusyError, WorkerError) as e:
        # an expected outcome (a refused gate, a leak, a typo, a missing GPU, a busy GPU): a message, not a trace
        print(f"pbtrain: {e}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
