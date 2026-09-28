# Known pitfalls

Each of these cost hours, a model or a machine. Symptom, rule, reason.

## The machine

**One GPU job at a time.** Under ROCm on Windows an out-of-memory does not fail cleanly: VRAM spills into shared memory
and the whole machine can freeze. Two trainings at once also doubled the time of each. `pbtrain` takes the GPU through
an operating-system lock (`runtime.gpu_lock`) that refuses while another process holds it and is released when that
process ends, however it ends. Watch the shared GPU memory counter
during the first minutes of a launch; above ~1 GB, stop the job. Budgets: [gpu.md](gpu.md).

**Stop a queue before its jobs.** A queue of trainings is a script loop: stopping the Python job first makes the loop
start the next one. Stop processes by PID only, never by image name (other people's jobs share the machine).

**Chain on markers, not on process lists.** Wait for a line like `=== done` in a log; waiting for "no python running"
waits forever on someone else's process. Two queues that wait for each other's marker deadlock silently.

**Never edit the code a running job imports.** An edit under a running queue killed a training mid-run.

**Idle priority can fail silently** (untyped `ctypes` truncates the process handle on 64-bit Windows).
`runtime.low_priority()` uses typed signatures and returns whether the priority was verified; `pbtrain doctor` shows it.

**Too many threads.** Four OpenMP threads on a loaded machine ran 60x slower than one. `pbtrain` caps threads (default
2; the training tests and golden cases use 1, the pii tests 2). Timings measured on a loaded machine are inflated
1.5-2x.

**WSL**: `/tmp` is a small tmpfs (exam files filled it); WSL can stop at 100 % CPU; tools that read `C:\` paths from
WSL can fail silently. Keep exam tools and their data inside the Linux file system.

**Antivirus** may quarantine files of security datasets (a proof-of-concept file of a public credentials dataset was
removed mid-evaluation). Scorers must skip missing files and say how many.

## Data and evaluation

**Exam leakage is easy.** Megabytes of exam binaries were once inside a training corpus (the same system libraries);
an old split function put a third of a benchmark's TEST repositories in "train"; an exclusion by file NAME missed
copies. The `data` stage checks train, select and every real-window set against every registered exam by CONTENT
([what exactly](evaluate.md#what-the-leak-checks-check)).

**Selection on the exam is training on the exam.** Choose seeds, ensembles and biases on DEV or `select`.

**The synthetic score does not predict real performance.** The worst seed on real binaries had a normal synthetic
score; recall 0.98 on the synthetic battery was 0.31 on real code.

**The tail seed.** About one seed in six ends far worse (a trajectory event driven by the batch order; once, one
unusual file). Warmup, clipping, a lower ternary threshold and a fixed bias only moved it to another seed. Six seeds,
and read the max.

**Blind negative mining mines real credentials.** Mine with a filter for anything credential-shaped; a background
treated as negative teaches silence.

**Real-window streams push towards silence**: they grow the batch and lower the positive prior. Watch recall.

**An optimum on the edge of a grid is not an optimum**: `operating_point.choose` raises; widen the grid.

**RNG streams differ across Python / NumPy versions, and CPU kernels across processors**: golden fingerprints record
the PyTorch, NumPy and Python versions, the platform and the CPU kernel path, and are compared only where all match.

## Code

**The Radeon backend replaces the block's forward.** It once ignored a quantization option, and it skips anything added
inside the block unless it is mirrored there. `training/tests/test_gpu_backend.py` on the GPU box.

**Exporters drop tensors silently unless coverage is enforced.** `pbtrain.export` raises on any tensor of the model it
did not write, and on modules the format cannot express (LoRA).

**A declared key must act.** A typo (`step:` for `steps:`) once trained 400 steps instead of 2,000 with a believable
result. Unknown keys are refused, in the task and in `specialist.yaml`; values are type- and range-checked; a `data:`
key the vertical did not read fails the stage.

**A stage's result is valid only with its chain.** A task edited after `data`, a stage re-run out of order, a stream
re-mined at the same path: every later stage refuses the chain (it records the task's digest and the sha256 of each
previous done.json), so re-run from the stage that changed.

**Weight decay is decided by shape, then by parameter NAME.** Vectors and scalars are never decayed, whatever their
name. A MATRIX is decayed unless its name holds one of `train/optim.py`'s NO_DECAY_SUBSTRINGS (`norm`, `conv`, `embed`,
`ngram`, `bias`, `gain`, `lora`, `A_log`, `dt_bias`, `dt_proj`; `.D` too): a new matrix parameter that must not be
decayed must carry one of them.

**Caches follow content, not paths.** A file re-mined into the same path must not reuse old weights: the seed cache
and the fingerprints hash file CONTENTS.

**Never pretrain a lookahead model on next-byte prediction**: it learns to copy the future.
