"""The operating point of a span model: the per-label bias added to the tag head's emissions before Viterbi.

It is chosen on `select` (never on `test`), travels in the GGUF (`purebyte.head.<i>.operating_bias`) and the engine
applies it unless the caller overrides it. Policies (declared in the task, `output_schema.bias_policy`):
    conservative  among the biases whose recall meets `min_select_recall`, the one with maximum precision
    f1            among those, the one with maximum F1 (the floor is only an alarm)
    (no floor)    maximum F1
An optimum on the EDGE of the grid means the grid does not contain it: it raises (widen the grid). A flat sweep means
the model does not respond to the bias at all: it is recorded here (the sweep's last entry carries the warning); the
`eval` stage refuses to deliver a model whose RELEASED seed has a flat sweep, and records the other seeds' flat sweeps.
"""
BIAS_GRID = (-6.0, -5.0, -4.0, -3.5, -3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5,
             3.0, 4.0, 5.0, 6.0)
POLICIES = ("conservative", "f1")


PER_TYPE_OFFSETS = (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0)


class OperatingPointError(ValueError):
    pass


def per_type(measure, base, k, offsets=PER_TYPE_OFFSETS):
    """One coordinate-ascent pass over the entity types around the scalar bias `base`: for each type in order, the
    offset that raises the (gated, micro) F1 on `select` the most; ties keep the current value. measure(vec) -> F1.
    Returns (vector of k absolute biases, its F1). For k > 1 only; it travels as `operating_bias_vec`."""
    vec = [float(base)] * k
    best = measure(vec)
    for e in range(k):
        for off in offsets:
            cand = list(vec)
            cand[e] = float(base) + off
            f = measure(cand)
            if f > best + 1e-12:
                best, vec = f, cand
    return vec, best


def choose(measure, grid=BIAS_GRID, min_recall=None, policy="conservative"):
    """measure(bias) -> {span_f1, precision, recall}. Returns (bias, f1, sweep)."""
    if policy not in POLICIES:
        raise OperatingPointError(f"unknown bias_policy {policy!r}; known: {POLICIES}")
    sweep = []
    for b in grid:
        m = measure(b)
        sweep.append(dict(bias=b, f1=round(m["span_f1"], 4), precision=round(m["precision"], 4),
                          recall=round(m["recall"], 4)))
    values = {round(x["f1"], 6) for x in sweep}
    if len(values) == 1:
        warning = "flat sweep: the bias does not move the output"
        floor_met = min_recall is None or sweep[0]["recall"] >= min_recall
        if not floor_met:
            warning += f"; recall {sweep[0]['recall']} is below the declared floor {min_recall}"
        return 0.0, next(iter(values)), sweep + [dict(warning=warning, floor_met=floor_met)]
    mr = max(x["recall"] for x in sweep)
    mp = max(x["precision"] for x in sweep)
    dominant = [x for x in sweep if x["recall"] == mr and x["precision"] == mp]
    dominant = dominant[0] if len(dominant) == 1 else None
    if min_recall is not None:
        holds = [x for x in sweep if x["recall"] >= min_recall]
        if not holds:
            best_r = max(sweep, key=lambda x: x["recall"])
            sweep = sweep + [dict(warning=(f"`min_select_recall: {min_recall}` is NOT reached on `select` (best "
                                           f"{best_r['recall']:.4f} at bias {best_r['bias']}); chosen by F1; the eval "
                                           f"gate decides the release"), floor_met=False)]
            chosen, criterion = max(sweep[:-1], key=lambda x: x["f1"]), "maximum F1 (no bias reaches the floor)"
        elif policy == "f1":
            chosen = max(holds, key=lambda x: (x["f1"], -abs(x["bias"])))
            criterion = f"recall >= {min_recall} (alarm), then maximum F1"
        else:
            chosen = max(holds, key=lambda x: (x["precision"], -x["bias"]))
            criterion = f"recall >= {min_recall}, then maximum precision"
        if dominant is not None and chosen["bias"] == dominant["bias"] and holds:
            sweep = sweep + [dict(warning=(f"the floor bound nothing: bias {dominant['bias']} maximizes precision "
                                           f"AND recall at once"))]
    else:
        chosen, criterion = max(sweep, key=lambda x: x["f1"]), "maximum F1"
    if chosen["bias"] in (grid[0], grid[-1]):
        raise OperatingPointError(f"the optimum ({criterion}) falls on the EDGE of the grid at {chosen['bias']}: widen "
                                  f"the grid {grid[0]}..{grid[-1]}. Sweep: {sweep}")
    return chosen["bias"], chosen["f1"], sweep
