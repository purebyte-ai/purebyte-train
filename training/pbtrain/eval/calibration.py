"""Calibration of the window-level (choice) head: expected calibration error with its noise floor.

For span models the operating point is the Viterbi bias (`operating_point.py`), not a temperature. The document head
gates the spans by its ARGMAX, which a temperature cannot change, and the exported file carries no temperature: the
`calibrate` stage keeps 1.0 and records, on `select`, the head's ECE next to the floor of the estimator, as information
about the probabilities the runtime reports (`p_positive`).

A small ECE does not say that a model is calibrated; it may only say that the estimator resolves no further. With
n = 2,000 and saturated confidences, a PERFECTLY calibrated model gives a median ECE of ~0.001 and a p95 of ~0.003.
`ece_floor` measures that floor for the model's own confidences (parametric bootstrap, its own generator: the global
RNG is not touched); an ECE below its p95 is not a measurement.
"""
import numpy as np
import torch
import torch.nn.functional as F


def ece(probs, y, bins=15):
    conf, pred = probs.max(1)
    acc = (pred == y).float()
    e = 0.0
    for i in range(bins):
        lo, hi = i / bins, (i + 1) / bins
        m = (conf > lo) & (conf <= hi)
        if m.any():
            e += m.float().mean().item() * abs(acc[m].mean().item() - conf[m].mean().item())
    return e


def ece_floor(probs, n_sim=200, bins=15, seed=0):
    """(median, p95) of the ECE of a perfectly calibrated model with THESE confidences."""
    conf = probs.max(1)[0].cpu()
    g = torch.Generator().manual_seed(seed)
    es = []
    for _ in range(n_sim):
        y = (torch.rand(len(conf), generator=g) < conf).long()
        es.append(ece(torch.stack([1 - conf, conf], 1), y, bins))
    es = np.sort(np.array(es))
    return float(np.median(es)), float(es[int(0.95 * (len(es) - 1))])


def fit_temperature(logits, y):
    """Temperature scaling fitted on `select` (never on `test`). The factory does not call it: span models keep 1.0."""
    log_t = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=60)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(logits / log_t.exp(), y)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp())


@torch.no_grad()
def choice_probs(m, ch, X, *, bs=16, device="cpu"):
    """The choice head's class probabilities [n, classes] (float32, on the CPU), eval mode; modes restored."""
    mods = (m, ch)
    was = [x.training for x in mods]
    for x in mods:
        x.eval()
    out = []
    try:
        for i in range(0, len(X), bs):
            h, _ = m(torch.from_numpy(X[i:i + bs].astype(np.int64)).to(device))
            out.append(torch.softmax(ch(h).float(), -1).cpu())
    finally:
        for x, w in zip(mods, was):
            x.train(w)
    return torch.cat(out)


def measure(m, ch, X, Y, device="cpu"):
    """{ece, floor_median, floor_p95, resolved, windows} of the choice head on windows X with labels Y."""
    p = choice_probs(m, ch, X, device=device)
    e = ece(p, torch.as_tensor(np.asarray(Y).astype(np.int64)))
    med, p95 = ece_floor(p)
    return dict(ece=round(e, 5), floor_median=round(med, 5), floor_p95=round(p95, 5), resolved=bool(e > p95),
                windows=int(len(X)))
