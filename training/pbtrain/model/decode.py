"""BIOES labels, their Viterbi decoding and spans (spec/FORMAT.md §11.6).

Label scheme for k entity types: 0 = O; for entity e: B = 1 + 4e, I = 2 + 4e, E = 3 + 4e, S = 4 + 4e.
Viterbi forbids invalid transitions; the operating point is moved by per-label biases added to the emissions (never to
O), without retraining. Decoding is in float64 with the lowest label index on ties, exactly as the engine.

Span masks (training data): `S` (T,) uint8 per window: 0 = outside; for ONE entity type any nonzero byte belongs to a
span (a 0/1 mask); for k > 1 types the value is 1 + type index; 255 (IGNORE) = unknown / outside the taxonomy, trained
with label -100 (ignored by the cross-entropy) and never counted.
"""
import numpy as np

NEG = -1e30
IGNORE = 255
IGNORE_LABEL = -100


def n_labels(n_entities: int) -> int:
    return 1 + 4 * n_entities


def label_names(entities):
    out = ["O"]
    for e in entities:
        out += [f"B-{e}", f"I-{e}", f"E-{e}", f"S-{e}"]
    return out


def transition_mask(n_entities: int) -> np.ndarray:
    """(L, L): 0 where the transition is valid, -1e30 where it is forbidden.

    O, E-x, S-x -> O, B-y, S-y ;  B-x, I-x -> I-x, E-x (same entity only)
    """
    L = n_labels(n_entities)
    m = np.full((L, L), NEG, dtype=np.float32)
    starts = [0] + [1 + 4 * e for e in range(n_entities)] + [4 + 4 * e for e in range(n_entities)]
    for src in [0] + [3 + 4 * e for e in range(n_entities)] + [4 + 4 * e for e in range(n_entities)]:
        for dst in starts:
            m[src, dst] = 0.0
    for e in range(n_entities):
        B, I, E = 1 + 4 * e, 2 + 4 * e, 3 + 4 * e
        m[B, I] = m[B, E] = 0.0
        m[I, I] = m[I, E] = 0.0
    return m


def start_mask(n_entities: int) -> np.ndarray:
    L = n_labels(n_entities)
    v = np.full(L, NEG, dtype=np.float32)
    v[0] = 0.0
    for e in range(n_entities):
        v[1 + 4 * e] = 0.0      # B
        v[4 + 4 * e] = 0.0      # S
    return v


def end_mask(n_entities: int) -> np.ndarray:
    L = n_labels(n_entities)
    v = np.full(L, NEG, dtype=np.float32)
    v[0] = 0.0
    for e in range(n_entities):
        v[3 + 4 * e] = 0.0      # E
        v[4 + 4 * e] = 0.0      # S
    return v


def label_bias(n_entities: int, scalar=0.0, per_type=None) -> np.ndarray:
    """(L,) bias: 0 on O, `scalar` (+ per_type[e]) on the four labels of type e (the engine's `--bias`)."""
    lb = np.zeros(n_labels(n_entities), dtype=np.float64)
    for e in range(n_entities):
        lb[1 + 4 * e:5 + 4 * e] = scalar + (per_type[e] if per_type is not None else 0.0)
    return lb


def viterbi(logp: np.ndarray, n_entities: int, label_bias=None) -> np.ndarray:
    """logp (T, L) per-byte log-probabilities -> labels (T,). `label_bias` (L,) is ADDED to the emissions."""
    T, L = logp.shape
    assert L == n_labels(n_entities), f"{L} labels, expected {n_labels(n_entities)}"
    em = logp.astype(np.float64)
    if label_bias is not None:
        em = em + np.asarray(label_bias, dtype=np.float64)
    trans = transition_mask(n_entities).astype(np.float64)
    dp = em[0] + start_mask(n_entities)
    bp = np.zeros((T, L), dtype=np.int32)
    for t in range(1, T):
        sc = dp[:, None] + trans                 # (L_prev, L_cur)
        bp[t] = sc.argmax(0)
        dp = sc.max(0) + em[t]
    dp = dp + end_mask(n_entities)
    path = np.zeros(T, dtype=np.int64)
    path[-1] = int(dp.argmax())
    for t in range(T - 1, 0, -1):
        path[t - 1] = bp[t, path[t]]
    return path


def viterbi_batch(logp: np.ndarray, n_entities: int, label_bias=None) -> np.ndarray:
    """(K, T, L) -> (K, T): the same float64 operations as `viterbi` for every row at once (bit-identical paths)."""
    K, T, L = logp.shape
    em = logp.astype(np.float64)
    if label_bias is not None:
        em = em + np.asarray(label_bias, dtype=np.float64)
    trans = transition_mask(n_entities).astype(np.float64)
    dp = em[:, 0] + start_mask(n_entities)
    bp = np.zeros((K, T, L), dtype=np.int32)
    for t in range(1, T):
        sc = dp[:, :, None] + trans              # (K, L_prev, L_cur)
        bp[:, t] = sc.argmax(1)
        dp = sc.max(1) + em[:, t]
    dp = dp + end_mask(n_entities)
    path = np.zeros((K, T), dtype=np.int64)
    path[:, -1] = dp.argmax(1)
    rows = np.arange(K)
    for t in range(T - 1, 0, -1):
        path[:, t - 1] = bp[rows, t, path[:, t]]
    return path


def spans(labels: np.ndarray, n_entities: int):
    """A valid BIOES sequence -> [(start, exclusive end, entity)]. Negative labels (IGNORE_LABEL in gold sequences)
    are read as O: an ignored byte is never a span and closes an open one."""
    out, ini, ent = [], None, None
    for t, l in enumerate(labels):
        if l <= 0:
            ini = None
            continue
        e, kind = (l - 1) // 4, (l - 1) % 4     # 0=B 1=I 2=E 3=S
        if kind == 0:
            ini, ent = t, e
        elif kind == 3:
            out.append((t, t + 1, e)); ini = None
        elif kind == 2 and ini is not None and ent == e:
            out.append((ini, t + 1, e)); ini = None
    return out


def mask_to_bioes(mask_row, n_entities: int = 1):
    """(T,) span mask -> (T,) int64 BIOES labels (-100 on IGNORE bytes). A 1-byte span is S, not B+E.

    One entity: every nonzero, non-IGNORE byte is inside a span and contiguous bytes form ONE span (on 0/1
    masks this is the historical single-entity conversion, bit for bit). k > 1 entities: a span is a maximal run of one value 1 + e.
    """
    row = np.asarray(mask_row)
    T = len(row)
    out = np.zeros(T, dtype=np.int64)
    t = 0
    while t < T:
        v = int(row[t])
        if v == 0:
            t += 1
            continue
        if v == IGNORE:
            out[t] = IGNORE_LABEL
            t += 1
            continue
        e = 0 if n_entities == 1 else v - 1
        if not 0 <= e < n_entities:
            raise ValueError(f"mask value {v} with {n_entities} entity type(s)")
        j = t
        while j + 1 < T and int(row[j + 1]) != IGNORE and (
                int(row[j + 1]) != 0 if n_entities == 1 else int(row[j + 1]) == v):
            j += 1
        if j == t:
            out[t] = 4 + 4 * e
        else:
            out[t], out[j] = 1 + 4 * e, 3 + 4 * e
            out[t + 1:j] = 2 + 4 * e
        t = j + 1
    return out


def mask_spans(mask_row, n_entities: int = 1):
    """Gold spans of a mask, the same way the model's are read: spans(mask_to_bioes(row))."""
    return spans(mask_to_bioes(mask_row, n_entities), n_entities)


def window_class(S):
    """Y per window: the window holds a span of some type (IGNORE bytes do not count)."""
    S = np.asarray(S)
    return ((S >= 1) & (S != IGNORE)).any(axis=-1).astype(np.uint8)
