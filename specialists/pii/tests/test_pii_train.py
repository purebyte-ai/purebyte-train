"""A tiny end-to-end run of the core trainer with the 12 types, on CPU: windows from the plugin, `train_spans` with
`n_entities=12` (a tag head of 49 labels, IGNORE bytes trained as -100), the core's span metrics, and determinism."""
import numpy as np
import torch
from pbtrain.eval.spans import forward_windows, score
from pbtrain.model.decode import IGNORE_LABEL, mask_to_bioes
from pbtrain.train.loop import SpanTask, train_spans
from pbtrain.train.recipe import SpanRecipe

from pii_testkit import build, data_block


def test_train_spans_with_twelve_types_on_cpu(pii_env):
    _, t, _ = build(pii_env, data_block(n_train=32, n_eval=16))
    labels = np.stack([mask_to_bioes(row, 12) for row in t.train[2]])
    assert (labels == IGNORE_LABEL).any() and labels.max() <= 48 and len(np.unique(labels[labels > 0])) > 4
    task = SpanTask(t.train, t.select, t.test)
    recipe = SpanRecipe(size="mini64", n_entities=12, steps=4, batch=8, lr=3e-3, warmup=1)
    logs = []
    m, th, ch = train_spans(task, recipe, seed=0, log=logs.append)
    assert th.proj.out_features == 49
    assert all(torch.isfinite(p).all() for mod in (m, th, ch) for p in mod.parameters())
    assert len(logs) == 4 and all(np.isfinite(x["loss"]) for x in logs)
    lp, doc = forward_windows(m, th, ch, t.test[0])
    assert lp.shape == (16, 128, 49) and doc.shape == (16,)
    s = score(lp, doc, t.test[2], t.test[1], 0.0, 12)
    assert all(0.0 <= s[k]["span_f1"] <= 1.0 for k in ("tag_only", "gated", "oracle"))
    m2, th2, ch2 = train_spans(task, recipe, seed=0)
    for a, b in ((m, m2), (th, th2), (ch, ch2)):
        for (ka, va), (kb, vb) in zip(a.state_dict().items(), b.state_dict().items()):
            assert ka == kb and torch.equal(va, vb), ka
