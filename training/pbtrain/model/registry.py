"""The registries: public type name -> PyTorch module. The names are the public engine's (spec/FORMAT.md), so a type
means the same thing when it is trained here and when it is run there.

Adding a block or head type (training/docs/new-specialist.md, "When the architecture must change"):
  1. the module here, with `TYPE = "<name>"` and the house rules (float heads, ternary projections, parameter names
     the optimizer groups understand: every vector and scalar is left out of weight decay whatever its name, and a
     MATRIX that must not be decayed carries one of `train/optim.py`'s NO_DECAY_SUBSTRINGS, as `dt_proj`, `conv`,
     `embed`, `ngram` and `conv_w` do);
  2. one line in BLOCKS or HEADS;
  3. its exporter in `pbtrain/export/pbformat.py` (EXPORT_BLOCKS / EXPORT_HEADS) writing exactly the spec tensors;
  4. the public side FIRST: the type must exist in purebyte/spec/FORMAT.md, the engine and the NumPy reference, or the
     exported file is refused (a file that uses anything the spec does not define is refused, §13);
  5. tests: bit-identity of the untouched presets (training/tests/test_golden_training.py) and export -> reference
     forward (training/tests/test_export_reference.py) for the new type.
"""
from .blocks import AttentionBlock, SSMv2Block
from .heads import ChoiceHead, OrdinalHead, TagHead

BLOCKS = {cls.TYPE: cls for cls in (SSMv2Block, AttentionBlock)}
HEADS = {cls.TYPE: cls for cls in (ChoiceHead, TagHead, OrdinalHead)}
# input modules and post-body modules are declared by ModelConfig fields: `ngram_*` -> n-gram memory (spec §8.1),
# `lookahead` -> lookahead (spec §10).


def head_type(head) -> str:
    t = getattr(type(head), "TYPE", None)
    if t not in HEADS:
        raise ValueError(f"head {type(head).__name__} is not a registered head type (known: {sorted(HEADS)})")
    return t
