"""ModelConfig: the shape of a PureByte byte model, and the named presets.

A checkpoint stores `asdict(cfg)` and `ModelConfig.from_dict` rebuilds it, so fields are never renamed. Fields that name
a variant that was tried and dropped (`init_style` other than "paired", `quant_style` "v1_pertensor", FFN, hierarchy,
patches) are still accepted with their default value, so older checkpoints load, and refused with any other value.

Invariants (checked in `__post_init__`):
- `d_inner = expand * d_model` is a multiple of `d_head`; `n_heads = d_inner / d_head` is a multiple of `n_groups`.
- Ternary matrices need `in % ternary_group == 0` (checked where they quantize: a silent per-row fallback once
  confounded model size with quantization granularity).
- n-gram gate / injection layer need tables.
"""
from dataclasses import dataclass, fields

# name -> (parent preset or None, overrides). A derived preset changes ONE field of its parent, by construction.
PRESETS = {
    # size ladder (pure Mamba-2, causal). `nano` (1.9 M) is the secrets-code size; `tiny` measured level with `small`.
    "pico":   (None, dict(d_model=128, n_layers=8, n_groups=2, chunk_size=64)),
    "nano":   (None, dict(d_model=256, n_layers=4, n_groups=2, chunk_size=64)),
    "tiny":   (None, dict(d_model=256, n_layers=8, n_groups=2, chunk_size=64)),
    "small":  (None, dict(d_model=384, n_layers=12, n_groups=3, chunk_size=64)),
    "medium": (None, dict(d_model=512, n_layers=10, n_groups=4, chunk_size=64)),
    "base":   (None, dict(d_model=768, n_layers=16, n_groups=6, chunk_size=128)),
    # n-gram memory: `tiny_n18` (2^18 buckets x 64, orders 2-4; 50 M table params, 25 MB at 4 bits) is the secrets-bin
    # size (the tables cut false alarms on real binaries by an order of magnitude). `nano_n` / `tiny_n`: 2^16 buckets.
    "nano_n":   (None, dict(d_model=256, n_layers=4, n_groups=2, chunk_size=64,
                            ngram_orders=(2, 3, 4), ngram_buckets=65536, ngram_dim=64)),
    "tiny_n":   (None, dict(d_model=256, n_layers=8, n_groups=2, chunk_size=64,
                            ngram_orders=(2, 3, 4), ngram_buckets=65536, ngram_dim=64)),
    "tiny_n18": (None, dict(d_model=256, n_layers=8, n_groups=2, chunk_size=64,
                            ngram_orders=(2, 3, 4), ngram_buckets=262144, ngram_dim=64)),
    # `nano_n15`: `nano_n` with 2^15 buckets (6.3 M table parameters, about 3.7 MB of tables at 4 bits): the pii size.
    "nano_n15": ("nano_n", dict(ngram_buckets=32768)),
    # one late attention layer (the best bits per byte of the architecture comparison; it wins where SEVERAL facts
    # travel far: extract-kv).
    "nano_a":  (None, dict(d_model=256, n_layers=4, n_groups=2, chunk_size=64, attn_layer=2)),
    "nano_an": ("nano_a", dict(ngram_orders=(2, 3, 4), ngram_buckets=65536, ngram_dim=64)),
    # Architecture candidates (one field on over the parent; zero-init / no RNG: born = the parent).
    "tiny_n18_la16": ("tiny_n18", dict(lookahead=16)),
    "tiny_n18_g":    ("tiny_n18", dict(ngram_gate=True)),
    "tiny_n18_g2":   ("tiny_n18", dict(ngram_gate=True, ngram_layer=2)),
    "nano_la16":     ("nano", dict(lookahead=16)),
    # Base models for small decisions (pure Mamba-2: every CPU fast path of the engine applies).
    "base20":     (None, dict(d_model=512, n_layers=12, n_groups=4, chunk_size=64)),
    "base20_n18": ("base20", dict(ngram_orders=(2, 3, 4), ngram_buckets=262144, ngram_dim=64)),
    "base50":     (None, dict(d_model=768, n_layers=12, n_groups=6, chunk_size=64)),
    "base50_n18": ("base50", dict(ngram_orders=(2, 3, 4), ngram_buckets=262144, ngram_dim=64)),
    # CPU test shapes (not a product size): 64 x 2 layers, group 64. Used by the equivalence and smoke tests.
    "mini64":  (None, dict(d_model=64, n_layers=2, n_groups=2, chunk_size=16, ternary_group=64)),
    "mini64_n": ("mini64", dict(ngram_orders=(2, 3), ngram_buckets=257, ngram_dim=16)),
}

DIRECTIONS = ("causal", "bimamba", "hydra")


@dataclass
class ModelConfig:
    d_model: int
    n_layers: int
    d_head: int = 64
    expand: int = 2                   # d_inner = expand * d_model
    n_groups: int = 2                 # B/C shared by n_heads / n_groups heads
    d_state: int = 64
    conv_width: int = 4               # 0 = no conv1d
    init_style: str = "paired"        # A and dt log-spaced and PAIRED (the only one kept)
    quant_style: str = "absmean_group"
    chunk_size: int = 64              # SSD chunk; memory of the backward ~ b*h*T*chunk
    ternary: bool = True              # False = float control
    ternary_group: int = 128
    tern_thr: float = 1.0             # x the absmean zeroing threshold (1.0 = round)
    direction: str = "causal"         # causal | bimamba | hydra (bidirectional)
    ffn_every: int = 0                # not built (kept for checkpoint compatibility; must stay 0)
    attn_layer: int = -1              # index of the one attention block, -1 = none
    hierarchical: str = "none"        # not built (must stay "none")
    patch_max: int = 0                # not built (must stay 0)
    ngram_orders: tuple = ()
    ngram_buckets: int = 0
    ngram_dim: int = 0
    ngram_heads: int = 1              # hash heads per order (the 4-head variant did not pay)
    lookahead: int = 0                # bytes to the RIGHT seen by every head; 0 = off
    ngram_gate: bool = False          # context gate on the n-gram vector
    ngram_layer: int = 0              # the n-gram vector enters BEFORE this block; 0 = at the input
    A_min: float = 1.0
    A_max: float = 16.0
    dt_min: float = 1e-3
    dt_max: float = 1e-1
    dt_floor: float = 1e-4

    @property
    def d_inner(self) -> int:
        return self.expand * self.d_model

    @property
    def n_heads(self) -> int:
        return self.d_inner // self.d_head

    @property
    def proj_out(self) -> int:
        """Rows of in_proj: z, x, B, C (dt goes separately, in float)."""
        return 2 * self.d_inner + 2 * self.n_groups * self.d_state

    def block_types(self):
        """The public block type of every layer (spec/FORMAT.md §9)."""
        return ["attention" if i == self.attn_layer else "ssm_v2" for i in range(self.n_layers)]

    def __post_init__(self):
        self.ngram_orders = tuple(self.ngram_orders)          # a checkpoint's json brings a list back
        if self.d_inner % self.d_head:
            raise ValueError(f"d_inner {self.d_inner} not divisible by d_head {self.d_head}")
        if self.n_heads % self.n_groups:
            raise ValueError(f"n_heads {self.n_heads} not divisible by n_groups {self.n_groups}")
        lost = {"init_style": (self.init_style, "paired"), "quant_style": (self.quant_style, "absmean_group"),
                "ffn_every": (self.ffn_every, 0), "hierarchical": (self.hierarchical, "none"),
                "patch_max": (self.patch_max, 0)}
        for k, (v, keep) in lost.items():
            if v != keep:
                raise ValueError(f"{k}={v!r} is a variant that is not built; only {keep!r} is")
        if self.direction not in DIRECTIONS:
            raise ValueError(f"unknown direction: {self.direction}; known: {DIRECTIONS}")
        if self.lookahead < 0:
            raise ValueError(f"lookahead must be >= 0, got {self.lookahead}")
        if (self.ngram_gate or self.ngram_layer) and not self.ngram_orders:
            raise ValueError("ngram_gate / ngram_layer need n-gram tables (ngram_orders)")
        if not 0 <= self.ngram_layer < self.n_layers:
            raise ValueError(f"ngram_layer {self.ngram_layer} out of range for {self.n_layers} layers")
        if self.attn_layer >= self.n_layers:
            raise ValueError(f"attn_layer {self.attn_layer} out of range for {self.n_layers} layers")
        if self.attn_layer >= 0 and self.d_model % 64:
            raise ValueError("the attention block uses heads of 64: d_model must be a multiple of 64")

    @classmethod
    def preset(cls, name, **overrides):
        """`ModelConfig.preset("nano", direction="hydra")`: the preset's values, then the caller's overrides."""
        if name not in PRESETS:
            raise ValueError(f"unknown size preset `{name}`; known: {sorted(PRESETS)}")
        values = {}
        chain = []
        while name is not None:
            chain.append(PRESETS[name][1])
            name = PRESETS[name][0]
        for part in reversed(chain):
            values.update(part)
        values.update(overrides)
        return cls(**values)

    @classmethod
    def from_dict(cls, d):
        """Rebuilds a config saved with `asdict` ; unknown keys are refused."""
        known = {f.name for f in fields(cls)}
        extra = sorted(set(d) - known)
        if extra:
            raise ValueError(f"checkpoint config has unknown fields {extra}")
        return cls(**d)
