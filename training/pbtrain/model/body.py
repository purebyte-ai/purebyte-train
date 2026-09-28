"""ByteModel: the body without heads (spec/FORMAT.md §6).

    X = embed(bytes)  (+ ngram(bytes) before block `ngram_layer`)  -> blocks  -> out_norm  (-> lookahead)

The block of each layer comes from the registry by its public type (`ModelConfig.block_types()`), so a new block type
is a module plus one registry line. Creation order is fixed (same seed, same weights).
"""
import torch.nn as nn

from .config import ModelConfig
from .layers.lookahead import Lookahead
from .layers.ngram import GatedNGramEmbedding, NGramEmbedding
from .layers.norm import RMSNorm


class ByteModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        from .registry import BLOCKS
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(256, cfg.d_model)
        ng_cls = GatedNGramEmbedding if cfg.ngram_gate else NGramEmbedding
        self.ngram = (ng_cls(cfg.ngram_orders, cfg.ngram_buckets, cfg.ngram_dim, cfg.d_model, heads=cfg.ngram_heads)
                      if cfg.ngram_orders else None)
        self.blocks = nn.ModuleList([BLOCKS[t](cfg) for t in cfg.block_types()])
        self.out_norm = RMSNorm(cfg.d_model)
        self.look = Lookahead(cfg.d_model, cfg.lookahead) if cfg.lookahead else None

    def forward(self, ids, state=None, mask=None):
        """ids (B,T) int64 -> (h (B,T,d), per-block states)."""
        if self.look is not None and state is not None:
            raise NotImplementedError("the lookahead reads bytes to the RIGHT: whole windows only, no carried state")
        x = self.embed(ids)
        e = self.ngram(ids) if self.ngram is not None else None
        at = self.cfg.ngram_layer if e is not None else -1
        states = [None] * len(self.blocks) if state is None else state
        new = []
        for i, (blk, st) in enumerate(zip(self.blocks, states)):
            if i == at:
                x = self.ngram.inject(x, e)
            x, st = blk(x, st, mask)
            new.append(st)
        h = self.out_norm(x)
        if self.look is not None:
            h = self.look(h, mask)
        return h, new

    def n_params(self) -> int:
        return sum(p.numel() for p in self.parameters())
