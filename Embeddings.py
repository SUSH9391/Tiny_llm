import torch
import torch.nn as nn

class Embeddings(nn.Module):
    """Learned token + positional embeddings (vanilla nanoGPT)."""

    def __init__(self, cfg):
        super().__init__()
        self.tok = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        # Position ids live in a buffer, NOT torch.arange(..., device=...) in
        # forward: tracing would bake the device as a constant and the frozen
        # artifact could then only ever run on the device it was traced on.
        # Buffers are remapped by map_location, so the model stays portable.
        self.register_buffer("pos_ids", torch.arange(cfg.block_size))

    def forward(self, idx):
        tok_emb = self.tok(idx)
        pos_emb = self.pos(self.pos_ids[:idx.size(1)])  # (T, C)
        return self.drop(tok_emb + pos_emb)
