import torch.nn as nn


class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, pad_id: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)

    def forward(self, ids):
        # ids: (B, L) → returns (B, L, d_model)
        return self.emb(ids)
