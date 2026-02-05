import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, mask_future=False):
        super().__init__()
        self.mask_future = mask_future

    def forward(self, q, k, v, mask=None):
        d_k = q.size(-1)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            mask = mask.to(dtype=scores.dtype, device=scores.device)
            if mask.dim() == 2:
                mask = mask.unsqueeze(1)
            mask = torch.where(mask == 0, torch.tensor(-1e9, device=scores.device, dtype=scores.dtype), 0.0)
            scores = scores + mask
        if self.mask_future:
            Q_len, K_len = scores.size(-2), scores.size(-1)
            # upper triangle (strictly above diagonal) -> True where j > i
            future_bool = torch.triu(torch.ones(Q_len, K_len, device=scores.device, dtype=torch.bool), diagonal=1)
            # convert to additive mask with big negatives; shape (1, Q, K) to broadcast over batch
            future_mask = torch.where(future_bool, torch.tensor(-1e9, device=scores.device, dtype=scores.dtype), 0.0).unsqueeze(0)
            scores = scores + future_mask
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        return output


class PositionWiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))

    def forward(self, x):
        return self.ffn(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, mask_future: bool = False, dropout: float = 0.1):
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.query_transform = nn.Linear(d_model, d_model, bias=False)
        self.key_transform = nn.Linear(d_model, d_model, bias=False)
        self.value_transform = nn.Linear(d_model, d_model, bias=False)
        self.output_transform = nn.Linear(d_model, d_model, bias=False)

        self.attention = Attention(mask_future=mask_future)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        batch_size, q_len, _ = q.size()
        _, k_len, _ = k.size()

        # 1) Linear projections
        Q = self.query_transform(q)  # (B, q_len, d_model)
        K = self.key_transform(k)  # (B, k_len, d_model)
        V = self.value_transform(v)  # (B, k_len, d_model)

        # 2) Split into heads: (B, num_heads, seq, head_dim)
        Q = Q.view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, k_len, self.num_heads, self.head_dim).transpose(1, 2)

        # 3) Merge heads into batch: (B * num_heads, seq, head_dim)
        Q = Q.reshape(batch_size * self.num_heads, q_len, self.head_dim)
        K = K.reshape(batch_size * self.num_heads, k_len, self.head_dim)
        V = V.reshape(batch_size * self.num_heads, k_len, self.head_dim)

        # 4) Broadcast mask over heads
        if mask is not None:
            # mask initially (B, seq_len) or (B, q_len, k_len)
            mask = mask.repeat_interleave(self.num_heads, dim=0)

        # 5) Scaled dot-product attention
        out = self.attention(Q, K, V, mask=mask)  # (B * num_heads, q_len, head_dim)

        # 6) Reshape back: (B, q_len, num_heads, head_dim) -> (B, q_len, d_model)
        out = out.view(batch_size, self.num_heads, q_len, self.head_dim).transpose(1, 2)
        out = out.reshape(batch_size, q_len, self.d_model)

        # 7) Final linear projection
        out = self.output_transform(out)
        out = self.dropout(out)
        return out
