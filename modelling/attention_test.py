import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    """
    Causal multi-head self-attention.
    Shapes:
      x: (B, T, C)  with C = d_model
      output: (B, T, C)
    """
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0, bias: bool = False):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Linear projections for Q, K, V in one go (faster & standard)
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=bias)
        # Output projection back to d_model
        self.proj = nn.Linear(d_model, d_model, bias=bias)

        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        # causal mask will be created on the fly based on sequence length

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None):
        """
        x: (B, T, C)
        attn_mask (optional): (T, T) additive mask with 0 for allowed, -inf for disallowed
                              If None, a standard causal mask is used.
        """
        B, T, C = x.shape

        # Project once, then split: (B, T, 3*C) -> 3 * (B, T, C)
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        # Reshape to heads: (B, T, nH, head_dim) -> (B, nH, T, head_dim)
        def reshape_heads(t):
            return t.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        q = reshape_heads(q)
        k = reshape_heads(k)
        v = reshape_heads(v)

        # Scaled dot-product attention
        # attn_scores: (B, nH, T, T)
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)

        # Build/merge causal mask
        # causal_mask: (T, T) with -inf above diagonal (block future positions)
        if attn_mask is None:
            causal = torch.triu(torch.ones(T, T, device=x.device), diagonal=1)
            causal = causal.masked_fill(causal == 1, float('-inf'))
            attn_scores = attn_scores + causal
        else:
            attn_scores = attn_scores + attn_mask  # assume additive mask

        attn = F.softmax(attn_scores, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum of values: (B, nH, T, T) @ (B, nH, T, head_dim) -> (B, nH, T, head_dim)
        y = attn @ v

        # Merge heads: (B, nH, T, head_dim) -> (B, T, C)
        y = y.transpose(1, 2).contiguous().view(B, T, C)

        # Output projection
        y = self.proj(y)
        y = self.proj_drop(y)
        return y

B, T, C = 2, 5, 64
x = torch.randn(B, T, C)
attn = MultiHeadSelfAttention(d_model=C, n_heads=8, dropout=0.1)
y = attn(x)  # (B, T, C)
