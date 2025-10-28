import torch
import torch.nn as nn
import torch.nn.functional as F
import math

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
