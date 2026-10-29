"""Scaled dot-product and multi-head attention, implemented from tensor ops."""
import math
from typing import Optional, Tuple
import torch
from torch import nn


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 mask: Optional[torch.Tensor] = None,
                                 dropout: Optional[nn.Dropout] = None) -> Tuple[torch.Tensor, torch.Tensor]:
    """q/k/v: (B,H,Tq/Tk,d); mask broadcasts to (B,H,Tq,Tk), True=blocked."""
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores = scores.masked_fill(mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores.float(), dim=-1)
    if mask is not None:
        weights = weights.masked_fill(mask, 0.0)
        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
    weights = weights.to(dtype=v.dtype)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, v), weights


class MultiHeadAttention(nn.Module):
    """Multi-head attention. Input/output: (B,T,D), weights: (B,H,Tq,Tk)."""
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.d_model, self.n_heads, self.d_k = d_model, n_heads, d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model); self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model); self.w_o = nn.Linear(d_model, d_model)
        self.attention_dropout = nn.Dropout(dropout)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None, return_weights: bool = False):
        b, tq, _ = query.shape; tk = key.size(1)
        def split(x, length):
            return x.view(b, length, self.n_heads, self.d_k).transpose(1, 2)
        q, k, v = split(self.w_q(query), tq), split(self.w_k(key), tk), split(self.w_v(value), tk)
        out, weights = scaled_dot_product_attention(q, k, v, mask, self.attention_dropout)
        out = out.transpose(1, 2).contiguous().view(b, tq, self.d_model)
        out = self.w_o(out)
        return (out, weights) if return_weights else out
