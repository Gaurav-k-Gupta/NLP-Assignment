"""Post-LayerNorm Transformer encoder/decoder blocks."""
import torch
from torch import nn
from .attention import MultiHeadAttention


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_model, d_ff), nn.ReLU(), nn.Dropout(dropout), nn.Linear(d_ff, d_model))
    def forward(self, x): return self.net(x)


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__(); self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout); self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model)
    def forward(self, x, src_mask):
        x = self.norm1(x + self.drop(self.attn(x, x, x, src_mask)))
        return self.norm2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout):
        super().__init__(); self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout); self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model); self.norm2 = nn.LayerNorm(d_model); self.norm3 = nn.LayerNorm(d_model)
    def forward(self, x, memory, tgt_mask, src_mask, return_cross_attn=False):
        x = self.norm1(x + self.drop(self.self_attn(x, x, x, tgt_mask)))
        result = self.cross_attn(x, memory, memory, src_mask, return_weights=return_cross_attn)
        if return_cross_attn: cross, weights = result
        else: cross = result
        x = self.norm2(x + self.drop(cross)); x = self.norm3(x + self.drop(self.ffn(x)))
        return (x, weights) if return_cross_attn else x
