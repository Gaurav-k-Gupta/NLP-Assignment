"""Full encoder-decoder Transformer assembled from local components only."""
import math
import torch
from torch import nn
from .layers import EncoderLayer, DecoderLayer
from .positional import SinusoidalPositionalEncoding


class TransformerSummarizer(nn.Module):
    def __init__(self, vocab_size: int, pad_id: int, d_model=256, n_heads=8, d_ff=1024,
                 n_encoder_layers=3, n_decoder_layers=3, dropout=0.1, tie_embeddings=True):
        super().__init__(); self.pad_id = pad_id; self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.src_position = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.tgt_position = SinusoidalPositionalEncoding(d_model, dropout=dropout)
        self.encoder = nn.ModuleList([EncoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(n_encoder_layers)])
        self.decoder = nn.ModuleList([DecoderLayer(d_model,n_heads,d_ff,dropout) for _ in range(n_decoder_layers)])
        self.output = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings: self.output.weight = self.embedding.weight
        self.apply(self._init_weights)
        if tie_embeddings: self.output.weight = self.embedding.weight

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear): nn.init.xavier_uniform_(module.weight)
        elif isinstance(module, nn.Embedding): nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None: nn.init.zeros_(module.bias)

    @staticmethod
    def source_mask(src, pad_id): return (src == pad_id)[:, None, None, :]
    @staticmethod
    def target_mask(tgt, pad_id):
        length = tgt.size(1); causal = torch.triu(torch.ones(length,length,dtype=torch.bool,device=tgt.device), diagonal=1)
        padding = (tgt == pad_id)[:, None, None, :]
        return causal[None,None,:,:] | padding

    def encode(self, src, src_mask=None):
        if src_mask is None: src_mask = self.source_mask(src, self.pad_id)
        x = self.src_position(self.embedding(src))
        for layer in self.encoder: x = layer(x, src_mask)
        return x

    def decode(self, tgt_in, memory, src_mask, tgt_mask=None, return_cross_attn=False):
        if tgt_mask is None: tgt_mask = self.target_mask(tgt_in, self.pad_id)
        x = self.tgt_position(self.embedding(tgt_in)); weights = None
        for i, layer in enumerate(self.decoder):
            want = return_cross_attn and i == len(self.decoder)-1
            result = layer(x, memory, tgt_mask, src_mask, want)
            if want: x, weights = result
            else: x = result
        logits = self.output(x)
        return (logits, weights) if return_cross_attn else logits

    def forward(self, src, tgt_in):
        src_mask = self.source_mask(src, self.pad_id); memory = self.encode(src, src_mask)
        return self.decode(tgt_in, memory, src_mask, self.target_mask(tgt_in, self.pad_id))

    def parameter_count(self): return sum(p.numel() for p in self.parameters() if p.requires_grad)
