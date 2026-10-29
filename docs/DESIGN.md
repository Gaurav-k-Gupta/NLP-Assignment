# Design notes: CNN/DailyMail Transformer summarizer

## Scope and choices

Q3 only. Use `bert-base-uncased` as a tokenizer, never as a model; its reserved `[unused0]` and `[unused1]` ids serve as BOS and NL, while `[CLS]` and `[SEP]` are BOS and EOS. PAD comes from the tokenizer. CNN/DailyMail highlights are split on newline before encoding so sentence boundaries survive WordPiece. Default limits are 400 source tokens and 128 target tokens including BOS/EOS. A single shared WordPiece vocabulary is used.

## Shapes and masks

Let source `src` be `(B,S)`, decoder input `tgt` `(B,T)`, model width `D`, heads `H`, and `d_k=D/H`. Embedding lookup and sinusoidal encoding produce `(B,S,D)` / `(B,T,D)`, scaled by `sqrt(D)`. Each projection creates `(B,L,D)`, reshaped/transposed to `(B,H,L,d_k)`. Attention scores are `(B,H,Tq,Tk)`; softmax is float32 and the weighted values return `(B,H,Tq,d_k)`, concatenated to `(B,Tq,D)`, then output-projected. FFN maps `(B,L,D)->(B,L,d_ff)->(B,L,D)`. Encoder memory is `(B,S,D)`; final decoder logits are `(B,T,V)`.

Masks are boolean where `True` means blocked. Source padding is `(B,1,1,S)` and blocks keys that equal PAD in encoder self-attention and cross-attention. Target padding is `(B,1,1,T)` and blocks PAD keys. Causal mask is `(1,1,T,T)` with `j>i`; their OR blocks future and padded keys in decoder self-attention. Query padding need not be masked because padded target positions are excluded from loss. Fully blocked rows are handled by zeroing their probabilities after softmax.

## Formula-to-code map

`scaled_dot_product_attention` computes `softmax(QK^T/sqrt(d_k)+mask)V`. `MultiHeadAttention` owns Wq/Wk/Wv/Wo. `SinusoidalPositionalEncoding` implements the paper's sine/cosine equations. `PositionwiseFeedForward` is Linear-ReLU-Dropout-Linear. `EncoderLayer` and `DecoderLayer` use post-LN exactly: `LayerNorm(x + Dropout(Sublayer(x)))`. Shared embeddings may be tied to the vocabulary projection. All Linear weights use Xavier initialization.

## Training and generation

Training uses decoder input `[BOS]+summary[:-1]` and targets `summary+[EOS]`. Teacher forcing computes all target positions in parallel; the causal mask prevents any position from reading future gold tokens, so this is equivalent to sequential next-token prediction. Cross entropy ignores PAD. Validation always uses unsmoothed CE and selects the lowest-loss checkpoint. Inference encodes once, then greedily appends one token per step until EOS or the configured length; it recomputes decoder prefixes and has no KV cache. Autoregressive generation, not teacher-forced logits, feeds evaluation and visualization.

## T4 estimates (not measurements)

Default `small`: D=256, H=8, 3 encoder and 3 decoder layers, FFN=1024, dropout=0.1. Embedding/projection alone is about `V*D` parameters (roughly 7.8M at a 30.5k vocabulary when tied); attention and FFN add roughly 7.1M, so total is around 15M. These are estimates only; code records the actual count. At S=400 and T=128, attention activations scale with `B*H*(S^2+T^2+TS)` and feed-forward activations with `B*(S+T)*D*d_ff`; batch size should be benchmarked on the actual T4. AMP and gradient accumulation provide memory controls. Throughput, epoch time, and convergence are not predictable without measurement and must be established by `scripts/benchmark.py` and real runs.
