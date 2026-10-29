import math
import torch
from summarizer.model.attention import scaled_dot_product_attention
from summarizer.model.positional import SinusoidalPositionalEncoding
from summarizer.model.transformer import TransformerSummarizer

def test_attention_shapes_and_mask():
    q=torch.randn(2,2,3,4); k=torch.randn(2,2,4,4); v=torch.randn(2,2,4,4)
    mask=torch.zeros(2,1,1,4,dtype=torch.bool); mask[:,:,:,-1]=True
    out,w=scaled_dot_product_attention(q,k,v,mask)
    assert out.shape==(2,2,3,4) and torch.all(w[:,:,:,-1]==0)

def test_positional_formula():
    pe=SinusoidalPositionalEncoding(4,dropout=0).eval().pe[0]
    assert torch.allclose(pe[2,0],torch.tensor(math.sin(2.0)))
    assert torch.allclose(pe[2,1],torch.tensor(math.cos(2.0)))

def test_model_masks_shapes_and_causality():
    m=TransformerSummarizer(20,0,d_model=8,n_heads=2,d_ff=16,n_encoder_layers=1,n_decoder_layers=1,dropout=0).eval()
    src=torch.tensor([[2,3,0]]); tgt=torch.tensor([[1,4,5]])
    assert m(src,tgt).shape==(1,3,20)
    assert m.target_mask(tgt,0)[0,0,0,1]==False
    assert m.target_mask(tgt,0)[0,0,0,2]==True
    out1=m(src,tgt); changed=tgt.clone(); changed[0,2]=6; out2=m(src,changed)
    assert torch.allclose(out1[:,:2],out2[:,:2],atol=1e-6)
