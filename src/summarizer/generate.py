"""Greedy autoregressive decoding, optionally recording final-layer cross-attention."""
import torch


@torch.no_grad()
def greedy_generate(model, src, bos_id, eos_id, max_length=128, return_attention=False):
    model.eval(); src_mask=model.source_mask(src,model.pad_id); memory=model.encode(src,src_mask)
    generated=torch.full((src.size(0),1),bos_id,dtype=torch.long,device=src.device)
    finished=torch.zeros(src.size(0),dtype=torch.bool,device=src.device); attn_steps=[]
    for _ in range(max_length-1):
        if return_attention: logits,weights=model.decode(generated,memory,src_mask,return_cross_attn=True)
        else: logits=model.decode(generated,memory,src_mask)
        nxt=logits[:,-1].argmax(-1); nxt=torch.where(finished,torch.full_like(nxt,eos_id),nxt)
        generated=torch.cat([generated,nxt[:,None]],dim=1)
        if return_attention: attn_steps.append(weights[:,:,-1,:].detach().cpu())
        finished |= nxt.eq(eos_id)
        if finished.all(): break
    return (generated,attn_steps) if return_attention else generated
