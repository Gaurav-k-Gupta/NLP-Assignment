"""Plot final-layer cross-attention saved during autoregressive generation."""
import argparse,json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
from summarizer.data import get_tokenizer
p=argparse.ArgumentParser(); p.add_argument('--run-name',default='run'); a=p.parse_args()
root=Path('results')/a.run_name; item=json.loads((root/'attention_example.json').read_text()); tok=get_tokenizer()
source=[x for x in item['source_ids'] if x!=tok.pad_token_id]; output=[]
for x in item['generated_ids']:
    if x in (tok.sep_token_id,tok.pad_token_id): break
    if x!=tok.cls_token_id: output.append(x)
labels=tok.convert_ids_to_tokens(source); rows=item['attention'][:len(output)]
matrix=np.asarray([np.asarray(r).mean(axis=0)[:len(source)] for r in rows]); width=min(75,len(source)); start=max(range(max(len(source)-width+1,1)),key=lambda i:matrix[:,i:i+width].sum()); mass=matrix[:,start:start+width].sum()/max(matrix.sum(),1e-12)
summary=tok.convert_ids_to_tokens(output)
fig,ax=plt.subplots(figsize=(16,7)); ax.imshow(matrix[:,start:start+width],aspect='auto',interpolation='nearest'); ax.set_xticks(range(width)); ax.set_xticklabels(labels[start:start+width],rotation=90,fontsize=7); ax.set_yticks(range(len(summary))); ax.set_yticklabels(summary,fontsize=7); ax.set_title('Autoregressive cross-attention averaged over heads; window mass=%.3f'%mass); fig.tight_layout(); fig.savefig(root/'attention_mean_heads.png',dpi=160); plt.close(fig)
perhead=np.asarray(rows) # (steps, heads, source)
fig,axes=plt.subplots(perhead.shape[1],1,figsize=(16,3*perhead.shape[1]),squeeze=False)
for h,ax in enumerate(axes[:,0]):
    ax.imshow(perhead[:,h,start:start+width],aspect='auto',interpolation='nearest'); ax.set_xticks(range(width)); ax.set_xticklabels(labels[start:start+width],rotation=90,fontsize=6); ax.set_yticks(range(len(summary))); ax.set_yticklabels(summary,fontsize=6); ax.set_title('Head %d'%h)
fig.tight_layout(); fig.savefig(root/'attention_each_head.png',dpi=160); plt.close(fig)
