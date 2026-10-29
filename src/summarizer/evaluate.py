"""Autoregressive test evaluation; test split is accessed only from this module."""
import argparse,gzip,json,os
from pathlib import Path
import torch
from .data import load_split,get_tokenizer,TokenizedDataset,make_loader
from .tokenization import SummaryTokenizer
from .model.transformer import TransformerSummarizer
from .generate import greedy_generate
from .metrics import score

def main():
    p=argparse.ArgumentParser(); p.add_argument('--config',default='configs/small.yaml'); p.add_argument('--run-name',default='run'); p.add_argument('--checkpoint',required=True); p.add_argument('--max-test-samples',type=int); p.add_argument('--batch-size',type=int,default=1); a=p.parse_args()
    import yaml
    cfg=yaml.safe_load(open(a.config)); rawtok=get_tokenizer(); tok=SummaryTokenizer(rawtok); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=TokenizedDataset(load_split('test',a.max_test_samples),tok,cfg['max_source_length'],cfg['max_target_length']); loader=make_loader(ds,a.batch_size,tok.special.pad_id,False)
    model=TransformerSummarizer(len(rawtok),tok.special.pad_id,**{k:cfg[k] for k in ('d_model','n_heads','d_ff','n_encoder_layers','n_decoder_layers','dropout','tie_embeddings')}).to(dev)
    # Checkpoint is a trusted artifact produced by this training code.
    state=torch.load(a.checkpoint,map_location=dev,weights_only=False); model.load_state_dict(state['model']); outdir=Path('results')/a.run_name; outdir.mkdir(parents=True,exist_ok=True)
    preds_path=outdir/'test_predictions.jsonl.gz'; predictions=[]
    with gzip.open(preds_path,'wt',encoding='utf8') as f:
        for b in loader:
            src=b['src'].to(dev)
            if not (outdir/'attention_example.json').exists():
                generated,attn=greedy_generate(model,src,tok.special.bos_id,tok.special.eos_id,cfg['max_target_length'],True)
                (outdir/'attention_example.json').write_text(json.dumps({'source_ids':src[0].cpu().tolist(),'generated_ids':generated[0].cpu().tolist(),'attention':[x[0].tolist() for x in attn]}))
            else: generated=greedy_generate(model,src,tok.special.bos_id,tok.special.eos_id,cfg['max_target_length'])
            for i,ids in enumerate(generated.cpu().tolist()):
                rec={'id':b['ids'][i],'reference':b['references'][i],'prediction':tok.decode_summary(ids)}; predictions.append(rec); f.write(json.dumps(rec,ensure_ascii=False)+'\n')
    metrics=score([x['reference'] for x in predictions],[x['prediction'] for x in predictions])
    payload={'metrics':metrics,'n_examples':len(predictions),'test_set_label':'full test set' if len(predictions)==11490 else 'SUBSET (n=%d)'%len(predictions),'decoding':'plain greedy','tokenizer':'bert-base-uncased','max_source_length':cfg['max_source_length'],'max_target_length':cfg['max_target_length'],'checkpoint_step':state.get('step'),'val_loss':state.get('val_loss')}
    (outdir/'test_metrics.json').write_text(json.dumps(payload,indent=2))
if __name__=='__main__': main()
