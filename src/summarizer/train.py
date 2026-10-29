"""Resumable teacher-forced training with best-validation checkpoint selection."""
import argparse, csv, json, math, os, random, time
from pathlib import Path
import numpy as np
import torch
import yaml
from torch.nn import functional as F
from .data import load_split,get_tokenizer,TokenizedDataset,make_loader,length_stats
from .tokenization import SummaryTokenizer
from .model.transformer import TransformerSummarizer


def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def rng_byte_tensor(state):
    """Normalize checkpoint RNG state to the CPU ByteTensor PyTorch expects."""
    return torch.as_tensor(state, dtype=torch.uint8, device='cpu').contiguous()

def schedule(step,d_model,warmup,multiplier):
    step=max(step,1); return multiplier*(d_model**-0.5)*min(step**-0.5,step*(warmup**-1.5))

def evaluate_loss(model,loader,device,pad_id):
    model.eval(); total=0.; count=0
    with torch.no_grad():
        for batch in loader:
            src=batch['src'].to(device); tgt=batch['tgt'].to(device)
            logits=model(src,tgt[:,:-1]); target=tgt[:,1:]
            loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),target.reshape(-1),ignore_index=pad_id,reduction='sum')
            total+=loss.item(); count+=(target!=pad_id).sum().item()
    avg=total/max(count,1); return avg,math.exp(min(avg,30))

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--config',default='configs/small.yaml'); ap.add_argument('--run-name',default='run')
    ap.add_argument('--max-train-samples',type=int); ap.add_argument('--max-val-samples',type=int)
    ap.add_argument('--epochs',type=int,default=10); ap.add_argument('--patience',type=int,default=3)
    ap.add_argument('--time-budget-minutes',type=float,default=0); ap.add_argument('--seed',type=int,default=17)
    ap.add_argument('--ckpt-dir',default=os.getenv('CKPT_DIR','checkpoints')); args=ap.parse_args()
    cfg=yaml.safe_load(open(args.config)); seed_all(args.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    rawtok=get_tokenizer(); tok=SummaryTokenizer(rawtok); cache=os.getenv('TOKEN_CACHE_DIR','cache'); Path(cache).mkdir(parents=True,exist_ok=True)
    train=TokenizedDataset(load_split('train',args.max_train_samples),tok,cfg['max_source_length'],cfg['max_target_length'])
    val=TokenizedDataset(load_split('validation',args.max_val_samples),tok,cfg['max_source_length'],cfg['max_target_length'])
    tr_loader=make_loader(train,cfg['batch_size'],tok.special.pad_id,True); va_loader=make_loader(val,cfg['batch_size'],tok.special.pad_id,False)
    model=TransformerSummarizer(len(rawtok),tok.special.pad_id,**{k:cfg[k] for k in ('d_model','n_heads','d_ff','n_encoder_layers','n_decoder_layers','dropout','tie_embeddings')}).to(dev)
    opt=torch.optim.Adam(model.parameters(),lr=0,betas=(.9,.98),eps=1e-9); scaler=torch.cuda.amp.GradScaler(enabled=dev.type=='cuda')
    run=Path('results')/args.run_name; run.mkdir(parents=True,exist_ok=True); ckptdir=Path(args.ckpt_dir); ckptdir.mkdir(parents=True,exist_ok=True)
    ckpt=ckptdir/(args.run_name+'_last.pt'); bestpath=ckptdir/(args.run_name+'_best.pt')
    step=0; best=float('inf'); stale=0; start_epoch=0; history=[]; t0=time.time()
    if ckpt.exists():
        # Full resume checkpoints include optimizer and RNG objects; load only trusted files we created.
        state=torch.load(ckpt,map_location=dev,weights_only=False); model.load_state_dict(state['model']); opt.load_state_dict(state['optimizer']); scaler.load_state_dict(state['scaler'])
        step=state['step']; best=state['best']; stale=state['stale']; start_epoch=state['epoch']; history=state['history']
        random.setstate(state['python_rng']); np.random.set_state(state['numpy_rng'])
        torch.set_rng_state(rng_byte_tensor(state['torch_rng']))
        if dev.type=='cuda' and state.get('cuda_rng'):
            torch.cuda.set_rng_state_all([rng_byte_tensor(rng) for rng in state['cuda_rng']])
    for epoch in range(start_epoch,args.epochs):
        model.train()
        for batch in tr_loader:
            src=batch['src'].to(dev); tgt=batch['tgt'].to(dev); step+=1
            lr=schedule(step,cfg['d_model'],cfg['warmup_steps'],cfg['learning_rate_multiplier'])
            for group in opt.param_groups: group['lr']=lr
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=dev.type=='cuda'):
                logits=model(src,tgt[:,:-1]); loss=F.cross_entropy(logits.reshape(-1,logits.size(-1)),tgt[:,1:].reshape(-1),ignore_index=tok.special.pad_id,label_smoothing=cfg.get('label_smoothing',0.0))
            scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),1.0); scaler.step(opt); scaler.update()
            history.append({'step':step,'epoch':epoch+1,'train_loss':loss.item(),'lr':lr,'tokens_per_sec':int((tgt!=tok.special.pad_id).sum().item())/max(time.time()-t0,1e-6),'val_loss':'','val_perplexity':''})
            budget=args.time_budget_minutes*60
            if budget and time.time()-t0>budget:
                save_state(ckpt,model,opt,scaler,step,best,stale,epoch,history); write_history(run,history); return
        val_loss,ppl=evaluate_loss(model,va_loader,dev,tok.special.pad_id)
        history.append({'step':step,'epoch':epoch+1,'train_loss':'','lr':'','tokens_per_sec':'','val_loss':val_loss,'val_perplexity':ppl})
        if val_loss<best:
            best=val_loss; stale=0; torch.save({'model':model.state_dict(),'step':step,'val_loss':best,'config':cfg},bestpath)
        else: stale+=1
        save_state(ckpt,model,opt,scaler,step,best,stale,epoch+1,history); write_history(run,history)
        if stale>=args.patience: break
    # Restore best parameters after training for downstream use.
    if bestpath.exists(): model.load_state_dict(torch.load(bestpath,map_location=dev,weights_only=False)['model'])
    (run/'config.json').write_text(json.dumps(cfg,indent=2)); (run/'run_info.json').write_text(json.dumps({'train_size':len(train),'val_size':len(val),'token_length_stats':length_stats(train),'parameter_count':model.parameter_count(),'device':str(dev),'torch_version':torch.__version__,'cuda_version':torch.version.cuda,'wall_seconds':time.time()-t0},indent=2))

def save_state(path,model,opt,scaler,step,best,stale,epoch,history):
    torch.save({'model':model.state_dict(),'optimizer':opt.state_dict(),'scaler':scaler.state_dict(),'step':step,'best':best,'stale':stale,'epoch':epoch,'history':history,'python_rng':random.getstate(),'numpy_rng':np.random.get_state(),'torch_rng':torch.get_rng_state(),'cuda_rng':torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},path)
def write_history(run,history):
    with (run/'history.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['step','epoch','train_loss','lr','tokens_per_sec','val_loss','val_perplexity']); w.writeheader(); w.writerows(history)
if __name__=='__main__': main()
