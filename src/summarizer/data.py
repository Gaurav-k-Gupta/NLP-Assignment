"""Dataset loading, token cache and length-bucketed batches."""
import os, json, random, gzip
from pathlib import Path
from typing import Optional
from itertools import islice
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm


def load_split(split: str, limit: Optional[int] = None):
    data_dir=os.getenv('DATA_DIR')
    if data_dir:
        path=Path(data_dir)/(split+'.jsonl.gz')
        if path.exists():
            def rows():
                with gzip.open(path,'rt',encoding='utf8') as f:
                    source=(json.loads(line) for line in f if line.strip())
                    yield from islice(source,limit) if limit is not None else source
            return rows()
    from datasets import load_dataset
    try: ds = load_dataset("cnn_dailymail", "3.0.0", split=split)
    except Exception as canonical_error:
        try: ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)
        except Exception as fallback_error:
            raise RuntimeError("Could not load canonical or abisee/cnn_dailymail dataset") from fallback_error
    if limit is not None: ds = ds.select(range(min(limit, len(ds))))
    return ds


def get_tokenizer(name="bert-base-uncased"):
    from transformers import AutoTokenizer
    name=os.getenv('TOKENIZER_DIR',name)
    return AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=Path(name).exists())


class TokenizedDataset(Dataset):
    def __init__(self, rows, tokenizer, max_source=400, max_target=128):
        self.rows=[]; self.tokenizer=tokenizer
        for row in tqdm(rows,desc='Tokenizing',unit='rows'):
            src=tokenizer.encode_article(row["article"], max_source)
            tgt=tokenizer.encode_summary(row["highlights"], max_target)
            self.rows.append((src,tgt,row.get("id", str(len(self.rows))),row["article"],row["highlights"]))
        self.lengths=[len(x[0]) for x in self.rows]
    def __len__(self): return len(self.rows)
    def __getitem__(self, i): return self.rows[i]


class BucketBatchSampler:
    def __init__(self, lengths, batch_size, shuffle=True, bucket_multiplier=50, seed=17):
        self.lengths=lengths; self.batch_size=batch_size; self.shuffle=shuffle
        self.bucket_size=batch_size*bucket_multiplier; self.seed=seed; self.epoch=0
    def __iter__(self):
        order=sorted(range(len(self.lengths)),key=lambda i:self.lengths[i])
        buckets=[order[i:i+self.bucket_size] for i in range(0,len(order),self.bucket_size)]
        rng=random.Random(self.seed+self.epoch); self.epoch+=1
        if self.shuffle:
            for b in buckets: rng.shuffle(b)
            rng.shuffle(buckets)
        batches=[b[i:i+self.batch_size] for b in buckets for i in range(0,len(b),self.batch_size)]
        if self.shuffle: rng.shuffle(batches)
        yield from batches
    def __len__(self): return (len(self.lengths)+self.batch_size-1)//self.batch_size


def collate_batch(rows, pad_id):
    srcs,tgts,ids,articles,refs=zip(*rows)
    src=torch.full((len(rows),max(map(len,srcs))),pad_id,dtype=torch.long)
    tgt=torch.full((len(rows),max(map(len,tgts))),pad_id,dtype=torch.long)
    for i,x in enumerate(srcs): src[i,:len(x)]=torch.tensor(x)
    for i,x in enumerate(tgts): tgt[i,:len(x)]=torch.tensor(x)
    return {"src":src,"tgt":tgt,"ids":ids,"articles":articles,"references":refs}


def make_loader(dataset, batch_size, pad_id, shuffle):
    return DataLoader(dataset, batch_sampler=BucketBatchSampler(dataset.lengths,batch_size,shuffle),
                      collate_fn=lambda rows:collate_batch(rows,pad_id), num_workers=0)


def length_stats(dataset):
    import numpy as np
    def stats(values, limit):
        return {"median":float(np.median(values)),"p90":float(np.percentile(values,90)),
                "max":int(max(values)),"percent_over_limit":100*sum(x>limit for x in values)/len(values)}
    return {"articles":stats([len(r[0]) for r in dataset.rows],400),
            "summaries":stats([len(r[1]) for r in dataset.rows],128),"n":len(dataset)}
