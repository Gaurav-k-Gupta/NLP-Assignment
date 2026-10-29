"""Select fixed-seed examples and simple automatic repetition/number failure flags."""
import argparse,gzip,json,random,re
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--run-name',default='run'); a=p.parse_args(); root=Path('results')/a.run_name
rows=[json.loads(x) for x in gzip.open(root/'test_predictions.jsonl.gz','rt',encoding='utf8')]
from summarizer.data import load_split
data=load_split('test')
articles={str(row['id']):row['article'] for row in data}
for row in rows: row['article']=articles.get(str(row['id']),'')
random.Random(17).shuffle(rows); selected=rows[:5]
def is_failure(r):
    words=re.findall(r'\b\w+\b',r['prediction'].lower())
    nums=set(re.findall(r'\b\d+\b',r['prediction'])); source=set(re.findall(r'\b\d+\b',r['article']))
    repeated=len(words)-len(set(words)) >= 3
    unsupported=bool(nums-source)
    return repeated or unsupported
failures=[r for r in rows if is_failure(r)][:3]
chosen=selected+failures
def block(r):
    nums=set(re.findall(r'\b\d+\b',r['prediction'])); source=set(re.findall(r'\b\d+\b',r['article'])) if 'article' in r else set()
    repeated=bool(re.search(r'\b(\w+)(?:\s+\w+){0,3}\s+\1\b',r['prediction'],re.I))
    return '\n'.join(['## Example','', '**Article excerpt:** '+r.get('article','(article text not saved in evaluation output)')[:900], '', '**Reference:** '+r['reference'],'','**Generated:** '+r['prediction'],'','**Flags:** repetition=%s; numbers=%s'%(repeated,sorted(nums)),''])
(root/'qualitative_examples.md').write_text('# Fixed-seed qualitative examples (seed=17)\n\n'+'\n'.join(block(r) for r in chosen),encoding='utf8')
