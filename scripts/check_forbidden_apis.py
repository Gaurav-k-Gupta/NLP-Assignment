"""Fail if assignment-prohibited high-level model APIs appear in project source."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
parts=[('nn.','Transformer'),('nn.','TransformerEncoder'),('nn.','TransformerEncoderLayer'),('nn.','TransformerDecoder'),('nn.','TransformerDecoderLayer'),('nn.','MultiheadAttention'),('F.','scaled_dot_product_attention'),('keras.layers.','MultiHeadAttention'),('Auto','Model'),('Bart','ForConditionalGeneration'),('T5','ForConditionalGeneration'),('Pegasus','ForConditionalGeneration')]
forbidden=[a+b for a,b in parts]
fail=[]
for p in ROOT.rglob('*.py'):
    if p==Path(__file__): continue
    if any(term in p.read_text(encoding='utf8',errors='ignore') for term in forbidden): fail.append(str(p.relative_to(ROOT)))
if fail:
    print('Forbidden API names found:',*fail,sep='\n'); sys.exit(1)
print('Forbidden API check passed.')
