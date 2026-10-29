import argparse,csv
from pathlib import Path
import matplotlib.pyplot as plt
p=argparse.ArgumentParser(); p.add_argument('run_name'); a=p.parse_args(); path=Path('results')/a.run_name/'history.csv'
rows=list(csv.DictReader(path.open())); tr=[(int(r['step']),float(r['train_loss'])) for r in rows if r['train_loss']]
va=[(int(r['step']),float(r['val_loss'])) for r in rows if r['val_loss']]
plt.plot(*zip(*tr),label='train'); plt.plot(*zip(*va),marker='o',label='validation')
if va:
    best=min(va,key=lambda x:x[1]); plt.axvline(best[0],ls='--',label='best validation step')
plt.xlabel('Step'); plt.ylabel('Loss'); plt.legend(); plt.tight_layout(); plt.savefig(path.with_name('loss_curves.png'),dpi=160)
