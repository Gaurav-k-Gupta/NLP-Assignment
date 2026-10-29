"""Collect available result files; missing measurements are explicitly NOT RUN YET."""
from pathlib import Path
import json
root=Path('results'); lines=['# Results asset index','']
for p in sorted(root.glob('*/test_metrics.json')):
    lines += ['## '+p.parent.name,'',p.read_text(),'']
if not list(root.glob('*/test_metrics.json')): lines.append('NOT RUN YET: no test metrics files exist.')
(root/'REPORT_ASSETS.md').write_text('\n'.join(lines))
