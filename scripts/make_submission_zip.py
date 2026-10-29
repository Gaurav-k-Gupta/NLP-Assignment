import argparse,zipfile
from pathlib import Path
p=argparse.ArgumentParser(); p.add_argument('--output',default='submission.zip'); a=p.parse_args()
root=Path(__file__).resolve().parents[1]; out=root/a.output
allowed=['README.md','requirements.txt','.gitignore','configs','docs','src','scripts','tests','notebooks','report','results']
with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
    for item in allowed:
        path=root/item
        if not path.exists(): continue
        files=[path] if path.is_file() else path.rglob('*')
        for f in files:
            if not f.is_file() or any(x in f.parts for x in ('.git','checkpoints','cache','data','__pycache__')): continue
            if f.stat().st_size>20*1024*1024: raise SystemExit('Refusing large file: '+str(f))
            z.write(f,f.relative_to(root))
print(out)
