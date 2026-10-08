"""Readiness gate, source audit, CUDA smoke and initial paired candidate screen."""
from pathlib import Path
import subprocess,sys,time,hashlib,json
ROOT=Path(__file__).resolve().parent
ready=Path('/content/l4-ready.json')
for _ in range(180):
 if ready.exists():break
 time.sleep(5)
else:raise RuntimeError('L4 bootstrap did not complete in 15 minutes; inspect /content/l4-bootstrap.log')
state=json.loads(ready.read_text());assert state['exit']==0,state
print('L4_INPUT_PREPARATION',flush=True)
subprocess.run([sys.executable,'prepare_l4.py'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'-m','hglab.gpu_smoke'],cwd=ROOT,check=True)
subprocess.run([sys.executable,'run_iteration.py','--phase','screen'],cwd=ROOT,check=True)
print('SCREEN_PIPELINE_COMPLETE',flush=True)
