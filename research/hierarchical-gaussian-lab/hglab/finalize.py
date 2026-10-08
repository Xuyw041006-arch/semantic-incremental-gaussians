"""Finalize after independent GPU ablations, keeping timing free of competing jobs."""
from pathlib import Path
import subprocess,sys,time
root=Path('/content/hierarchical-gaussian'); deadline=time.monotonic()+1800
while not (root/'results/kitchen-600k-no-replay/metadata.json').exists():
    if time.monotonic()>deadline: raise TimeoutError('Ablation did not complete')
    time.sleep(5)
subprocess.run([sys.executable,'-u','-m','hglab.semantic_field','--checkpoint',
    'results/kitchen-300k-l1/checkpoint.pt','--masks','semantics','--output','semantic-field-300k','--steps','1200'],cwd=root,check=True)
subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=root,check=True)
subprocess.run([sys.executable,'-u','-m','hglab.bundle'],cwd=root,check=True)
print('FINALIZATION_COMPLETE',flush=True)
