"""Queue fair budget/replay ablations after the main semantic run, no GPU overlap."""
from pathlib import Path
import subprocess,sys,time

root=Path('/content/hierarchical-gaussian')
deadline=time.monotonic()+2400
while not (root/'semantic-field/metrics.json').exists():
    if time.monotonic()>deadline: raise TimeoutError('Main geometry/semantic run did not finish')
    if (Path('/content/kitchen-field.log').exists() and 'Traceback' in Path('/content/kitchen-field.log').read_text()):
        raise RuntimeError('Fix semantic field before running independent benchmarks')
    time.sleep(10)
for name,cap,replay in [('kitchen-300k-l1',300000,.3),('kitchen-600k-no-replay',600000,0.)]:
    with Path('/content/'+name+'.log').open('w') as log:
        subprocess.run([sys.executable,'-u','-m','hglab.train','--data','/content/mip360/kitchen',
            '--output','results/'+name,'--cap',str(cap),'--replay',str(replay),
            '--width','800','--stages','6','--steps','1800'],cwd=root,stdout=log,stderr=subprocess.STDOUT,check=True)
    print('ABLATION_COMPLETE',name,flush=True)
print('ALL_ABLATIONS_COMPLETE',flush=True)
