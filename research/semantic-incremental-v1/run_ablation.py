"""Preregistered 2x2x2 factorial; independent processes and three paired seeds."""
import json,subprocess,sys,time,hashlib,itertools,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent

def main():
    config=json.loads((ROOT/'experiment-plan.json').read_text())
    # Chosen before observing outcomes: bounded-update experiment, not convergence.
    config['steps']=600
    (ROOT/'experiment-plan.json').write_text(json.dumps(config,indent=2))
    results=ROOT/'results';results.mkdir(exist_ok=True)
    source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in ROOT.rglob('*.py') if 'results' not in p.parts}
    (ROOT/'source-sha256.json').write_text(json.dumps(source_hashes,indent=2))
    with (ROOT/'tests.log').open('w') as log:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([sys.executable,'-m','hglab.gpu_smoke'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    # Run the complete data path once at small capacity before the expensive matrix.
    ds=config['datasets'][0];folder=ROOT/'data'/ds['name']
    smoke=[sys.executable,'-m','hglab.incremental','--data',ds['data'],'--prefixes',str(folder/'prefixes'),
        '--masks',str(folder/'teachers'),'--output',str(ROOT/'work/smoke'),'--cap','6000','--steps','2','--factors','111']
    with (ROOT/'smoke.log').open('w') as log:subprocess.run(smoke,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
    jobs=[];order=np.random.default_rng(20261007)
    for seed in config['seeds']:
        for ds in config['datasets']:
            variants=[''.join(v) for v in itertools.product('01',repeat=3)];order.shuffle(variants)
            jobs.extend((ds,seed,f) for f in variants)
    (ROOT/'job-order.json').write_text(json.dumps([dict(dataset=d['name'],seed=s,factors=f) for d,s,f in jobs],indent=2))
    completed=[];started=time.time()
    for number,(ds,seed,factors) in enumerate(jobs,1):
        out=results/ds['name']/factors/f'seed-{seed}';out.mkdir(parents=True,exist_ok=True)
        status=dict(state='running',job=number,total=len(jobs),dataset=ds['name'],seed=seed,factors=factors,
            completed=completed,started=started)
        (ROOT/'status.json').write_text(json.dumps(status,indent=2))
        if not (out/'run.json').exists():
            folder=ROOT/'data'/ds['name'];command=[sys.executable,'-m','hglab.incremental',
                '--data',ds['data'],'--prefixes',str(folder/'prefixes'),'--masks',str(folder/'teachers'),
                '--output',str(out),'--factors',factors,'--seed',str(seed),
                '--cap',str(config['cap']),'--width',str(config['width']),'--steps',str(config['steps'])]
            tick=time.perf_counter()
            with (out/'train.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            (out/'process-wall.json').write_text(json.dumps(dict(seconds=time.perf_counter()-tick)))
        completed.append(dict(dataset=ds['name'],seed=seed,factors=factors))
        print('ABLATION_COMPLETE',number,'/',len(jobs),ds['name'],factors,seed,flush=True)
    (ROOT/'status.json').write_text(json.dumps(dict(state='complete',completed=completed,total=len(jobs),seconds=time.time()-started),indent=2))
    print('ALL_48_ABLATIONS_COMPLETE',flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();(ROOT/'failure.txt').write_text(traceback.format_exc());sys.exit(1)
