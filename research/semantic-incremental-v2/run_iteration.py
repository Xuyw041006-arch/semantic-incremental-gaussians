"""Small, logged candidate screen; each job runs in an independent CUDA process."""
import argparse,json,subprocess,sys,time,hashlib,itertools,traceback
from pathlib import Path
ROOT=Path(__file__).resolve().parent
INPUT=Path('/content/semantic-incremental')

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',default='screen',choices=['screen','confirmation','factorial','verification']);a=p.parse_args()
    plan=json.loads((ROOT/'iteration-plan.json').read_text())
    sources={str(f.relative_to(ROOT)):hashlib.sha256(f.read_bytes()).hexdigest() for f in ROOT.rglob('*.py') if not any(x in f.parts for x in ['results','work','__pycache__'])}
    dest=ROOT/('source-sha256-'+a.phase+'.json');dest.write_text(json.dumps(sources,indent=2))
    jobs=[]
    datasets=[('tum-desk',str(INPUT/'data/tum640/manifest.json')),('kitchen','/content/mip360/kitchen')]
    if a.phase=='verification':datasets=[('tum-desk2',str(INPUT/'data/desk2-640/manifest.json'))]
    seeds=[7] if a.phase in ['screen','factorial'] else [17] if a.phase=='confirmation' else [7,17]
    factors=['000','111'] if a.phase in ['screen','verification'] else [''.join(x) for x in itertools.product('01',repeat=3)]
    order=__import__('numpy').random.default_rng(20261008)
    for seed in seeds:
        for ds,data in datasets:
            shuffled=factors.copy();order.shuffle(shuffled)
            jobs.extend(dict(dataset=ds,data=data,seed=seed,factors=f) for f in shuffled)
    (ROOT/(a.phase+'-jobs.json')).write_text(json.dumps(jobs,indent=2))
    completed=[];start=time.time()
    for j in jobs:
        status=dict(phase=a.phase,state='running',completed=completed,total=len(jobs),current=j,started=start)
        (ROOT/'status.json').write_text(json.dumps(status,indent=2))
        out=ROOT/'results'/j['dataset']/j['factors']/f"seed-{j['seed']}";out.mkdir(parents=True,exist_ok=True)
        if not (out/'run.json').exists():
            dsroot=INPUT/'data'/j['dataset']
            command=[sys.executable,'-m','hglab.incremental','--data',j['data'],'--prefixes',str(dsroot/'prefixes'),'--masks',str(dsroot/'teachers'),
                '--output',str(out),'--factors',j['factors'],'--seed',str(j['seed']),'--cap',str(plan['cap']),'--width',str(plan['width']),
                '--steps',str(plan['steps']),'--semantic-strength',str(plan['semantic_strength']),'--replay-max',str(plan['replay_max'])]
            tick=time.perf_counter()
            with (out/'train.log').open('w') as log:subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
            (out/'process-wall.json').write_text(json.dumps(dict(seconds=time.perf_counter()-tick)))
        history=json.loads((out/'metrics.json').read_text())
        print('ITERATION_COMPLETE',json.dumps(dict(**j,metrics=history[-1])),flush=True)
        completed.append(j)
    (ROOT/'status.json').write_text(json.dumps(dict(phase=a.phase,state='complete',completed=completed,total=len(jobs),seconds=time.time()-start),indent=2))
    print('PHASE_COMPLETE',a.phase,flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();(ROOT/'failure.txt').write_text(traceback.format_exc());sys.exit(1)
