"""Wait for the frozen matrix, audit it, benchmark the teacher, preserve evidence."""
from pathlib import Path
import hashlib,json,subprocess,sys,time,traceback,zipfile
ROOT=Path(__file__).resolve().parent

def main():
    while True:
        if (ROOT/'failure.txt').exists():raise RuntimeError((ROOT/'failure.txt').read_text())
        try:status=json.loads((ROOT/'status.json').read_text())
        except (FileNotFoundError,json.JSONDecodeError):time.sleep(10);continue
        if status['state']=='complete':break
        time.sleep(10)
    with (ROOT/'analysis-tests.log').open('w') as f:
        subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([sys.executable,'analyze_results.py','--checkpoints'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'make_comparisons.py'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'benchmark_teacher.py'],cwd=ROOT,check=True)
    provenance={}
    for ds in ('tum-desk','kitchen'):
        folder=ROOT/'data'/ds;entries=[]
        for row in json.loads((folder/'teachers'/'index.json').read_text()):
            e=dict(index=row['index'],test=row['test'],reused=row['reused'],cache_seconds=row['seconds'])
            if row['reused']:
                p=Path('/content/hierarchical-gaussian/semantics')/f'masks-{row["index"]:04d}.json'
                if p.exists():e['original_teacher_seconds']=json.loads(p.read_text())['seconds']
            entries.append(e)
        provenance[ds]=dict(teachers=entries,prefix_preparation=json.loads((folder/'prefixes'/'preparation.json').read_text()))
    subprocess.run([sys.executable,'-m','pip','freeze'],stdout=open(ROOT/'environment-pip-freeze.txt','w'),check=True)
    provenance['nvidia_smi']=subprocess.check_output(['nvidia-smi'],text=True)
    import platform,torch
    provenance['runtime']=dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda)
    weights=list(Path('/content/hierarchical-gaussian').rglob('sam2.1_hiera_tiny.pt'))
    if weights:provenance['sam_checkpoint_sha256']=hashlib.sha256(weights[0].read_bytes()).hexdigest()
    provenance['input_sha256']={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in (ROOT/'data').rglob('*') if p.is_file() and p.suffix in ('.json','.npz','.npy')}
    source_files=[]
    plan=json.loads((ROOT/'experiment-plan.json').read_text())
    for ds in plan['datasets']:
        data=Path(ds['data'])
        if data.suffix=='.json':
            source_files.append(data)
            for frame in json.loads(data.read_text())['frames']:
                source_files.extend([data.parent/frame['rgb'],data.parent/frame['depth']])
        else:
            images=data/'images_4' if (data/'images_4').exists() else data/'images'
            source_files.extend(p for p in images.iterdir() if p.is_file())
            source_files.extend(p for p in (data/'sparse').rglob('*.bin'))
    provenance['original_posed_input_sha256']={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in source_files}
    adapter=Path('/content/semantic-gaussian-real/sglab/tum.py')
    if adapter.exists():provenance['external_tum_adapter_sha256']=hashlib.sha256(adapter.read_bytes()).hexdigest()
    provenance['qualification']='Kitchen reuses 41 SAM/CLIP training caches generated in earlier 779px experiment; masks resized to 640px; all variants share identical immutable inputs. Newly generated teacher views use 640px.'
    (ROOT/'experiment-provenance.json').write_text(json.dumps(provenance,indent=2))
    subprocess.run([sys.executable,'write_report.py'],cwd=ROOT,check=True)
    files=[]
    for p in ROOT.rglob('*'):
        if not p.is_file() or '__pycache__' in p.parts or 'work' in p.parts:continue
        if p.name in ('artifact-manifest.json','finish-status.json'):continue
        rel=p.relative_to(ROOT)
        if p.name=='checkpoint.pt':
            if p.parent.name!='seed-7' or p.parent.parent.name not in ('000','111'):continue
        elif p.suffix not in ('.py','.md','.txt','.json','.csv','.png','.jpg','.npz','.npy','.log','.ipynb'):continue
        if rel.parts[0]=='data' and rel.parts[1]=='tum640' and p.name!='manifest.json':continue
        files.append(p)
    manifest={str(p.relative_to(ROOT)):dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in files}
    (ROOT/'artifact-manifest.json').write_text(json.dumps(manifest,indent=2));files.append(ROOT/'artifact-manifest.json')
    target=Path('/content/semantic-incremental-results.zip')
    with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
        for p in files:z.write(p,'semantic-incremental-ablation/'+str(p.relative_to(ROOT)))
    completed=dict(state='complete',runs=len(status['completed']),artifact=str(target),bytes=target.stat().st_size,
        sha256=hashlib.sha256(target.read_bytes()).hexdigest(),archive_contains_all_raw_metrics=True,
        selected_checkpoints=f"Four seed-7 000/111 checkpoints and maps; other {len(status['completed'])-4} planned checkpoints remain in Colab runtime",
        report=str(ROOT/'EXPERIMENT_REPORT.md'))
    (ROOT/'finish-status.json').write_text(json.dumps(completed,indent=2));print(json.dumps(completed,indent=2),flush=True)

if __name__=='__main__':
    try:main()
    except Exception:
        traceback.print_exc();(ROOT/'finish-failure.txt').write_text(traceback.format_exc());sys.exit(1)
