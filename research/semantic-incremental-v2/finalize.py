"""Audit all completed L4 groups, create figures, report and portable evidence."""
from pathlib import Path
import json,itertools,subprocess,sys,hashlib,time,zipfile,platform,shutil
import numpy as np
AUX=Path(__file__).resolve().parent
ROOT=Path.cwd();INPUT=Path('/content/semantic-incremental')

def save(p,x):Path(p).write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 verification=json.loads((ROOT/'verification-status.json').read_text());assert verification['state']=='complete' and len(verification['completed'])==4
 for file in ROOT.glob('source-sha256-*.json'):
  for name,digest in json.loads(file.read_text()).items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,(file.name,name)
 if not (ROOT/'data').exists():(ROOT/'data').symlink_to(INPUT/'data',target_is_directory=True)
 plan=json.loads((INPUT/'experiment-plan.json').read_text());plan['seeds']=[7,17]
 save(ROOT/'experiment-plan.json',plan)
 completed=[dict(dataset=d['name'],seed=s,factors=''.join(f)) for d in plan['datasets'] for s in plan['seeds'] for f in itertools.product('01',repeat=3)]
 assert all((ROOT/'results'/x['dataset']/x['factors']/f"seed-{x['seed']}"/'run.json').exists() for x in completed)
 save(ROOT/'status.json',dict(state='complete',total=32,completed=completed,phase='final_audit'))
 shutil.copy(ROOT/'source-sha256-confirmation.json',ROOT/'source-sha256.json')
 subprocess.run([sys.executable,str(AUX/'analyze_results.py'),'--checkpoints'],cwd=ROOT,check=True)
 subprocess.run([sys.executable,str(AUX/'run_replay_control.py')],cwd=ROOT,check=True)
 save(ROOT/'status.json',dict(state='complete',total=32,completed=completed,phase='final_audit'))
 if not all((ROOT/'analysis'/f'{ds}-visual-{view}.png').exists() for ds in ['tum-desk','kitchen','tum-desk2'] for view in ['early','late']):
  subprocess.run([sys.executable,str(AUX/'make_comparisons.py')],cwd=ROOT,check=True)
 if not (ROOT/'analysis/depth-summary.json').exists():subprocess.run([sys.executable,str(AUX/'audit_depth.py')],cwd=ROOT,check=True)
 if not (ROOT/'teacher-benchmark.json').exists():subprocess.run([sys.executable,str(AUX/'benchmark_teacher.py')],cwd=ROOT,check=True)
 from analyze_results import audit_run,audit_prefix,csv_write
 audit_prefix(INPUT/'data/tum-desk2/prefixes',plan)
 rows=[]
 for seed in [7,17]:
  for factors in ['000','111']:
   folder=ROOT/'results/tum-desk2'/factors/f'seed-{seed}'
   row,history,check=audit_run(folder,plan,'tum-desk2',seed,factors,True);rows.append(row)
   assert check is not None
 csv_write(ROOT/'analysis/verification-runs.csv',rows);save(ROOT/'analysis/verification-runs.json',rows)
 differences={k:[next(r[k] for r in rows if r['seed']==s and r['factors']=='111')-next(r[k] for r in rows if r['seed']==s and r['factors']=='000') for s in [7,17]] for k in ['psnr','old_psnr','fine_sam_miou','small_fine_sam_miou']}
 save(ROOT/'analysis/verification-summary.json',dict(independent_sequence_not_environment=True,seeds=[7,17],paired_deltas=differences,means={k:float(np.mean(v)) for k,v in differences.items()}))
 (ROOT/'monitor-stop').write_text('final audit complete')
 with (ROOT/'environment-pip-freeze.txt').open('w') as f:subprocess.run([sys.executable,'-m','pip','freeze'],stdout=f,check=True)
 import torch
 save(ROOT/'environment.json',dict(python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(),nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True)))
 subprocess.run([sys.executable,str(AUX/'write_report_html.py')],cwd=ROOT,check=True)
 logs=ROOT/'logs';logs.mkdir(exist_ok=True)
 for name in ['l4-bootstrap.log','l4-screen.log','l4-coordinate.log','l4-finalize-retry.log','l4-monitor.log','desk2-download.log']:
  p=Path('/content')/name
  if p.exists():shutil.copy2(p,logs/name)
 provenance=ROOT/'input-provenance';provenance.mkdir(exist_ok=True)
 for name in ['experiment-provenance.json','protocol-amendment.json','experiment-plan-original-48.json','source-sha256.json']:
  p=INPUT/name
  if p.exists():shutil.copy2(p,provenance/name)
 save(ROOT/'auxiliary-source-sha256.json',{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in AUX.glob('*.py')})
 files=[p for p in ROOT.rglob('*') if p.is_file() and '__pycache__' not in p.parts and 'work' not in p.parts and 'data' not in p.relative_to(ROOT).parts and p.name not in ['artifact-manifest.json','finish-status.json']]
 chosen=[]
 for p in files:
  if p.name=='checkpoint.pt' and (p.parent.name!='seed-7' or p.parent.parent.name!='111'):continue
  chosen.append((p,'semantic-incremental-v2/'+str(p.relative_to(ROOT))))
 for ds in ['tum-desk','kitchen','tum-desk2']:
  for p in (INPUT/'data'/ds).rglob('*'):
   if p.is_file():chosen.append((p,'semantic-incremental-v2/data/'+str(p.relative_to(INPUT/'data'))))
 for folder in ['tum640','desk2-640']:
  p=INPUT/'data'/folder/'manifest.json';chosen.append((p,'semantic-incremental-v2/data/'+folder+'/manifest.json'))
 if AUX!=ROOT:
  for p in AUX.glob('*.py'):chosen.append((p,'semantic-incremental-v2/'+p.name))
 manifest={name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p,name in chosen}
 save(ROOT/'artifact-manifest.json',manifest);chosen.append((ROOT/'artifact-manifest.json','semantic-incremental-v2/artifact-manifest.json'))
 target=Path('/content/semantic-incremental-v2-results.zip')
 with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED,compresslevel=3) as z:
  for p,name in chosen:z.write(p,name)
 save(ROOT/'finish-status.json',dict(state='complete',main_runs=32,verification_runs=4,matched_replay_control_runs=4,total_completed_runs=40,artifact=str(target),bytes=target.stat().st_size,sha256=hashlib.sha256(target.read_bytes()).hexdigest(),checkpoints_preserved='Final full-model seed-7 checkpoints for all three recordings; raw metrics for every variant and seed',report=str(ROOT/'RESULTS.html')))
 print((ROOT/'finish-status.json').read_text(),flush=True)
if __name__=='__main__':main()
