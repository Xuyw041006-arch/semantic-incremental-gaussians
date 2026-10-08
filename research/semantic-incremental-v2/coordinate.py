"""Continue only after explicit screen/confirmation checks; preserve every phase."""
from pathlib import Path
import subprocess,sys,json,time,traceback,numpy as np
AUX=Path(__file__).resolve().parent
ROOT=Path.cwd()

def phase_record(name):
 s=json.loads((ROOT/'status.json').read_text());assert s['state']=='complete'
 (ROOT/(name+'-status.json')).write_text(json.dumps(s,indent=2))
 return s

def paired_check(seed_list):
 rows=[]
 for ds in ['tum-desk','kitchen']:
  deltas=[]
  for seed in seed_list:
   a=json.loads((ROOT/'results'/ds/'000'/f'seed-{seed}'/'metrics.json').read_text())[-1]
   b=json.loads((ROOT/'results'/ds/'111'/f'seed-{seed}'/'metrics.json').read_text())[-1]
   deltas.append({k:b[k]-a[k] for k in ['psnr','old_psnr','fine_sam_miou']})
  rows.append(dict(dataset=ds,delta={k:float(np.mean([d[k] for d in deltas])) for k in deltas[0]},paired=deltas))
 passed=bool(np.mean([r['delta']['psnr'] for r in rows])>0 and min(r['delta']['psnr'] for r in rows)>=-.1 and np.mean([r['delta']['old_psnr'] for r in rows])>0 and min(r['delta']['fine_sam_miou'] for r in rows)>=-.015)
 return dict(passed=passed,seeds=seed_list,results=rows)

def main():
 while True:
  if (ROOT/'failure.txt').exists():raise RuntimeError((ROOT/'failure.txt').read_text())
  try:s=json.loads((ROOT/'status.json').read_text())
  except (FileNotFoundError,json.JSONDecodeError):time.sleep(5);continue
  if s['phase']=='screen' and s['state']=='complete':break
  time.sleep(5)
 phase_record('screen')
 subprocess.run([sys.executable,str(AUX/'summarize_screen.py')],cwd=ROOT,check=True)
 screen=json.loads((ROOT/'screen-summary.json').read_text())
 if not screen['passed']:
  (ROOT/'attention-needed.json').write_text(json.dumps(dict(reason='Candidate did not meet screen thresholds; inspect and revise a new version.'),indent=2));print('SCREEN_NEEDS_REVISION',flush=True);return
 print('SCREEN_ACCEPTED_FROZEN_CANDIDATE',flush=True)
 for phase in ['factorial','confirmation']:
  subprocess.run([sys.executable,'run_iteration.py','--phase',phase],cwd=ROOT,check=True)
  phase_record(phase)
 confirmation=paired_check([7,17]);confirmation['second_seed_only']=paired_check([17])
 (ROOT/'confirmation-summary.json').write_text(json.dumps(confirmation,indent=2))
 print('CONFIRMATION',json.dumps(confirmation),flush=True)
 if not confirmation['passed'] or not confirmation['second_seed_only']['passed']:
  (ROOT/'attention-needed.json').write_text(json.dumps(dict(reason='Confirmation unstable; preserve results and revise before untouched verification.'),indent=2));return
 print('CANDIDATE_LOCKED_BEFORE_UNTOUCHED_SEQUENCE',flush=True)
 subprocess.run([sys.executable,'prepare_verification.py'],cwd=ROOT,check=True)
 subprocess.run([sys.executable,'run_iteration.py','--phase','verification'],cwd=ROOT,check=True)
 phase_record('verification')
 subprocess.run([sys.executable,str(AUX/'finalize.py')],cwd=ROOT,check=True)
 print('COORDINATED_EXPERIMENT_COMPLETE',flush=True)

if __name__=='__main__':
 try:main()
 except Exception:
  traceback.print_exc();(ROOT/'coordinator-failure.txt').write_text(traceback.format_exc());sys.exit(1)
