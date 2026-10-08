"""Post-hoc diagnostic: remove region information while matching R's replay rate."""
from pathlib import Path
import json,sys,subprocess,shutil,time,hashlib,numpy as np
ROOT=Path.cwd();INPUT=Path('/content/semantic-incremental')

def save(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False))
def main():
 outroot=ROOT/'replay-control';outroot.mkdir(exist_ok=True)
 plan=json.loads((ROOT/'iteration-plan.json').read_text())
 save(outroot/'protocol.json',dict(recorded_before_control_results=True,post_hoc_diagnostic=True,
  purpose='Separate semantic replay selection from higher replay frequency; candidate is not retuned.',
  candidate_factors='001',control='Same temporal exploration, camera-pose novelty, 48-frame capacity and stage-dependent replay probability; region coverage is empty.',
  datasets=['tum-desk','kitchen'],seeds=[7,17],steps=600,stages=6,cap=300000))
 source=ROOT/'work/replay-control-source';source.mkdir(parents=True,exist_ok=True)
 for folder in ['hglab','sglab']:shutil.copytree(ROOT/folder,source/folder,dirs_exist_ok=True)
 policy=source/'hglab/policies.py';text=policy.read_text()
 marker='    frequency={}\n'
 assert text.count(marker)==1
 text=text.replace(marker,'    coverage={}  # Diagnostic: preserve time/pose selection but remove region information.\n'+marker)
 policy.write_text(text);(outroot/'policies-control.txt').write_text(text)
 save(outroot/'source-sha256.json',{str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*.py')})
 sys.path.insert(0,str(ROOT));from analyze_results import audit_run,csv_write
 rows=[];paired=[];completed=[]
 for ds,data in [('tum-desk',str(INPUT/'data/tum640/manifest.json')),('kitchen','/content/mip360/kitchen')]:
  for seed in [7,17]:
   folder=outroot/'results'/ds/'001'/f'seed-{seed}';folder.mkdir(parents=True,exist_ok=True)
   status=dict(phase='replay_control',state='running',completed=completed,total=4,current=dict(dataset=ds,seed=seed,factors='time_pose_control'))
   save(outroot/'status.json',status);save(ROOT/'status.json',status)
   if not (folder/'run.json').exists():
    command=[sys.executable,'-m','hglab.incremental','--data',data,'--prefixes',str(INPUT/'data'/ds/'prefixes'),'--masks',str(INPUT/'data'/ds/'teachers'),
     '--output',str(folder),'--factors','001','--seed',str(seed),'--cap',str(plan['cap']),'--width',str(plan['width']),
     '--steps',str(plan['steps']),'--semantic-strength',str(plan['semantic_strength']),'--replay-max',str(plan['replay_max'])]
    tick=time.perf_counter()
    with (folder/'train.log').open('w') as log:subprocess.run(command,cwd=source,stdout=log,stderr=subprocess.STDOUT,check=True)
    save(folder/'process-wall.json',dict(seconds=time.perf_counter()-tick))
   row,history,check=audit_run(folder,plan,ds,seed,'001',True);assert check is not None
   reference=json.loads((ROOT/'results'/ds/'001'/f'seed-{seed}'/'metrics.json').read_text())
   assert [r['optimized_pixels'] for r in reference]==[r['optimized_pixels'] for r in history]
   assert [r['replay_probability'] for r in reference]==[r['replay_probability'] for r in history]
   rows.append(row)
   paired.append(dict(dataset=ds,seed=seed,delta={k:reference[-1][k]-history[-1][k] for k in ['psnr','old_psnr','fine_sam_miou']}))
   completed.append(dict(dataset=ds,seed=seed));print('MATCHED_REPLAY_CONTROL_COMPLETE',ds,seed,paired[-1],flush=True)
 save(outroot/'status.json',dict(state='complete',completed=completed,total=4))
 save(ROOT/'analysis/replay-control-runs.json',rows);csv_write(ROOT/'analysis/replay-control-runs.csv',rows)
 save(ROOT/'analysis/replay-control-summary.json',dict(post_hoc_diagnostic=True,paired_semantic_R_minus_time_pose_control=paired))
if __name__=='__main__':main()
