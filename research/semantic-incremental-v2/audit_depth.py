"""Diagnostic held-out metric in meters; never used for training or selection."""
from pathlib import Path
import sys,json,csv
sys.path.insert(0,str(Path.cwd()))
import torch,numpy as np
from hglab.readers import load_scene,depth_image
from hglab.base_train import render
ROOT=Path.cwd();INPUT=Path('/content/semantic-incremental')
@torch.no_grad()
def main():
 torch.set_num_threads(2);rows=[];summary=[]
 for ds,folder in [('tum-desk','tum640'),('tum-desk2','desk2-640')]:
  scene=load_scene(INPUT/'data'/folder/'manifest.json',640)
  for seed in [7,17]:
   for factors in ['000','111']:
    p=ROOT/'results'/ds/factors/f'seed-{seed}'
    ck=torch.load(p/'checkpoint.pt',map_location='cpu',weights_only=False)
    params={k:v.cuda() for k,v in ck['params'].items()}
    group=[]
    for im in scene['images']:
     if not im['test']:continue
     image,alpha,_=render(params,im,2,mode='RGB+ED')
     pred=image[0,...,3].cpu().numpy();a=alpha[0,...,0].cpu().numpy();gt=depth_image(im)
     valid=np.isfinite(gt)&(gt>.2)&(gt<5.);covered=valid&(a>.1)&np.isfinite(pred)
     error=(pred[covered]-gt[covered]).astype(np.float64)
     # Report covered-surface error and coverage, plus a declared missed-pixel penalty.
     penalized=np.where(covered,pred,5.).clip(.01,5.)-gt
     row=dict(dataset=ds,seed=seed,factors=factors,index=im['index'],valid_pixels=int(valid.sum()),covered_pixels=int(covered.sum()),
      squared_error=float((error**2).sum()),absolute_error=float(np.abs(error).sum()),
      penalized_squared_error=float(np.square(penalized[valid].astype(np.float64)).sum()))
     rows.append(row);group.append(row)
    n=sum(r['valid_pixels'] for r in group);c=sum(r['covered_pixels'] for r in group)
    summary.append(dict(dataset=ds,seed=seed,factors=factors,covered_depth_rmse_m=float(np.sqrt(sum(r['squared_error'] for r in group)/max(1,c))),
     covered_depth_mae_m=sum(r['absolute_error'] for r in group)/max(1,c),depth_coverage=c/max(1,n),
     penalized_depth_rmse_m=float(np.sqrt(sum(r['penalized_squared_error'] for r in group)/max(1,n))),
     qualification='GT depth .2–5m; coverage alpha>.1; uncovered predictions assigned 5m only in explicitly penalized metric; final stage; independent diagnostic, not selection objective'))
    del params,ck;torch.cuda.empty_cache()
 (ROOT/'analysis/depth-summary.json').write_text(json.dumps(summary,indent=2))
 with (ROOT/'analysis/depth-views.csv').open('w') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 print('HELDOUT_DEPTH_DIAGNOSTIC_COMPLETE',json.dumps(summary),flush=True)
if __name__=='__main__':main()
