"""Evaluate the frozen candidate screen using explicit practical thresholds."""
from pathlib import Path
import json,numpy as np
ROOT=Path.cwd()
s=json.loads((ROOT/'status.json').read_text());assert s['phase']=='screen' and s['state']=='complete'
rows=[]
for ds in ['tum-desk','kitchen']:
 result={f:json.loads((ROOT/'results'/ds/f/'seed-7/metrics.json').read_text())[-1] for f in ['000','111']}
 delta={k:result['111'][k]-result['000'][k] for k in ['psnr','ssim','old_psnr','fine_sam_miou','small_fine_sam_miou','update_p95_ms']}
 rows.append(dict(dataset=ds,baseline=result['000'],candidate=result['111'],delta=delta))
passed=bool(np.mean([r['delta']['psnr'] for r in rows])>0 and min(r['delta']['psnr'] for r in rows)>=-.1 and np.mean([r['delta']['old_psnr'] for r in rows])>0 and min(r['delta']['fine_sam_miou'] for r in rows)>=-.015)
report=dict(passed=passed,selection_only=True,seed=7,rule=json.loads((ROOT/'iteration-plan.json').read_text())['selection_rule'],results=rows)
(ROOT/'screen-summary.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
