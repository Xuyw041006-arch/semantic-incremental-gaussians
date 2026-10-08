"""Whole-GPU sampling across all iteration phases; append-only telemetry."""
from pathlib import Path
import csv,json,time,subprocess
ROOT=Path(__file__).resolve().parent
p=ROOT/'gpu-telemetry.csv'
with p.open('a') as f:
 w=csv.writer(f)
 if p.stat().st_size==0:w.writerow(['unix_seconds','phase','job','dataset','factors','seed','gpu_used_mib','gpu_util_percent'])
 while not (ROOT/'monitor-stop').exists():
  try:
   s=json.loads((ROOT/'status.json').read_text())
   if s.get('state')=='running':
    j=s['current'];value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
    w.writerow([time.time(),s['phase'],len(s['completed'])+1,j['dataset'],j['factors'],j['seed'],*map(float,value)]);f.flush()
  except FileNotFoundError:pass
  except Exception as e:print(type(e).__name__,str(e),flush=True)
  time.sleep(1.)
