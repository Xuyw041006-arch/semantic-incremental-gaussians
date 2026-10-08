"""External whole-GPU telemetry, separate from PyTorch tensor-allocation peaks."""
from pathlib import Path
import csv,json,time,subprocess,sys,psutil
ROOT=Path(__file__).resolve().parent
with (ROOT/'gpu-telemetry.csv').open('w') as f:
    writer=csv.writer(f);writer.writerow(['unix_seconds','job','dataset','factors','seed','gpu_used_mib','gpu_util_percent'])
    while True:
        status=ROOT/'status.json'
        if status.exists():
            try:s=json.loads(status.read_text())
            except json.JSONDecodeError:time.sleep(.2);continue
            if s.get('state')=='complete':break
            try:
                value=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
                writer.writerow([time.time(),s.get('job'),s.get('dataset'),s.get('factors'),s.get('seed'),*map(float,value)]);f.flush()
            except Exception as e:print(type(e).__name__,flush=True)
        if (ROOT/'failure.txt').exists():break
        time.sleep(1.)
