"""Reproducible CPU budget ablations; independent processes for peak RSS."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


def main():
    p=argparse.ArgumentParser(); p.add_argument("--output",default="results/ablations"); p.add_argument("--frames",type=int,default=100)
    args=p.parse_args(); root=Path(args.output); root.mkdir(parents=True,exist_ok=True)
    variants=[("cap-8000",8000,[]),("cap-24000",24000,[]),("cap-50000",50000,[]),("no-freeze-24000",24000,["--no-freeze"])]
    rows=[]
    for name,cap,extra in variants:
        folder=root/name
        with (root/(name+".log")).open("w") as log:
            subprocess.run([sys.executable,"-m","sglab","run","--frames",str(args.frames),"--max-gaussians",str(cap),
                "--no-adaptive-budget","--output",str(folder),*extra],stdout=log,stderr=subprocess.STDOUT,check=True)
        summary=json.loads((folder/"summary.json").read_text()); metrics=json.loads((folder/"metrics.json").read_text())
        row={"variant":name,"gaussians":summary["gaussians"],"capacity_mib":round(summary["map_capacity_mib"],3),
            "update_p50_ms":round(summary["update_median_ms"],3),"update_p95_ms":round(summary["update_p95_ms"],3),
            "peak_rss_mib":max(m["process_peak_rss_mib"] for m in metrics),"rejected_observations":sum(m["rejected"] for m in metrics)}
        rows.append(row); print(row,flush=True)
    with (root/"comparison.csv").open("w") as f:
        writer=csv.DictWriter(f,fieldnames=rows[0]); writer.writeheader(); writer.writerows(rows)
    (root/"comparison.json").write_text(json.dumps({"note":"Synthetic oracle input, fixed sample cap, single run per variant. CPU fusion only; not a quality or learned segmentation benchmark.","rows":rows},indent=2))


if __name__=="__main__": main()
