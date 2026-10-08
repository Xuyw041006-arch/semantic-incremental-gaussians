import argparse
from pathlib import Path
import json
import time
import numpy as np
from .data import ManifestSequence
from .synthetic import SyntheticSequence
from .mapping import GaussianMap, MapConfig


def main():
    p=argparse.ArgumentParser(description="Incremental semantic Gaussian mapping lab")
    p.add_argument("command",choices=["serve","run","segment","make-demo","import-tum"])
    p.add_argument("--manifest",help="RGB-D manifest; omit for synthetic oracle sequence")
    p.add_argument("--output",default="results/latest")
    p.add_argument("--dataset",help="unpacked TUM dataset directory")
    p.add_argument("--every",type=int,default=3,help="sample every N TUM RGB frames")
    p.add_argument("--frames",type=int,default=100)
    p.add_argument("--max-gaussians",type=int,default=24000)
    p.add_argument("--target-ms",type=float,default=45)
    p.add_argument("--port",type=int,default=8765)
    p.add_argument("--device",default="auto",choices=["auto","cuda","cpu"])
    p.add_argument("--refine",action="store_true",help="Run optional CUDA gsplat refinement (run only)")
    p.add_argument("--no-freeze",action="store_true",help="Ablation: continue updating stable splats")
    p.add_argument("--no-adaptive-budget",action="store_true",help="Deterministic fixed sample cap for evaluation")
    args=p.parse_args()
    if args.frames<1: p.error("--frames must be positive")
    if args.command=="import-tum":
        if not args.dataset or args.every<1: p.error("import-tum requires --dataset and --every >= 1")
        from .tum import import_tum
        print(import_tum(args.dataset,args.output,args.frames,args.every)); return
    if args.command=="segment":
        if not args.manifest: p.error("segment requires --manifest")
        from .segmentation import segment_manifest
        print(segment_manifest(args.manifest,args.output,args.device)); return
    sequence=ManifestSequence(args.manifest) if args.manifest else SyntheticSequence(count=args.frames)
    cfg=MapConfig(max_gaussians=args.max_gaussians,target_update_ms=args.target_ms,
                  freeze_stable=not args.no_freeze,adaptive_budget=not args.no_adaptive_budget,
                  dynamic_labels=tuple(i for i,l in enumerate(sequence.labels) if l["name"]=="person"))
    if args.command=="serve":
        if args.refine: p.error("live CPU viewer does not use --refine; use run on Colab")
        from .server import serve
        serve(sequence,cfg,args.output,args.port); return
    if args.command=="make-demo":
        from PIL import Image
        output=Path(args.output); output.mkdir(parents=True,exist_ok=True); frames=[]
        for i in range(len(sequence)):
            f=sequence.get(i); prefix=f"{i:06}"
            Image.fromarray(f.rgb).save(output/(prefix+".png"))
            spec={"rgb":prefix+".png","c2w":f.c2w.tolist(),"timestamp":f.timestamp,
                  "segmentation_source":f.segmentation_source}
            for name in ("depth","semantic","instance","confidence"):
                file=prefix+"_"+name+".npy"; np.save(output/file,getattr(f,name)); spec[name]=file
            frames.append(spec)
        (output/"manifest.json").write_text(json.dumps({"name":sequence.name,"description":sequence.description,
            "pose_convention":"opencv_c2w_meters","depth_scale":1,"K":sequence.get(0).K.tolist(),
            "labels":sequence.labels,"frames":frames},indent=2,ensure_ascii=False))
        print(output/"manifest.json"); return
    mapping=GaussianMap(sequence.labels,cfg); refiner=None
    if args.refine:
        from .refine import CudaRefiner
        refiner=CudaRefiner()
    from .evaluation import probe_old_view
    anchors={}; probes=[]
    for i in range(min(args.frames,len(sequence))):
        start=time.perf_counter(); frame=sequence.get(i); source_ms=(time.perf_counter()-start)*1000
        stats,_,_=mapping.integrate(frame); stats["source_ms"]=round(source_ms,3)
        if hasattr(sequence,"frames"): stats["cached_segmentation_ms"]=sequence.frames[i].get("segmentation_ms")
        if refiner:
            refiner.observe(frame,i); gpu=refiner.refine(mapping)
            if gpu: stats.update(gpu)
        stats["mapping_total_ms"]=round(stats["update_ms"]+stats.get("refine_ms",0),3)
        if i in (0,10,20): anchors[i]=frame
        if i%20==0 or i==min(args.frames,len(sequence))-1:
            for anchor_id,anchor in anchors.items(): probes.append({"after_frame":i,"anchor_frame":anchor_id,**probe_old_view(mapping,anchor)})
        print(f"{i+1:3d} | splats {mapping.count:6d} | update {stats['mapping_total_ms']:7.2f} ms | added {stats['added']:4d} | stable {stats['stable']:6d}",flush=True)
    mapping.export(args.output)
    times=[s["mapping_total_ms"] for s in mapping.history]
    summary={"backend":"cuda_gsplat_refine" if refiner else "cpu_isotropic_fusion","frames":mapping.frame_id,
        "input":sequence.description,"gaussians":mapping.count,"map_capacity_mib":mapping.data.nbytes/1024**2,
        "update_median_ms":float(np.median(times)),"update_p95_ms":float(np.percentile(times,95)),
        "update_target_ms":cfg.target_update_ms,"cpu_budget_overruns":sum(s["budget_overrun"] for s in mapping.history),
        "gpu_measured":bool(refiner),"old_view_probe_note":"Gaussian centre projection, not a rendered-view PSNR benchmark",
        "old_view_probes":probes}
    (Path(args.output)/"summary.json").write_text(json.dumps(summary,indent=2,ensure_ascii=False))
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
