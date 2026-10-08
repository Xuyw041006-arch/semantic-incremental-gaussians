"""Real RGB-D budget experiment with held-out, full-image CUDA rendering."""
import argparse
import gc
import gzip
import json
from pathlib import Path
import time
import numpy as np
from PIL import Image
from .data import ManifestSequence
from .mapping import GaussianMap,MapConfig
from .refine import CudaRefiner


def render(mapping,frame):
    import torch
    from gsplat import rasterization
    d=mapping.data[:mapping.count]; h,w=frame.depth.shape
    def tensor(a): return torch.tensor(np.asarray(a),device='cuda',dtype=torch.float32)
    quats=torch.zeros((mapping.count,4),device='cuda'); quats[:,0]=1
    kwargs=dict(means=tensor(d['mean']),quats=quats,scales=tensor(d['sigma'])[:,None].expand(-1,3).contiguous(),
        opacities=tensor(d['opacity']),viewmats=tensor(np.linalg.inv(frame.c2w))[None],Ks=tensor(frame.K)[None],
        width=w,height=h,packed=True)
    with torch.inference_mode():
        rgb,alpha,_=rasterization(colors=tensor(d['color']),render_mode='RGB+ED',**kwargs)
        probabilities=d['evidence']; label=np.where(probabilities.sum(1)>0,probabilities.argmax(1),0)
        palette=np.array([l['color'] for l in mapping.labels])/255
        sem,_,_=rasterization(colors=tensor(palette[label]),**kwargs)
    return rgb[0].cpu().numpy(),alpha[0,...,0].cpu().numpy(),sem[0].cpu().numpy()


def quality(mapping,frame):
    from skimage.metrics import structural_similarity
    pred,alpha,sem=render(mapping,frame)
    rgb=np.clip(pred[...,:3],0,1); target=frame.rgb/255
    valid=np.isfinite(frame.depth)&(frame.depth>.15)&(frame.depth<8)
    for label in mapping.config.dynamic_labels: valid &= frame.semantic!=label
    covered=valid&(alpha>.1)
    if not valid.any(): return {'full_psnr':None,'coverage':0},rgb,sem
    mse=np.mean((rgb[valid]-target[valid])**2)
    # RGB holes remain black and count in full_psnr; coverage is reported separately.
    score={'full_psnr':float(-10*np.log10(max(float(mse),1e-10))),
        'coverage':float(covered.sum()/valid.sum()),
        'covered_depth_mae_m':float(np.abs(pred[...,3][covered]-frame.depth[covered]).mean()) if covered.any() else None}
    _,ssim_map=structural_similarity(target,rgb,channel_axis=2,data_range=1,full=True)
    score['valid_ssim']=float(ssim_map.mean(-1)[valid].mean())
    return score,rgb,np.clip(sem,0,1)


def run_experiment(manifest,output,cap=24000,freeze=True,refine=True,record_replay=False):
    import torch
    sequence=ManifestSequence(manifest); output=Path(output); output.mkdir(parents=True,exist_ok=True)
    holdout=[i for i in range(len(sequence)) if i%6==3]
    train=[i for i in range(len(sequence)) if i not in holdout]
    cfg=MapConfig(max_gaussians=cap,voxel_size=.04,freeze_stable=freeze,adaptive_budget=False,
        dynamic_labels=tuple(i for i,l in enumerate(sequence.labels) if l['name']=='person'))
    mapping=GaussianMap(sequence.labels,cfg); refiner=CudaRefiner(steps=5) if refine else None
    old_views=holdout[:3]; scores=[]; snapshots=[]
    if record_replay: (output/'replay').mkdir(exist_ok=True)
    started=time.perf_counter(); torch.cuda.empty_cache()
    for position,index in enumerate(train):
        begin=time.perf_counter(); frame=sequence.get(index); source_ms=(time.perf_counter()-begin)*1000
        stats,_,_=mapping.integrate(frame); stats['source_frame']=index; stats['source_ms']=source_ms
        if refiner:
            refiner.observe(frame,position); gpu=refiner.refine(mapping)
            if gpu: stats.update(gpu)
        stats['mapping_total_ms']=stats['update_ms']+stats.get('refine_ms',0)
        if (position+1)%20==0 or position==len(train)-1:
            for old_index in old_views:
                if old_index>index: continue
                score,_,_=quality(mapping,sequence.get(old_index))
                snapshots.append({'after_train_frame':position+1,'heldout_frame':old_index,**score})
        if record_replay and (position%2==0 or position==len(train)-1):
            # Save actual optimized maps, so desktop playback needs no CUDA or re-fusion.
            replay={'source_frame':index,'next_frame':position+1,'stats':stats,
                'map':mapping.packet(),'trajectory':mapping.trajectory,'camera':frame.c2w.tolist(),
                'objects':mapping.tracker.summary()}
            with gzip.open(output/'replay'/f'{position:04}.json.gz','wt') as stream:
                json.dump(replay,stream,separators=(',',':'))
        if position%10==0 or position==len(train)-1:
            print(f"frame {position+1}/{len(train)} | map {mapping.count}/{cap} | mapping {stats['mapping_total_ms']:.1f} ms",flush=True)
    mapping_wall=time.perf_counter()-started
    mapping.export(output)
    metadata=json.loads((output/'metadata.json').read_text())
    metadata['backend']='cuda_gsplat_refine' if refine else 'cpu_isotropic_fusion'
    metadata['replay_note']='Actual optimized map snapshots every two training frames; replay encoding excluded from mapping latency.'
    (output/'metadata.json').write_text(json.dumps(metadata,indent=2))
    for index in holdout:
        score,rgb,sem=quality(mapping,sequence.get(index)); scores.append({'heldout_frame':index,**score})
        if index in (holdout[0],holdout[len(holdout)//2],holdout[-1]):
            Image.fromarray((rgb*255).astype(np.uint8)).save(output/f'heldout-{index:03}-rgb.png')
            Image.fromarray((sem*255).astype(np.uint8)).save(output/f'heldout-{index:03}-semantic.png')
    times=np.array([s['mapping_total_ms'] for s in mapping.history]); gpu_stats=[s for s in mapping.history if 'gpu_peak_mib' in s]
    summary={'dataset':sequence.name,'train_frames':len(train),'heldout_frames':len(holdout),
        'max_gaussians':cap,'gaussians':mapping.count,'freeze_stable':freeze,'cuda_refine':refine,
        'mapping_p50_ms':float(np.median(times)),'mapping_p95_ms':float(np.percentile(times,95)),
        'mapping_total_seconds_including_evaluation':mapping_wall,
        'refine_peak_mib':max((s['gpu_peak_mib'] for s in gpu_stats),default=0),
        'map_capacity_mib':mapping.data.nbytes/1024**2,
        'process_peak_rss_mib':max(s['process_peak_rss_mib'] for s in mapping.history),
        'heldout_mean_full_psnr':float(np.mean([s['full_psnr'] for s in scores])),
        'heldout_mean_valid_ssim':float(np.mean([s['valid_ssim'] for s in scores])),
        'heldout_mean_coverage':float(np.mean([s['coverage'] for s in scores])),
        'rejected_observations':sum(s['rejected'] for s in mapping.history),
        'object_tracks':len(mapping.tracker.tracks),'gpu':torch.cuda.get_device_name(),
        'note':'Groundtruth poses; predicted panoptic masks cached separately; heldout views excluded from fusion/refinement; one exploratory run.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2))
    (output/'heldout-quality.json').write_text(json.dumps(scores,indent=2))
    (output/'old-view-retention.json').write_text(json.dumps(snapshots,indent=2))
    print(json.dumps(summary,indent=2),flush=True)
    del mapping,refiner; gc.collect(); torch.cuda.empty_cache()
    return summary


def main():
    p=argparse.ArgumentParser(); p.add_argument('--manifest',required=True); p.add_argument('--output',required=True)
    p.add_argument('--cap',type=int,default=24000); p.add_argument('--no-freeze',action='store_true'); p.add_argument('--no-refine',action='store_true')
    p.add_argument('--record-replay',action='store_true',help='Save actual map snapshots; run separately from latency measurement')
    args=p.parse_args(); run_experiment(args.manifest,args.output,args.cap,not args.no_freeze,not args.no_refine,args.record_replay)


if __name__=='__main__': main()
