"""Held-out RGB and SAM region agreement, with unknowns counted as misses."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from scipy.optimize import linear_sum_assignment
from .base_train import render
from .colmap import image_rgb
from .readers import depth_image,sparse_depth
from .online_semantics import masks_for
from .hierarchy import mask_physical_scale,scale_layers


def camera_small(im,width=384):
    o=im.copy();w=min(width,im['width']);h=round(im['height']*w/im['width'])
    o['K']=im['K'].copy();o['K'][0]*=w/im['width'];o['K'][1]*=h/im['height']
    o['width']=w;o['height']=h;return o


def matched_iou(pred,target):
    """Hungarian label matching; every GT candidate counts, even if unobserved."""
    gt=np.unique(target);gt=gt[gt>0];pr=np.unique(pred);pr=pr[pr>0]
    if not len(gt):return dict(miou=None,small_miou=None,coverage=0.,regions=0)
    area=np.array([(target==g).sum() for g in gt]);scores=np.zeros((len(gt),len(pr)))
    for a,g in enumerate(gt):
        mask=target==g
        vals,counts=np.unique(pred[mask],return_counts=True);intersection=dict(zip(vals,counts))
        for b,p in enumerate(pr):
            inter=intersection.get(p,0);union=area[a]+np.count_nonzero(pred==p)-inter
            scores[a,b]=inter/max(1,union)
    results=np.zeros(len(gt))
    if len(pr):
        a,b=linear_sum_assignment(-scores);results[a]=scores[a,b]
    small=area<=np.percentile(area,25)
    return dict(miou=float(results.mean()),small_miou=float(results[small].mean()),
        coverage=float(np.mean(pred[target>0]>0)),regions=len(gt),per_region_iou=results.tolist())


@torch.no_grad()
def semantic_render(params,im,ids):
    n=len(ids);device=params['means'].device;ids=torch.as_tensor(ids,device=device,dtype=torch.int64)
    values=torch.unique(ids);values=values[values>0]
    best=torch.zeros((im['height'],im['width']),device=device)
    prediction=torch.zeros_like(best,dtype=torch.int64)
    for start in range(0,len(values),16):
        chunk=values[start:start+16];colors=(ids[:,None]==chunk[None,:]).float()
        image,alpha,_=render(params,im,colors=colors)
        confidence,which=image[0].max(-1);update=(confidence>best)&(alpha[0,...,0]>.1)
        prediction[update]=chunk[which[update]];best[update]=confidence[update]
    return prediction.cpu().numpy()


@torch.no_grad()
def evaluate(params,scene,end,masks_folder,targets,ids,out,stage,degree):
    from skimage.metrics import structural_similarity
    from .readers import load_scene
    masks_folder=Path(masks_folder);out=Path(out)
    teachers={r['index']:r for r in json.loads((masks_folder/'index.json').read_text()) if r['test']}
    rgb_rows=[];sem_rows=[];old_end=round(len(scene['images'])/6)
    for im in scene['images'][:end]:
        if not im['test']:continue
        pred,alpha,_=render(params,im,degree)
        pred=pred[0].clamp(0,1).cpu().numpy();target=image_rgb(im).astype(np.float32)/255
        mse=float(np.mean((pred-target)**2))
        row=dict(index=im['index'],psnr=float(-10*np.log10(max(mse,1e-10))),
            ssim=float(structural_similarity(target,pred,data_range=1.,channel_axis=2)))
        rgb_rows.append(row)
        if im['index'] not in teachers:continue
        small=camera_small(im);masks,_=masks_for(teachers[im['index']],masks_folder,small)
        # Evaluation-only reference depth. Never passed to training or policies.
        depth=depth_image(small) if 'depth_path' in small else sparse_depth(scene['points'],small)
        scales=np.array([mask_physical_scale(m,depth,small['K']) for m in masks])
        layers=scale_layers(masks,scales,targets);layers[1,layers[0]==layers[1]]=0
        rendered=[]
        for level in range(2):
            p=semantic_render(params,small,ids[level]);rendered.append(p)
            result=matched_iou(p,layers[level]);sem_rows.append(dict(index=im['index'],level=level,**result))
        if stage in (1,6) and out.name=='seed-7' and out.parent.name in ('000','111'):
            Image.fromarray((np.concatenate([target,pred],axis=1)*255).astype(np.uint8)).save(out/f'view-{stage}-{im["index"]:04d}.jpg')
            palette=np.random.default_rng(31).integers(40,240,(max(int(ids.max())+1,1),3),dtype=np.uint8);palette[0]=40
            rgb=np.asarray(Image.fromarray((target*255).astype(np.uint8)).resize((small['width'],small['height'])))
            Image.fromarray(np.concatenate([rgb,*[palette[p] for p in rendered]],axis=1)).save(out/f'semantic-{stage}-{im["index"]:04d}.jpg')
    (out/f'rgb-quality-{stage}.json').write_text(json.dumps(rgb_rows,indent=2))
    (out/f'semantic-quality-{stage}.json').write_text(json.dumps(sem_rows,indent=2))
    def average(rows,key):
        values=[r[key] for r in rows if r.get(key) is not None]
        return float(np.mean(values)) if values else None
    old_rgb=[r for r in rgb_rows if r['index']<old_end];old_sem=[r for r in sem_rows if r['index']<old_end]
    return dict(psnr=average(rgb_rows,'psnr'),ssim=average(rgb_rows,'ssim'),old_psnr=average(old_rgb,'psnr'),
        object_sam_miou=average([r for r in sem_rows if r['level']==0],'miou'),
        fine_sam_miou=average([r for r in sem_rows if r['level']==1],'miou'),
        small_fine_sam_miou=average([r for r in sem_rows if r['level']==1],'small_miou'),
        fine_coverage=average([r for r in sem_rows if r['level']==1],'coverage'),
        old_fine_sam_miou=average([r for r in old_sem if r['level']==1],'miou'),
        gaussian_object_coverage=float(np.mean(ids[0]>0)),gaussian_fine_coverage=float(np.mean(ids[1]>0)),
        evaluation_gpu_peak_mib=torch.cuda.max_memory_allocated()/1024**2)
