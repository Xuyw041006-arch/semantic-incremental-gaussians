"""Evaluation-only important-object RGB metrics on held-out SAM masks."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from .base_train import render
from .readers import load_scene
from .colmap import image_rgb
from .online_semantics import masks_for
from .priority import DEFAULT_IMPORTANT,parse_queries,boundary_band


@torch.no_grad()
def evaluate_priority(data,masks,checkpoint,output,width=640,important=DEFAULT_IMPORTANT):
    scene=load_scene(data,width);masks=Path(masks);output=Path(output);output.mkdir(exist_ok=True,parents=True)
    ck=torch.load(checkpoint,map_location='cuda',weights_only=False)
    params=ck['params'];queries=parse_queries(important);rows=[]
    teachers={r['index']:r for r in json.loads((masks/'index.json').read_text()) if r['test']}
    from PIL import Image
    for im in scene['images'][:ck['end']]:
        if not im['test'] or im['index'] not in teachers:continue
        row=teachers[im['index']];regions,_=masks_for(row,masks,im);focus=np.zeros((im['height'],im['width']),bool)
        for region,pred in zip(regions,row['predictions']):
            if pred['iou']*pred['stability']>=.75 and any(pred['label'].lower()==q or pred['label'].lower().startswith(q+' ') for q in queries):focus|=region
        prediction,_,_=render(params,im,ck['args']['degree']);prediction=prediction[0].clamp(0,1).cpu().numpy()
        target=image_rgb(im).astype(np.float32)/255;error=((prediction-target)**2).mean(-1)
        edge=boundary_band(focus.astype(np.int32),2)
        r=dict(index=im['index'],important_pixels=int(focus.sum()),boundary_pixels=int(edge.sum()))
        for name,mask in [('important',focus),('boundary',edge),('other',~focus),('overall',np.ones_like(focus))]:
            r[name+'_psnr']=float(-10*np.log10(max(float(error[mask].mean()),1e-10))) if mask.any() else None
        rows.append(r)
        if im['index'] in (7,31,55):
            Image.fromarray((np.concatenate([target,prediction],axis=1)*255).astype(np.uint8)).save(output/f'heldout-{im["index"]:03d}.jpg',quality=94)
    result=dict(qualification='Held-out predicted SAM/CLIP object masks, not human labels; evaluation never feeds training',frames=rows)
    for name in ('important','boundary','other','overall'):
        values=[r[name+'_psnr'] for r in rows if r[name+'_psnr'] is not None]
        result[name+'_psnr']=float(np.mean(values)) if values else None
    (output/'priority-quality.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2),flush=True)
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--masks',required=True)
    p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True)
    p.add_argument('--width',type=int,default=640);p.add_argument('--important',default=DEFAULT_IMPORTANT)
    a=p.parse_args();evaluate_priority(**vars(a))
