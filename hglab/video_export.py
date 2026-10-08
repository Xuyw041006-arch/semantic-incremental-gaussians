"""True Gaussian renders plus explicitly labelled point-center display LOD."""
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from .base_train import render
from .colmap import image_rgb
from .evaluation import camera_small, semantic_render


def palette(ids):
    ids=np.asarray(ids,dtype=np.int64)
    c=np.column_stack([45+(ids*97)%190,45+(ids*57)%190,45+(ids*137)%190]).astype(np.uint8)
    c[ids==0]=35
    return c


@torch.no_grad()
def stage_preview(params,scene,end,ids,out,stage,degree):
    out=Path(out);n=len(params['means'])
    take=np.random.default_rng(71).choice(n,min(n,20000),replace=False)
    xyz=params['means'][take].detach().cpu().numpy().astype(np.float32)
    rgb=(params['sh0'][take,0].detach().cpu().numpy()*.28209479177387814+.5).clip(0,1)
    np.savez_compressed(out/f'preview-{stage}.npz',points=xyz,colors=(rgb*255).astype(np.uint8),
                        ids=ids[:,take],total=n)
    from videogs.splats import export_splats
    values={k:v.detach().cpu().numpy() for k,v in params.items()}
    export_splats(out/f'stage-{stage}.splat',dict(means=values['means'],scales=np.exp(values['scales']),
        quats=values['quats'],opacity=1/(1+np.exp(-values['opacities'])),sh=np.concatenate([values['sh0'],values['shN']],1)),ids)
    for label,im in [('current',scene['images'][end-1]),('old',scene['images'][0])]:
        small=camera_small(im,640);pred,_,_=render(params,small,degree)
        rgb=(pred[0].clamp(0,1).cpu().numpy()*255).astype(np.uint8)
        Image.fromarray(rgb).save(out/f'stage-{stage}-{label}-rgb.jpg',quality=92)
        if label=='current':
            masks=[palette(semantic_render(params,small,ids[level]).ravel()).reshape(small['height'],small['width'],3) for level in range(2)]
            target=np.asarray(Image.fromarray(image_rgb(im)).resize((small['width'],small['height'])))
            panel=np.concatenate([target,rgb,*masks],axis=1)
            Image.fromarray(panel).save(out/f'stage-{stage}-panel.jpg',quality=92)
