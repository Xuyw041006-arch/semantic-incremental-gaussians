"""SAM 2.1 nested masks + CLIP crop descriptors on training keyframes only."""
import argparse
from pathlib import Path
import json
import time
import urllib.request
import numpy as np
from PIL import Image
import torch
import open_clip
from sam2.build_sam import build_sam2
from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
from .colmap import read_scene,image_rgb
from .hierarchy import containment_tree,scale_layers

VOCAB=['wall','floor','ceiling','kitchen cabinet','cabinet door','cabinet handle',
       'kitchen countertop','kitchen island','dining table','table top','table leg',
       'chair','chair seat','chair back','chair leg','refrigerator','refrigerator door',
       'oven','stove','microwave','sink','faucet','dishwasher','range hood','window',
       'window frame','door','shelf','bowl','plate','cup','bottle','kettle','coffee machine',
       'plant','plant pot','lamp','painting','curtain','kitchen appliance','background',
       'toy bulldozer','yellow toy vehicle','LEGO model','toy wheel','rubber track',
       'bulldozer blade','toy vehicle cabin','table mat','wooden table','table edge',
       'cutting board','cloth','towel','sofa','cushion','bookshelf','book','basket',
       'light switch','electrical socket','cabinet hinge']


def overlay(rgb,ids):
    palette=np.random.default_rng(8).integers(45,235,(max(1,int(ids.max())+1),3),dtype=np.uint8)
    out=rgb.copy(); known=ids>0; out[known]=(.35*rgb[known]+.65*palette[ids[known]]).astype('uint8')
    return out


def run(args):
    torch.set_num_threads(2); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    ckpt=output/'sam2.1_hiera_tiny.pt'
    if not ckpt.exists():
        urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',ckpt)
    sam=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(ckpt),device='cuda',apply_postprocessing=False)
    generator=SAM2AutomaticMaskGenerator(sam,points_per_side=16,points_per_batch=64,
        pred_iou_thresh=.75,stability_score_thresh=.9,crop_n_layers=1,
        crop_n_points_downscale_factor=2,box_nms_thresh=.7)
    clip,_,preprocess=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',device='cuda',force_quick_gelu=True)
    clip.eval(); tokenize=open_clip.get_tokenizer('ViT-B-32')
    with torch.inference_mode():
        text=clip.encode_text(tokenize(['a photo of a '+s for s in VOCAB]).to('cuda')).float()
        text=F_normalize(text)
    scene=read_scene(args.data,args.width)
    images=[im for im in scene['images'] if not im['test']][::args.every]
    records=[]
    for im in images:
        path=output/f'masks-{im["index"]:04d}.npz'
        meta_path=path.with_suffix('.json')
        if path.exists() and meta_path.exists(): records.append(json.loads(meta_path.read_text())); continue
        rgb=image_rgb(im); torch.cuda.synchronize(); begin=time.perf_counter()
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16): anns=generator.generate(rgb)
        anns=[a for a in anns if a['area']>=max(100,rgb.shape[0]*rgb.shape[1]*.0003)]
        anns=sorted(anns,key=lambda a:a['predicted_iou']*a['stability_score'],reverse=True)[:100]
        if not anns: continue
        masks=np.array([a['segmentation'] for a in anns]); parents=containment_tree(masks)
        crops=[]
        for a in anns:
            x,y,w,h=map(int,a['bbox']); pad=max(4,int(min(w,h)*.05))
            masked=rgb.copy(); masked[~a['segmentation']]=127
            crop=Image.fromarray(masked[max(0,y-pad):y+h+pad,max(0,x-pad):x+w+pad])
            crops.append(preprocess(crop))
        features=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for j in range(0,len(crops),16):
                features.append(F_normalize(clip.encode_image(torch.stack(crops[j:j+16]).to('cuda')).float()).cpu())
        features=torch.cat(features).numpy(); scores=features@text.cpu().numpy().T
        pred=scores.argmax(1); scales=2*np.sqrt(masks.mean((1,2)))
        layers=scale_layers(masks,scales,targets=(.8,.35,.12))
        torch.cuda.synchronize(); elapsed=time.perf_counter()-begin
        np.savez_compressed(path,masks=np.packbits(masks,axis=2),shape=np.array(masks.shape),
                            clip=features.astype('float16'),parents=parents,preview_layers=layers)
        row=dict(index=im['index'],name=im['name'],masks=len(anns),seconds=elapsed,clip_quick_gelu=True,
                 predictions=[dict(id=j+1,label=VOCAB[pred[j]],clip_cosine=float(scores[j,pred[j]]),
                                   iou=float(a['predicted_iou']),stability=float(a['stability_score']),
                                   parent=int(parents[j])+1) for j,a in enumerate(anns)])
        meta_path.write_text(json.dumps(row,indent=2)); records.append(row)
        if len(records)%5==1:
            panel=np.concatenate([rgb,*[overlay(rgb,l) for l in layers]],axis=1)
            Image.fromarray(panel).save(output/f'preview-{im["index"]:04d}.jpg')
        (output/'index.json').write_text(json.dumps(records,indent=2))
        print('MASKS',im['index'],len(anns),round(elapsed,2),flush=True)
    np.save(output/'text-features.npy',text.cpu().numpy()); (output/'vocabulary.json').write_text(json.dumps(VOCAB))
    print('MASK_EXTRACTION_COMPLETE',len(records),flush=True)


def F_normalize(x): return x/x.norm(dim=-1,keepdim=True).clamp_min(1e-8)


def upgrade_clip_cache(maskdir,scene):
    """Correct legacy OpenCLIP activation config without repeating SAM inference."""
    records=json.loads((maskdir/'index.json').read_text())
    if all(r.get('clip_quick_gelu') for r in records): return
    clip,_,prep=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',device='cuda',force_quick_gelu=True)
    clip.eval(); tok=open_clip.get_tokenizer('ViT-B-32'); begin=time.perf_counter()
    with torch.inference_mode(): text=F_normalize(clip.encode_text(tok(['a photo of a '+s for s in VOCAB]).cuda()).float()).cpu().numpy()
    for row in records:
        path=maskdir/f'masks-{row["index"]:04d}.npz'; data=dict(np.load(path))
        masks=np.unpackbits(data['masks'],axis=2)[:,:,:int(data['shape'][2])].astype(bool)
        rgb=image_rgb(scene['images'][row['index']]); crops=[]
        for mask in masks:
            y,x=np.where(mask); x0,x1=x.min(),x.max()+1; y0,y1=y.min(),y.max()+1
            pad=max(4,int(min(x1-x0,y1-y0)*.05)); pic=rgb.copy(); pic[~mask]=127
            crops.append(prep(Image.fromarray(pic[max(0,y0-pad):y1+pad,max(0,x0-pad):x1+pad])))
        features=[]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
            for j in range(0,len(crops),16): features.append(F_normalize(clip.encode_image(torch.stack(crops[j:j+16]).cuda()).float()).cpu().numpy())
        features=np.concatenate(features); data['clip']=features.astype('float16')
        np.savez_compressed(path,**data); scores=features@text.T
        for j,pred in enumerate(row['predictions']):
            k=scores[j].argmax(); pred['label']=VOCAB[k]; pred['clip_cosine']=float(scores[j,k])
        row['clip_quick_gelu']=True
        path.with_suffix('.json').write_text(json.dumps(row,indent=2))
        print('CLIP_CACHE_UPGRADED',row['index'],flush=True)
    np.save(maskdir/'text-features.npy',text)
    (maskdir/'index.json').write_text(json.dumps(records,indent=2))
    (maskdir/'clip-upgrade.json').write_text(json.dumps(dict(seconds=time.perf_counter()-begin,quick_gelu=True)))
    del clip
    torch.cuda.empty_cache()


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--width',type=int,default=800); p.add_argument('--every',type=int,default=6)
    run(p.parse_args())
