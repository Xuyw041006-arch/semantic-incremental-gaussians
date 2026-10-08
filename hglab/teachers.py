"""Shared deterministic 2D teacher cache; training consumes only arrived entries."""
import argparse,json,time,shutil
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from .readers import load_scene
from .colmap import image_rgb
from .hierarchy import containment_tree

VOCAB=['wall','floor','ceiling','table','table top','table leg','chair','chair back','chair seat','chair leg',
 'cabinet','cabinet door','cabinet handle','cup','bottle','book','plant','lamp','window','door',
 'LEGO model','toy bulldozer','rubber track','toy wheel','bulldozer blade','toy vehicle cabin',
 'monitor','monitor screen','monitor stand','keyboard','keyboard key','mouse','laptop','laptop screen',
 'desk','desk drawer','electrical socket','cable','cloth','background']

def run(args):
    import open_clip
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    torch.set_num_threads(2);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    scene=load_scene(args.data,args.width)
    images=[im for im in scene['images'] if not im['test']][::getattr(args,'train_stride',6)]
    images+=([im for im in scene['images'] if im['test']][::getattr(args,'test_stride',3)])
    images=sorted(images,key=lambda im:im['index']);records=[]
    checkpoint=getattr(args,'checkpoint',None)
    candidates=[Path(checkpoint)] if checkpoint else list(Path('/content/hierarchical-gaussian').rglob('sam2.1_hiera_tiny.pt'))
    if not candidates:raise RuntimeError('Existing official SAM2 checkpoint missing')
    sam=build_sam2('configs/sam2.1/sam2.1_hiera_t.yaml',str(candidates[0]),device='cuda',apply_postprocessing=False)
    gen=SAM2AutomaticMaskGenerator(sam,points_per_side=16,points_per_batch=64,pred_iou_thresh=.75,
        stability_score_thresh=.9,crop_n_layers=1,crop_n_points_downscale_factor=2,box_nms_thresh=.7)
    clip,_,prep=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',device='cuda',force_quick_gelu=True)
    clip.eval();tok=open_clip.get_tokenizer('ViT-B-32')
    with torch.inference_mode():
        text=clip.encode_text(tok(['a photo of a '+s for s in VOCAB]).cuda()).float()
        text=(text/text.norm(dim=-1,keepdim=True)).cpu().numpy()
    reuse=Path(args.reuse) if args.reuse else None
    for im in images:
        dest=out/f'masks-{im["index"]:04d}.npz';meta=dest.with_suffix('.json')
        if dest.exists() and meta.exists():records.append(json.loads(meta.read_text()));continue
        existing=list(reuse.rglob(dest.name)) if reuse and reuse.exists() else []
        existing=[p for p in existing if p.with_suffix('.json').exists() and
            json.loads(p.with_suffix('.json').read_text()).get('name')==im['name']]
        begin=time.perf_counter();reused=bool(existing)
        if reused:
            data=dict(np.load(existing[0]));row=json.loads(existing[0].with_suffix('.json').read_text())
            features=data['clip'].astype(np.float32)
        else:
            rgb=image_rgb(im)
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):anns=gen.generate(rgb)
            anns=sorted([a for a in anns if a['area']>=max(100,rgb.shape[0]*rgb.shape[1]*.0003)],
                key=lambda a:a['predicted_iou']*a['stability_score'],reverse=True)[:100]
            if not anns:raise RuntimeError('No teacher masks for frame '+str(im['index']))
            masks=np.asarray([a['segmentation'] for a in anns]);crops=[]
            for a in anns:
                x,y,w,h=map(int,a['bbox']);masked=rgb.copy();masked[~a['segmentation']]=127
                crops.append(prep(Image.fromarray(masked[y:y+h,x:x+w])))
            fs=[]
            with torch.inference_mode(),torch.autocast('cuda',dtype=torch.float16):
                for j in range(0,len(crops),16):
                    v=clip.encode_image(torch.stack(crops[j:j+16]).cuda()).float();fs.append((v/v.norm(dim=-1,keepdim=True)).cpu().numpy())
            features=np.concatenate(fs);data=dict(masks=np.packbits(masks,axis=2),shape=np.asarray(masks.shape),
                clip=features.astype(np.float16),parents=containment_tree(masks))
            row=dict(index=im['index'],name=im['name'],predictions=[dict(iou=float(a['predicted_iou']),
                stability=float(a['stability_score'])) for a in anns])
        scores=features@text.T
        for j,p in enumerate(row['predictions']):
            k=int(scores[j].argmax());p.update(label=VOCAB[k],clip_cosine=float(scores[j,k]))
        torch.cuda.synchronize();row.update(test=im['test'],seconds=time.perf_counter()-begin,reused=reused,
            teacher='SAM2.1 Tiny + CLIP ViT-B-32 OpenAI QuickGELU',pseudo_ground_truth=True)
        np.savez_compressed(dest,**data);meta.write_text(json.dumps(row,indent=2));records.append(row)
        (out/'index.json').write_text(json.dumps(records,indent=2))
        print('TEACHER',im['index'],'test',im['test'],'reuse',reused,'seconds',round(row['seconds'],2),flush=True)
    (out/'index.json').write_text(json.dumps(records,indent=2));np.save(out/'text-features.npy',text)
    (out/'vocabulary.json').write_text(json.dumps(VOCAB));print('TEACHERS_COMPLETE',len(records),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True)
    p.add_argument('--width',type=int,default=640);p.add_argument('--reuse')
    p.add_argument('--checkpoint');p.add_argument('--train-stride',type=int,default=6)
    p.add_argument('--test-stride',type=int,default=3);run(p.parse_args())
