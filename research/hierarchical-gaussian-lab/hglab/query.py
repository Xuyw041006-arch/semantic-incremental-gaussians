"""Arbitrary text and physical-scale point queries against the measured map."""
import argparse,json
from pathlib import Path
import numpy as np
from .viewer import scale_gate


def main():
    p=argparse.ArgumentParser(); p.add_argument('--field',required=True); p.add_argument('--text')
    p.add_argument('--gaussian',type=int); p.add_argument('--scale',type=float,default=.25)
    p.add_argument('--threshold',type=float,default=.75); p.add_argument('--output',default='query-mask.npy')
    a=p.parse_args(); root=Path(a.field)
    if a.text:
        import torch,open_clip
        clip,_,_=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',device='cuda',force_quick_gelu=True)
        token=open_clip.get_tokenizer('ViT-B-32')(['a photo of '+a.text]).cuda()
        with torch.inference_mode(): q=clip.encode_text(token).float(); q=(q/q.norm(dim=-1,keepdim=True)).cpu().numpy()[0]
        h=np.load(root/'hierarchy.npz'); proto=h['prototypes'].astype('float32'); weights=h['weights']; nodes=json.loads((root/'nodes.json').read_text())
        scales=json.loads((root/'metrics.json').read_text())['summary']['query_scales']
        level=int(np.argmin(np.abs(np.log(np.array(scales))-np.log(a.scale))))
        ranked=[]
        for node in nodes:
            if node['level']!=level or node['gaussians']<20: continue
            b=node['descriptor_begin']; e=b+node['descriptor_count']
            ranked.append((float(np.sum((proto[b:e]@q)*weights[b:e])),node['id'],node['name']))
        ranked.sort(reverse=True)
        if not ranked: raise ValueError('No supported regions at this scale')
        best=ranked[0][1]; mask=h['ids'][level]==best
        print(json.dumps(dict(query=a.text,level=level,ranking=ranked[:8],selected=int(mask.sum()))))
    elif a.gaussian is not None:
        f=np.load(root/'affinity-fp16.npy').astype('float32'); gate=scale_gate(dict(np.load(root/'gate-weights.npz')),a.scale)
        if not 0<=a.gaussian<len(f): raise ValueError('Gaussian index out of range')
        f*=gate; f/=np.linalg.norm(f,axis=1,keepdims=True).clip(1e-8)
        mask=f@f[a.gaussian]>=a.threshold; print('SELECTED',int(mask.sum()))
    else: p.error('Choose --text or --gaussian')
    np.save(a.output,mask)


if __name__=='__main__': main()
