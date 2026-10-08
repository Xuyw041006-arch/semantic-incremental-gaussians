"""Extend open-vocabulary labels without changing masks or learned geometry/features."""
from pathlib import Path
import argparse,json
import numpy as np
import torch,open_clip
from .extract_masks import VOCAB,F_normalize


def run(maskdir,fields):
    maskdir=Path(maskdir); torch.set_num_threads(2)
    clip,_,_=open_clip.create_model_and_transforms('ViT-B-32',pretrained='openai',device='cuda',force_quick_gelu=True)
    tok=open_clip.get_tokenizer('ViT-B-32')
    with torch.inference_mode(): text=F_normalize(clip.encode_text(tok(['a photo of a '+s for s in VOCAB]).cuda()).float()).cpu().numpy()
    np.save(maskdir/'text-features.npy',text); (maskdir/'vocabulary.json').write_text(json.dumps(VOCAB))
    records=json.loads((maskdir/'index.json').read_text())
    for row in records:
        data=np.load(maskdir/f'masks-{row["index"]:04d}.npz'); scores=data['clip'].astype('float32')@text.T
        for j,pred in enumerate(row['predictions']):
            k=scores[j].argmax(); pred['label']=VOCAB[k]; pred['clip_cosine']=float(scores[j,k])
        (maskdir/f'masks-{row["index"]:04d}.json').write_text(json.dumps(row,indent=2))
    (maskdir/'index.json').write_text(json.dumps(records,indent=2))
    for field in fields:
        field=Path(field); h=np.load(field/'hierarchy.npz'); nodes=json.loads((field/'nodes.json').read_text())
        p=h['prototypes'].astype('float32'); weights=h['weights']
        for node in nodes:
            b=node['descriptor_begin']; e=b+node['descriptor_count']; scores=((p[b:e]@text.T)*weights[b:e,None]).sum(0)
            rank=np.argsort(scores)[::-1]; node['name']=VOCAB[rank[0]]; node['clip_cosine']=float(scores[rank[0]])
            node['alternatives']=[dict(name=VOCAB[j],score=float(scores[j])) for j in rank[:3]]
        (field/'nodes.json').write_text(json.dumps(nodes,indent=2))
    print('VOCABULARY_REFRESHED',len(VOCAB),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--masks',required=True); p.add_argument('--fields',nargs='+',required=True); a=p.parse_args()
    run(a.masks,a.fields)
