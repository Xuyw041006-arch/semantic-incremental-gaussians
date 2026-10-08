"""Scale-conditioned affinity distillation and sparse object language descriptors.

Inspired by SAGA's scale gate and LaGa's view-dependent descriptors. This is a
bounded-memory implementation, not either authors' official reproduction.
2D masks are SAM predictions, not semantic ground truth. Geometry is fixed here.
"""
import argparse
import json
import time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from .colmap import read_scene,image_rgb
from .train import render
from .hierarchy import mask_physical_scale,scale_layers,multiview_descriptors,infer_parent_ids
from .extract_masks import overlay,upgrade_clip_cache


def load_checkpoint(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    params={k:v.cuda().requires_grad_(False) for k,v in ck['params'].items()}
    scene=read_scene(ck['args']['data'],ck['args']['width'])
    return ck,params,scene


def small_camera(im,width=384):
    im=im.copy(); w=min(width,im['width']); h=round(im['height']*w/im['width'])
    im['K']=im['K'].copy(); im['K'][0]*=w/im['width']; im['K'][1]*=h/im['height']
    im['width']=w; im['height']=h
    return im


@torch.no_grad()
def prepare_views(params,scene,maskdir,end,targets):
    views=[]
    for row in json.loads((maskdir/'index.json').read_text()):
        if row['index']>=end: continue
        im=small_camera(scene['images'][row['index']]); h,w=im['height'],im['width']
        data=np.load(maskdir/f'masks-{row["index"]:04d}.npz')
        masks=np.unpackbits(data['masks'],axis=2)[:,:,:int(data['shape'][2])].astype(bool)
        masks=np.array([np.asarray(Image.fromarray(m).resize((w,h),Image.Resampling.NEAREST)) for m in masks])
        pred,alpha,_=render(params,im,2,mode='RGB+ED')
        depth=pred[0,...,3].cpu().numpy(); valid=alpha[0,...,0].cpu().numpy()>.5
        scales=np.array([mask_physical_scale(m&valid,depth,im['K']) for m in masks])
        layers=scale_layers(masks,scales,targets); layers[:,~valid]=0
        views.append(dict(im=im,layers=layers,scales=scales,clip=data['clip'].astype('float32'),
                          metadata=row,depth=depth,alpha=valid))
    return views


def sampled_pixels(labels,rng,per_mask=12,max_masks=16):
    present=np.unique(labels); present=present[present>0]
    if len(present)>max_masks: present=rng.choice(present,max_masks,replace=False)
    ids=[]
    for label in present:
        where=np.flatnonzero(labels.ravel()==label)
        if len(where)<4: continue
        ids.extend(rng.choice(where,min(per_mask,len(where)),replace=False))
    return np.asarray(ids,dtype=np.int64)


def pair_loss(features,labels):
    sim=features@features.T
    equal=labels[:,None]==labels[None,:]
    off=~torch.eye(len(labels),device='cuda',dtype=torch.bool)
    pos=equal&off; neg=~equal&off
    if not pos.any() or not neg.any(): return None
    logits=(sim-.3)*10
    loss=.5*F.softplus(-logits[pos]).mean()+.5*F.softplus(logits[neg]).mean()
    accuracy=.5*((sim[pos]>.5).float().mean()+(sim[neg]<.5).float().mean())
    return loss,accuracy


@torch.no_grad()
def visible_projection(params,view):
    xyz=params['means']; im=view['im']; h,w=im['height'],im['width']
    mat=torch.as_tensor(np.linalg.inv(im['c2w']),device='cuda',dtype=torch.float32)
    p=xyz@mat[:3,:3].T+mat[:3,3]; z=p[:,2]
    K=torch.as_tensor(im['K'],device='cuda',dtype=torch.float32)
    uv=p@K.T; xy=(uv[:,:2]/z[:,None]).round().long()
    good=(z>.01)&(xy[:,0]>=0)&(xy[:,0]<w)&(xy[:,1]>=0)&(xy[:,1]<h)&(params['opacities'].sigmoid()>.15)
    ids=torch.where(good)[0]; xx=xy[ids,0]; yy=xy[ids,1]
    depth=torch.as_tensor(view['depth'],device='cuda')[yy,xx]
    good=(z[ids]-depth).abs()<(.025+.035*depth)
    return ids[good].cpu().numpy(),yy[good].cpu().numpy(),xx[good].cpu().numpy()


def associate_nodes(params,views,features,gate,targets,output,maskdir):
    n=len(params['means']); assignments=np.zeros((3,n),np.int32); confidence=np.zeros((3,n),np.float32)
    nodes=[{}, {}, {}]; counters=[0,0,0]
    feature_cpu=F.normalize(features.detach(),dim=-1).cpu().numpy()
    with torch.no_grad():
        gates=[gate(torch.tensor([[np.log(s)]],device='cuda',dtype=torch.float32))[0].cpu().numpy() for s in targets]
    for view in views:
        ids,y,x=visible_projection(params,view)
        for level in range(3):
            mask_ids=view['layers'][level,y,x]
            old=assignments[level,ids].copy()
            for mid in np.unique(mask_ids):
                if mid<=0: continue
                selected=mask_ids==mid; gs=ids[selected]
                if len(gs)<20: continue
                feat=view['clip'][mid-1]; feat=feat/max(np.linalg.norm(feat),1e-8)
                affinity=feature_cpu[gs]*gates[level]
                affinity/=np.linalg.norm(affinity,axis=1,keepdims=True).clip(1e-8)
                affinity=affinity.mean(0); affinity/=max(np.linalg.norm(affinity),1e-8)
                vals,counts=np.unique(old[selected],return_counts=True)
                best=0; best_score=0.
                for candidate,intersection in zip(vals,counts):
                    if candidate<=0: continue
                    visible_old=np.count_nonzero(old==candidate)
                    overlap=intersection/max(1,min(len(gs),visible_old))
                    proto=nodes[level][int(candidate)]['features'][-1]
                    sem=float(feat@proto)
                    aff=float(affinity@nodes[level][int(candidate)]['affinities'][-1])
                    score=overlap*max(0,sem)*(.7+.3*max(0,aff))
                    if overlap>.35 and sem>.65 and aff>.35 and score>best_score: best=int(candidate); best_score=score
                if best==0:
                    counters[level]+=1; best=counters[level]
                    nodes[level][best]=dict(features=[],views=[],mask_ids=[],affinities=[])
                node=nodes[level][best]; node['features'].append(feat); node['affinities'].append(affinity)
                node['views'].append(view['im']['index']); node['mask_ids'].append(int(mid))
                strength=float(view['metadata']['predictions'][mid-1]['iou'])
                change=(confidence[level,gs]<strength+.03)|(assignments[level,gs]==best)
                assignments[level,gs[change]]=best; confidence[level,gs[change]]=strength
        print('LIFT',view['im']['index'],'visible',len(ids),'nodes',counters,flush=True)
    text=np.load(maskdir/'text-features.npy') if (maskdir/'text-features.npy').exists() else None
    # Descriptor banks stay per object. Never allocate N_Gaussians x 512 language.
    descriptions=[]; all_prototypes=[]; all_weights=[]; offsets=[0]
    vocab=json.loads((maskdir/'vocabulary.json').read_text()) if text is not None else []
    parents=[{},infer_parent_ids(assignments[0],assignments[1]),infer_parent_ids(assignments[1],assignments[2])]
    for level,bank in enumerate(nodes):
        for nid,node in bank.items():
            prototypes,weights=multiview_descriptors(node['features'])
            score=(prototypes@text.T*weights[:,None]).sum(0) if text is not None else np.array([0.])
            order=np.argsort(score)[::-1]
            count=int(np.count_nonzero(assignments[level]==nid))
            descriptions.append(dict(level=level,id=nid,parent=parents[level].get(nid,0),gaussians=count,
                observations=len(set(node['views'])),views=sorted(set(node['views'])),
                name=vocab[order[0]] if vocab else 'unknown',clip_cosine=float(score[order[0]]),
                alternatives=[dict(name=vocab[j],score=float(score[j])) for j in order[:3]] if vocab else [],
                descriptor_begin=offsets[-1],descriptor_count=len(prototypes)))
            all_prototypes.extend(prototypes); all_weights.extend(weights); offsets.append(offsets[-1]+len(prototypes))
    np.savez_compressed(output/'hierarchy.npz',ids=assignments,confidence=confidence.astype('float16'),
        prototypes=np.asarray(all_prototypes,dtype='float16'),weights=np.asarray(all_weights,dtype='float16'))
    (output/'nodes.json').write_text(json.dumps(descriptions,indent=2))
    return assignments,descriptions


def run(args):
    torch.manual_seed(11); np.random.seed(11); torch.set_num_threads(2)
    output=Path(args.output); output.mkdir(parents=True,exist_ok=True); maskdir=Path(args.masks)
    ck,params,scene=load_checkpoint(args.checkpoint); targets=[.7,.25,.08]
    upgrade_clip_cache(maskdir,scene)
    views=prepare_views(params,scene,maskdir,ck['end'],targets)
    observed_scales=np.concatenate([v['scales'][v['scales']>0] for v in views])
    # Adapt granularity to the actual scene; COLMAP scale is arbitrary.
    targets=np.percentile(observed_scales,[80,50,20]).clip(.015,2.).tolist()
    for view in views:
        data=np.load(maskdir/f'masks-{view["im"]["index"]:04d}.npz')
        masks=np.unpackbits(data['masks'],axis=2)[:,:,:int(data['shape'][2])].astype(bool)
        w,h=view['im']['width'],view['im']['height']
        masks=np.array([np.asarray(Image.fromarray(m).resize((w,h),Image.Resampling.NEAREST)) for m in masks])
        view['layers']=scale_layers(masks,view['scales'],targets); view['layers'][:,~view['alpha']]=0
    print('ADAPTIVE_QUERY_SCALES',targets,flush=True)
    if len(views)<6: raise ValueError('Need at least six predicted-mask keyframes')
    train=[v for i,v in enumerate(views) if i%5!=0]; val=[v for i,v in enumerate(views) if i%5==0]
    features=torch.nn.Parameter(torch.randn(len(params['means']),16,device='cuda')*.1)
    gate=torch.nn.Sequential(torch.nn.Linear(1,32),torch.nn.ReLU(),torch.nn.Linear(32,16),torch.nn.Sigmoid()).cuda()
    opt=torch.optim.Adam([features,*gate.parameters()],lr=.003)
    rng=np.random.default_rng(11); records=[]; begin=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
    def one(view,level):
        rendered,alpha,_=render(params,view['im'],colors=F.normalize(features,dim=-1))
        f=rendered[0]/alpha[0].clamp_min(1e-5)
        g=gate(torch.tensor([[np.log(targets[level])]],device='cuda',dtype=torch.float32))[0]
        sampled=sampled_pixels(view['layers'][level],rng)
        if len(sampled)<24: return None
        pixel=F.normalize(f.reshape(-1,16)[torch.as_tensor(sampled,device='cuda')]*g,dim=-1)
        labels=torch.as_tensor(view['layers'][level].ravel()[sampled],device='cuda')
        return pair_loss(pixel,labels)
    def validation():
        nonlocal rng
        old_rng=rng; rng=np.random.default_rng(2026); result=[]
        with torch.no_grad():
            for level in range(3):
                values=[one(v,level) for v in val]
                values=[float(a[1]) for a in values if a is not None]
                result.append(float(np.mean(values)) if values else None)
        rng=old_rng; return result
    initial_validation=validation()
    for step in range(args.steps):
        view=train[int(rng.integers(len(train)))]; level=int(rng.integers(3))
        out=one(view,level)
        if out is None: continue
        loss,acc=out; opt.zero_grad(); loss.backward(); opt.step()
        if step%100==0:
            with torch.no_grad():
                values=[one(v,level) for v in val]; values=[float(a[1]) for a in values if a is not None]
            row=dict(step=step,loss=float(loss.detach()),train_pair_accuracy=float(acc.detach()),
                     validation_pair_accuracy=float(np.mean(values)) if values else None,level=level)
            records.append(row); print('AFFINITY',json.dumps(row),flush=True)
    with torch.no_grad():
        np.save(output/'affinity-fp16.npy',F.normalize(features,dim=-1).cpu().numpy().astype('float16'))
        gates=np.array([gate(torch.tensor([[np.log(s)]],device='cuda',dtype=torch.float32))[0].cpu().numpy() for s in targets])
        np.save(output/'scale-gates.npy',gates)
    final_validation=validation()
    torch.save(dict(gate=gate.state_dict(),targets=targets),output/'gate.pt')
    np.savez(output/'gate-weights.npz',**{k:v.detach().cpu().numpy() for k,v in gate.state_dict().items()})
    assignments,nodes=associate_nodes(params,views,features,gate,targets,output,maskdir)
    for view in views[::max(1,len(views)//6)]:
        im=view['im']; rgb=image_rgb(im); panels=[rgb]
        for level in range(3):
            palette=np.random.default_rng(13).uniform(.15,.95,(max(1,int(assignments[level].max())+1),3)).astype('float32'); palette[0]=.15
            colors=torch.as_tensor(palette[assignments[level]],device='cuda')
            with torch.no_grad(): pred,_,_=render(params,im,colors=colors)
            panels.append((pred[0].clamp(0,1).cpu().numpy()*255).astype('uint8'))
        Image.fromarray(np.concatenate(panels,axis=1)).save(output/f'levels-{im["index"]:04d}.jpg')
    stats=dict(gaussians=len(params['means']),mask_keyframes=len(views),affinity_steps=args.steps,
        seconds=time.perf_counter()-begin,gpu_peak_mib=torch.cuda.max_memory_allocated()/1024**2,
        feature_dimension=16,query_scales=targets,units='normalized COLMAP units; not meters',
        initial_validation_pair_accuracy=initial_validation,final_validation_pair_accuracy=final_validation,
        assignment_coverage=[float((a>0).mean()) for a in assignments],
        nodes=[sum(n['level']==l and n['gaussians']>0 for n in nodes) for l in range(3)],
        learned_affinity_mib=len(params['means'])*16*2/1024**2,
        dense_clip_512_fp16_mib_avoided=len(params['means'])*512*2/1024**2,
        qualification='SAM mask pair consistency, not human-annotated mIoU/PQ; post-mapping semantic pass')
    (output/'metrics.json').write_text(json.dumps(dict(summary=stats,training=records),indent=2))
    print('SEMANTIC_FIELD_COMPLETE',json.dumps(stats),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--masks',required=True)
    p.add_argument('--output',required=True); p.add_argument('--steps',type=int,default=1200)
    run(p.parse_args())
