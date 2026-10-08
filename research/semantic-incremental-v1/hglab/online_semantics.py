"""Prefix-only persistent region memory. Fine masks are candidate parts, not GT."""
import argparse,json,time
from pathlib import Path
import numpy as np
from PIL import Image
from .readers import load_scene,project,depth_image,sparse_depth
from .hierarchy import scale_layers


def masks_for(row,folder,im):
    data=np.load(folder/f'masks-{row["index"]:04d}.npz')
    masks=np.unpackbits(data['masks'],axis=2)[:,:,:int(data['shape'][2])].astype(bool)
    masks=np.asarray([np.asarray(Image.fromarray(m).resize((im['width'],im['height']),Image.Resampling.NEAREST)) for m in masks])
    return masks,data['clip'].astype(np.float32)


def mask_extents(masks,xyz,y,x):
    scales=[]
    for m in masks:
        p=xyz[m[y,x]]
        scales.append(float(2*np.sqrt(np.var(p,axis=0).sum())) if len(p)>=6 else 0.)
    return np.asarray(scales)


def prepare(args):
    start=time.perf_counter();scene=load_scene(args.data,args.width);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    folder=Path(args.masks);rows=json.loads((folder/'index.json').read_text())
    train=[r for r in rows if not scene['images'][r['index']]['test']]
    n=len(scene['points']);labels=np.zeros((2,n),np.int32);conf=np.zeros((2,n),np.float32)
    nodes={};next_id=1;targets=None;consumed=[];total=len(scene['images'])
    np.savez_compressed(out/'anchors.npz',points=scene['points'],colors=scene['colors'],release=scene['release'])
    for stage in range(1,args.stages+1):
        end=round(total*stage/args.stages);prev=round(total*(stage-1)/args.stages);boxes={}
        for row in [r for r in train if prev<=r['index']<end]:
            idx=row['index'];im=scene['images'][idx];available=np.flatnonzero(scene['release']<=idx)
            xyz=scene['points'][available]
            depth=depth_image(im) if 'depth_path' in im else sparse_depth(xyz,im)
            visible,y,x,_=project(xyz,im,depth,scene['depth_tolerance']);anchor=available[visible]
            masks,features=masks_for(row,folder,im);scales=mask_extents(masks,scene['points'][anchor],y,x)
            if targets is None:
                valid=scales[scales>0]
                if len(valid)<3:continue
                targets=np.percentile(valid,[65,20]).clip(.015,2.).tolist()
            layers=scale_layers(masks,scales,targets)
            # A repeated whole-object mask does not create a synthetic part.
            layers[1,layers[0]==layers[1]]=0
            for level in range(2):
                local=layers[level,y,x]
                for mid in np.unique(local):
                    if mid<=0:continue
                    selected=local==mid;gs=anchor[selected]
                    if len(gs)<6:continue
                    feat=features[mid-1];feat/=max(np.linalg.norm(feat),1e-8)
                    quality=float(row['predictions'][mid-1]['iou'])*float(row['predictions'][mid-1]['stability'])
                    parent=0
                    if level:
                        vals,counts=np.unique(labels[0,gs],return_counts=True);good=vals>0
                        if good.any():
                            vals=vals[good];counts=counts[good];j=counts.argmax()
                            if counts[j]/len(gs)>=.65:parent=int(vals[j])
                    old=labels[level,gs];vals,counts=np.unique(old,return_counts=True)
                    best=0;best_score=0.
                    for candidate,count in zip(vals,counts):
                        if candidate<=0:continue
                        node=nodes[str(candidate)]
                        if node['level']!=level or (level and parent and node['parent'] not in (0,parent)):continue
                        sem=max(float(feat@np.asarray(v)) for v in node['descriptors'])
                        overlap=count/max(1,len(gs));score=overlap*max(0,sem)
                        if overlap>=.25 and sem>=.65 and score>best_score:best=int(candidate);best_score=score
                    if not best:
                        if len(nodes)>=1024:continue
                        best=next_id;next_id+=1
                        nodes[str(best)]=dict(id=best,level=level,parent=parent,born_stage=stage,born_frame=idx,
                            observations=[],descriptors=[],names=[],confidence=quality,relation='candidate_containment' if level else 'object_region')
                    node=nodes[str(best)]
                    if parent and node['parent']==0:node['parent']=parent
                    node['observations'].append(idx);node['names'].append(row['predictions'][mid-1]['label'])
                    # Bounded multi-view bank; distinct views retained without future clustering.
                    if not node['descriptors'] or max(float(feat@np.asarray(v)) for v in node['descriptors'])<.95:
                        if len(node['descriptors'])<4:node['descriptors'].append(feat.astype(np.float16).tolist())
                    node['confidence']=.8*node['confidence']+.2*quality
                    update=(conf[level,gs]<quality+.02)|(labels[level,gs]==best)
                    labels[level,gs[update]]=best;conf[level,gs[update]]=quality
                    yy,xx=np.where(layers[level]==mid)
                    if len(xx):boxes.setdefault(str(idx),[]).append(dict(center=[float(xx.mean()),float(yy.mean())],
                        node=best,level=level,confidence=quality))
            consumed.append(idx)
        # Current graph may revise past frame coverage, using only currently known semantics.
        known=np.flatnonzero(scene['release']<end);coverage={};all_boxes={}
        for im in scene['images'][:end]:
            if im['test']:continue
            xyz=scene['points'][known];d=depth_image(im) if 'depth_path' in im else sparse_depth(xyz,im)
            ids,y,x,_=project(xyz,im,d,scene['depth_tolerance']);a=known[ids];entries=[];regions=[]
            for level in range(2):
                v=labels[level,a];c=conf[level,a]
                for node_id in np.unique(v[c>=.75]):
                    if node_id<=0:continue
                    mask=(v==node_id)&(c>=.75)
                    if mask.sum()<6:continue
                    # Encoded coverage tokens preserve the object/fine weighting.
                    entries.append(int(node_id)*2+level)
                    regions.append(dict(center=[float(x[mask].mean()),float(y[mask].mean())],node=int(node_id),
                        level=level,confidence=float(c[mask].mean())))
            coverage[str(im['index'])]=entries;all_boxes[str(im['index'])]=regions
        births={str(int(k)*2+v['level']):v['born_stage'] for k,v in nodes.items()}
        np.savez_compressed(out/f'prefix-{stage}.npz',ids=labels[:,:len(scene['points'])],confidence=conf.astype(np.float16))
        record=dict(stage=stage,end=end,targets=targets,nodes=nodes,coverage=coverage,boxes=all_boxes,births=births,
            consumed_teacher_frames=consumed.copy(),max_teacher_frame=max(consumed,default=-1),
            released_anchors=int((scene['release']<end).sum()),units=scene['units'],
            qualification='SAM candidate region hierarchy; containment is not verified named-part ground truth')
        assert record['max_teacher_frame']<end
        assert all(not scene['images'][i]['test'] for i in consumed)
        (out/f'prefix-{stage}.json').write_text(json.dumps(record))
        print('PREFIX_GRAPH',stage,'end',end,'nodes',len(nodes),'consumed',len(consumed),flush=True)
    (out/'preparation.json').write_text(json.dumps(dict(seconds=time.perf_counter()-start,stages=args.stages,
        anchors=n,teacher_rows=len(rows),training_teacher_rows=len(train),kind=scene['kind'],width=args.width,
        consumed_test_masks=False,normalization='first-pose meters' if scene['kind'].startswith('RGBD') else 'offline COLMAP global normalization')))
    print('PREFIX_PREPARATION_COMPLETE',time.perf_counter()-start,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--masks',required=True)
    p.add_argument('--output',required=True);p.add_argument('--width',type=int,default=640);p.add_argument('--stages',type=int,default=6)
    prepare(p.parse_args())
