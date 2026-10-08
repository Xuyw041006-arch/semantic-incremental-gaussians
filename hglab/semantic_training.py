"""Fuse arrived SAM/CLIP supervision while optimizing the Gaussian map."""
from collections import OrderedDict
from pathlib import Path
import numpy as np
import torch
from .priority import descriptor_table


def render_features(params,im,degree=0):
    """Fit semantic attributes without noisy teacher features dragging geometry."""
    from .base_train import render
    fixed={k:(v if k=='features' else v.detach()) for k,v in params.items()}
    return render(fixed,im,degree,colors=params['features'])


class SemanticSupervisor:
    def __init__(self,folder,meta,raw,dimensions=16):
        self.folder=Path(folder);self.meta=meta;self.raw=raw;self.cache=OrderedDict()
        self.features=torch.as_tensor(descriptor_table(meta['nodes'],dimensions),device='cuda')
        self.priority=torch.ones(len(self.features),device='cuda')
        self.background=torch.zeros(len(self.features),device='cuda')
        for key,value in meta.get('priorities',{}).items():
            self.priority[int(key)]=value['priority'];self.background[int(key)]=float(value['background'])

    def initial_features(self,anchors):
        labels=self.raw['ids'][:,anchors];chosen=np.where(labels[1]>0,labels[1],labels[0])
        return self.features[torch.as_tensor(chosen,device='cuda')].cpu().numpy()

    @torch.no_grad()
    def refresh(self,state,step):
        anchors=state['anchor_ids'];n=len(anchors);device=anchors.device
        ids=torch.as_tensor(self.raw['ids'],device=device,dtype=torch.int64)[:,anchors]
        confidence=torch.as_tensor(self.raw['confidence'].astype(np.float32),device=device)[:,anchors]
        state['object_ids']=ids[0].clone();state['fine_ids']=ids[1].clone()
        state['semantic_conf']=confidence.max(0).values
        state['semantic_priority']=torch.as_tensor(self.raw['priority'].astype(np.float32),device=device)[anchors]
        state['boundary_score']=torch.as_tensor(self.raw['boundary'].astype(np.float32),device=device)[anchors]
        state['background_score']=torch.as_tensor(self.raw['background'].astype(np.float32),device=device)[anchors]*confidence[0]
        for key in ('ema_error','ema_contribution','observations','born_step'):
            old=state.get(key)
            if old is None:state[key]=torch.full((n,),float(step) if key=='born_step' else 0.,device=device)
            elif len(old)<n:
                state[key]=torch.cat([old,torch.full((n-len(old),),float(step) if key=='born_step' else 0.,device=device)])

    def frame(self,index,crop=None):
        name=self.meta.get('semantic_frames',{}).get(str(index))
        if name is None:return None
        assert index<self.meta['end'] and index in self.meta['consumed_teacher_frames']
        if index not in self.cache:
            with np.load(self.folder/name) as raw:
                self.cache[index]={k:torch.as_tensor(raw[k].astype(np.float32) if k!='ids' else raw[k].astype(np.int64),device='cuda') for k in raw.files}
            if len(self.cache)>4:self.cache.popitem(last=False)
        self.cache.move_to_end(index);data=self.cache[index]
        if crop is None:return data
        x,y,w,h=crop
        return {k:v[...,y:y+h,x:x+w] for k,v in data.items()}

    def pixel_weights(self,data):
        ids=data['ids'];conf=data['confidence'];good=(ids>0)&(conf>=.75)
        task=torch.where(good,self.priority[ids],1.).max(0).values
        important=(task-1).clamp(0,2)/2
        return 1+.5*important+.5*important*data['boundary']

    @torch.no_grad()
    def observe(self,params,state,info,pred,target,data,step):
        gid=info['gaussian_ids'];xy=info['means2d'].detach().round().long()
        x=xy[:,0];y=xy[:,1];h,w=target.shape[:2]
        valid=(x>=0)&(x<w)&(y>=0)&(y<h)
        gid=gid[valid];x=x[valid];y=y[valid]
        depth=info['depths'][valid];front=pred[0,y,x,3].detach()
        visible=(front>0)&((depth-front).abs()<.035+.04*front)
        gid=gid[visible];x=x[visible];y=y[visible]
        if not len(gid):return
        residual=(pred[0,y,x,:3].detach()-target[y,x]).abs().mean(-1)
        radius=info['radii'][valid][visible].max(-1).values.float()
        contribution=params['opacities'][gid].sigmoid()*(radius/max(h,w)).square().clamp_max(.1)
        seen=state['observations'][gid]>0
        state['ema_error'][gid]=torch.where(seen,.9*state['ema_error'][gid]+.1*residual,residual)
        state['ema_contribution'][gid]=torch.where(seen,.9*state['ema_contribution'][gid]+.1*contribution,contribution)
        state['observations'][gid]+=1
        ids=data['ids'][:,y,x];conf=data['confidence'][:,y,x]
        good=(ids>0)&(conf>=.75)
        for level,key in enumerate(('object_ids','fine_ids')):
            update=good[level]&((conf[level]>=state['semantic_conf'][gid]-.05)|(state[key][gid]==ids[level]))
            state[key][gid[update]]=ids[level,update]
        known=good.any(0)
        if known.any():
            q=gid[known];confidence=conf[:,known].max(0).values
            task=torch.where(good[:,known],self.priority[ids[:,known]],1.).max(0).values
            bg=good[0,known]&self.background[ids[0,known]].bool()&(task<2)
            state['semantic_conf'][q]=confidence
            state['semantic_priority'][q]=torch.where(bg,0.,task)
            state['boundary_score'][q]=.8*state['boundary_score'][q]+.2*data['boundary'][y[known],x[known]]*confidence
            state['background_score'][q]=torch.where(bg,confidence,0.)

    def feature_target(self,data):
        ids=data['ids'];good=(ids>0)&(data['confidence']>=.75)
        chosen=torch.where(good[1],ids[1],torch.where(good[0],ids[0],0))
        return self.features[chosen],chosen>0
