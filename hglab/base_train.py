"""Budgeted incremental anisotropic 3DGS, using official gsplat CUDA kernels.

Reconstruction uses offline COLMAP poses/coordinates. RGB training, point release,
densification and bounded keyframe replay are restricted to the arrived prefix.
This is a posed mapping experiment, not a camera-tracking SLAM implementation.
"""
import argparse
from collections import OrderedDict
from pathlib import Path
import json
import time
import math
import random
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree
import torch
import torch.nn.functional as F
from gsplat import rasterization
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate, split
from .colmap import read_scene, image_rgb, metadata


class BudgetStrategy(DefaultStrategy):
    def __init__(self, cap, **kw):
        super().__init__(**kw)
        self.cap = cap

    @torch.no_grad()
    def _grow_gs(self, params, optimizers, state, step):
        room = max(0,self.cap-len(params['means']))
        score = state['grad2d']/state['count'].clamp_min(1)
        candidates = torch.where(score>self.grow_grad2d)[0]
        if room == 0 or not len(candidates): return 0,0
        candidates = candidates[torch.argsort(score[candidates],descending=True)[:room]]
        selected = torch.zeros_like(score,dtype=torch.bool); selected[candidates]=True
        small = params['scales'].exp().max(-1).values<=self.grow_scale3d*state['scene_scale']
        dupli = selected&small; spl = selected&~small
        nd,ns = int(dupli.sum()),int(spl.sum())
        if nd: duplicate(params,optimizers,state,dupli)
        if ns:
            spl = torch.cat([spl,torch.zeros(nd,device=spl.device,dtype=torch.bool)])
            split(params,optimizers,state,spl,revised_opacity=self.revised_opacity)
        return nd,ns


def initial_values(xyz, rgb, degree):
    dist = cKDTree(xyz).query(xyz,k=min(4,len(xyz)))[0]
    scale = np.sqrt(np.mean(dist[:,1:]**2,axis=1)).clip(0.0003,0.1)
    n = len(xyz); sh = np.zeros((n,(degree+1)**2,3),np.float32)
    sh[:,0] = (rgb/255.-.5)/.28209479177387814
    q = np.zeros((n,4),np.float32); q[:,0]=1
    return dict(means=xyz.astype(np.float32),scales=np.log(scale).astype(np.float32)[:,None].repeat(3,1),
                quats=q,opacities=np.full(n,math.log(.1/.9),np.float32),sh0=sh[:,:1],shN=sh[:,1:])


@torch.no_grad()
def append_points(params, optimizers, state, values):
    n = len(values['means'])
    for key in params:
        old = params[key]
        new = torch.nn.Parameter(torch.cat([old,torch.as_tensor(values[key],device=old.device)]))
        opt = optimizers[key]; os = opt.state.pop(old,{})
        for k,v in os.items():
            if isinstance(v,torch.Tensor) and v.ndim and v.shape == old.shape:
                os[k] = torch.cat([v,torch.zeros_like(new[-n:])])
        opt.param_groups[0]['params']=[new]; opt.state[new]=os; params[key]=new
    for key,value in list(state.items()):
        if key!='anchor_ids' and isinstance(value,torch.Tensor):
            state[key]=torch.cat([value,torch.zeros((n,*value.shape[1:]),device=value.device,dtype=value.dtype)])


def ssim_loss(x,y):
    # Differentiable SSIM over 7x7 windows, full image, RGB channel-first.
    x=x.permute(2,0,1)[None]; y=y.permute(2,0,1)[None]
    mx=F.avg_pool2d(x,7,1,3); my=F.avg_pool2d(y,7,1,3)
    vx=F.avg_pool2d(x*x,7,1,3)-mx*mx; vy=F.avg_pool2d(y*y,7,1,3)-my*my
    cov=F.avg_pool2d(x*y,7,1,3)-mx*my
    return 1-(((2*mx*my+.01**2)*(2*cov+.03**2))/
              ((mx*mx+my*my+.01**2)*(vx+vy+.03**2))).mean()


def render(params, im, degree=2, mode='RGB', colors=None, grad=False):
    return rasterization(means=params['means'],quats=params['quats'],
        scales=params['scales'].exp(),opacities=params['opacities'].sigmoid(),
        colors=torch.cat([params['sh0'],params['shN']],1) if colors is None else colors,
        sh_degree=degree if colors is None else None,
        viewmats=torch.as_tensor(np.linalg.inv(im['c2w']),device='cuda',dtype=torch.float32)[None],
        Ks=torch.as_tensor(im['K'],device='cuda',dtype=torch.float32)[None],
        width=im['width'],height=im['height'],packed=True,render_mode=mode,
        absgrad=grad,near_plane=.01,far_plane=100.)


@torch.no_grad()
def evaluate(params,scene,end,output,stage,degree):
    from skimage.metrics import structural_similarity
    rows=[]
    for im in scene['images'][:end]:
        if not im['test']: continue
        pred,alpha,_=render(params,im,degree)
        pred=pred[0].clamp(0,1).cpu().numpy(); target=image_rgb(im)/255.
        mse=float(np.mean((pred-target)**2))
        row=dict(index=im['index'],name=im['name'],psnr=-10*np.log10(max(mse,1e-10)),
                 ssim=float(structural_similarity(target,pred,data_range=1,channel_axis=2)),
                 coverage=float((alpha>.1).float().mean()))
        rows.append(row)
        if im['index'] in (0,8,16) or im['index']==next((i for i in range(end-1,-1,-1) if i%scene['test_every']==0),0):
            panel=np.concatenate([target,pred],axis=1)
            Image.fromarray((panel*255).astype('uint8')).save(output/f'stage-{stage}-view-{im["index"]}.jpg')
    (output/f'quality-{stage}.json').write_text(json.dumps(rows,indent=2))
    return rows


@torch.no_grad()
def export_map(params,path):
    vals={k:v.detach().cpu().numpy() for k,v in params.items()}
    extra={'features':vals['features'].astype('float16')} if 'features' in vals else {}
    np.savez_compressed(path,means=vals['means'].astype('float32'),
        scales=np.exp(vals['scales']).astype('float16'),quats=vals['quats'].astype('float16'),
        opacity=(1/(1+np.exp(-vals['opacities']))).astype('float16'),
        sh=np.concatenate([vals['sh0'],vals['shN']],1).astype('float16'),**extra)


def run(args):
    torch.manual_seed(args.seed); np.random.seed(args.seed); random.seed(args.seed)
    torch.set_num_threads(2)
    scene=read_scene(args.data,args.width); output=Path(args.output); output.mkdir(parents=True,exist_ok=True)
    ims=scene['images']; total=len(ims); released=np.zeros(len(scene['points']),bool)
    params=None; optimizers={}; state=None; history=[]; reservoir=[]; observed=0; step=0
    strategy=BudgetStrategy(args.cap,refine_start_iter=150,refine_stop_iter=args.stages*args.steps-800,
        refine_every=100,reset_every=3000,grow_grad2d=.00035,absgrad=True,
        prune_scale3d=.3,verbose=False)
    cache=OrderedDict()
    def pixels(i):
        if i not in cache:
            cache[i]=torch.as_tensor(image_rgb(ims[i])/255.,dtype=torch.float32)
            if len(cache)>32: cache.popitem(last=False)
        cache.move_to_end(i)
        return cache[i].to('cuda')
    start=time.perf_counter()
    for stage in range(args.stages):
        end=round(total*(stage+1)/args.stages); prev=round(total*stage/args.stages)
        current=[i for i in range(prev,end) if not ims[i]['test']]
        old_reservoir=reservoir.copy()
        for i in current:
            observed+=1
            if len(reservoir)<48: reservoir.append(i)
            else:
                j=random.randrange(observed)
                if j<48: reservoir[j]=i
        stage_cap=round(args.cap*(stage+1)/args.stages)
        strategy.cap=stage_cap
        point_ids=np.where((scene['release']<end)&~released)[0]
        room=stage_cap-(0 if params is None else len(params['means']))
        point_ids=point_ids[:max(0,room)]
        if len(point_ids):
            values=initial_values(scene['points'][point_ids],scene['colors'][point_ids],args.degree)
            released[point_ids]=True
            if params is None:
                params=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.as_tensor(v,device='cuda')) for k,v in values.items()})
                lr=dict(means=.00016,scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.000125)
                optimizers={k:torch.optim.Adam([v],lr=lr[k],eps=1e-15) for k,v in params.items()}
                state=strategy.initialize_state(scene_scale=1.)
                strategy.check_sanity(params,optimizers)
            else: append_points(params,optimizers,state,values)
        torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); begin=time.perf_counter(); times=[]
        for local in range(args.steps):
            # Fixed-size reservoir plus latest chunk; no unseen RGB targets.
            pool=old_reservoir if old_reservoir and random.random()<args.replay else current
            i=random.choice(pool); im=ims[i]; target=pixels(i)
            degree=min(args.degree,step//1000)
            optimizers['means'].param_groups[0]['lr']=.00016*(.01**(step/(args.stages*args.steps)))
            torch.cuda.synchronize(); tick=time.perf_counter()
            pred,alpha,info=render(params,im,degree,grad=True)
            strategy.step_pre_backward(params,optimizers,state,step,info)
            # L1-only fitting was stable in the real-data gradient diagnostic.
            # SSIM is an evaluation metric; the unvalidated pooled-SSIM training
            # objective is deliberately excluded from the production experiment.
            pred=pred[0]; loss=(pred-target).abs().mean()
            loss.backward()
            for opt in optimizers.values(): opt.step(); opt.zero_grad(set_to_none=True)
            strategy.step_post_backward(params,optimizers,state,step,info,packed=True)
            with torch.no_grad(): params['scales'].clamp_(math.log(.00005),math.log(.5))
            torch.cuda.synchronize(); times.append((time.perf_counter()-tick)*1000); step+=1
            if local%200==0:
                print(json.dumps(dict(stage=stage+1,step=step,arrived=end,gaussians=len(params['means']),loss=float(loss.detach()),ms=round(times[-1],1))),flush=True)
        torch.cuda.synchronize(); elapsed=time.perf_counter()-begin
        peak=torch.cuda.max_memory_allocated()/1024**2
        quality=evaluate(params,scene,end,output,stage+1,min(args.degree,(step-1)//1000))
        row=dict(stage=stage+1,arrived=end,train_frames=sum(not im['test'] for im in ims[:end]),
                 gaussians=len(params['means']),cap=stage_cap,steps=args.steps,chunk_seconds=elapsed,
                 update_p50_ms=float(np.median(times)),update_p95_ms=float(np.percentile(times,95)),
                 gpu_peak_mib=peak,psnr=float(np.mean([q['psnr'] for q in quality])),
                 ssim=float(np.mean([q['ssim'] for q in quality])),
                 old_view_psnr=float(np.mean([q['psnr'] for q in quality if q['index']<round(total/args.stages)])))
        history.append(row); (output/'metrics.json').write_text(json.dumps(history,indent=2))
        # Display LOD only; the full optimized map remains in checkpoint/map.npz.
        idx=torch.linspace(0,len(params['means'])-1,min(60000,len(params['means'])),device='cuda').long()
        export_map({k:v[idx] for k,v in params.items()},output/f'preview-{stage+1}.npz')
        checkpoint=dict(params={k:v.detach().cpu() for k,v in params.items()},args=vars(args),end=end,step=step)
        torch.save(checkpoint,output/'checkpoint.pt')
        torch.save(checkpoint,output/f'checkpoint-{stage+1}.pt')
        print('STAGE_COMPLETE',json.dumps(row),flush=True)
    export_map(params,output/'map.npz')
    meta=dict(dataset='Mip-NeRF 360 kitchen (real photographs)',**metadata(scene),
              protocol='Filename-ordered chunks; offline COLMAP poses + points; >=2 arrived train observations release points; every 8th image held out.',
              args=vars(args),wall_seconds=time.perf_counter()-start,device=torch.cuda.get_device_name())
    (output/'metadata.json').write_text(json.dumps(meta,indent=2))
    print('GEOMETRY_COMPLETE',len(params['means']),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--output',required=True)
    p.add_argument('--cap',type=int,default=600000); p.add_argument('--width',type=int,default=800)
    p.add_argument('--stages',type=int,default=6); p.add_argument('--steps',type=int,default=1500)
    p.add_argument('--degree',type=int,default=2); p.add_argument('--replay',type=float,default=.3); p.add_argument('--seed',type=int,default=7)
    run(p.parse_args())
