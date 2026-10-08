"""Real CUDA mapping with independently switchable S/B/R policies.

S: semantic novelty/error frame sampling and targeted, equal-size crop sampling.
B: confidence-gated region-balanced growth, admission and marginal-coverage pruning.
R: object/fine-region coverage replay with an explicit exploration reserve.
Gaussian cloning/splitting inherits anchor IDs via gsplat's state transformations.
"""
import argparse,json,time,random,resource
from pathlib import Path
from collections import OrderedDict
import numpy as np
from PIL import Image
import torch
from gsplat.strategy import DefaultStrategy
from gsplat.strategy.ops import duplicate,split,remove
from .base_train import initial_values,append_points,render,export_map
from .readers import load_scene,depth_image
from .colmap import image_rgb
from .policies import semantic_importance,sampling_weights,coverage_replay,choose_crop


class HierarchicalBudget(DefaultStrategy):
    def __init__(self,cap,enabled=False,**kw):
        super().__init__(**kw);self.cap=cap;self.enabled=enabled
        self.labels=None;self.confidence=None;self.events=[]

    @torch.no_grad()
    def importance(self,state):
        a=state['anchor_ids'];ids=self.labels[:,a];c=self.confidence[:,a]
        score=torch.zeros(len(a),device=a.device)
        for level,weight in enumerate((1.,2.)):
            good=(ids[level]>0)&(c[level]>=.75)
            if good.any():
                counts=torch.bincount(ids[level,good],minlength=int(self.labels.max())+1)
                v=counts[ids[level,good]].clamp_min(1).float().rsqrt();v/=v.mean().clamp_min(1e-8)
                score[good]+=weight*c[level,good]*v.clamp_max(4)
        return score

    @torch.no_grad()
    def _grow_gs(self,params,optimizers,state,step):
        room=max(0,self.cap-len(params['means']))
        if not room:return 0,0
        score=state['grad2d']/state['count'].clamp_min(1)
        candidates=torch.where(score>self.grow_grad2d)[0]
        if not len(candidates):return 0,0
        rank=score[candidates]
        if self.enabled:rank=rank*(1+.35*self.importance(state)[candidates])
        candidates=candidates[torch.argsort(rank,descending=True)[:room]]
        selected=torch.zeros_like(score,dtype=torch.bool);selected[candidates]=True
        small=params['scales'].exp().max(-1).values<=self.grow_scale3d*state['scene_scale']
        dup=selected&small;spl=selected&~small;nd=int(dup.sum());ns=int(spl.sum())
        if nd:duplicate(params,optimizers,state,dup)
        if ns:
            spl=torch.cat([spl,torch.zeros(nd,device=spl.device,dtype=torch.bool)])
            split(params,optimizers,state,spl,revised_opacity=self.revised_opacity)
        self.events.append(dict(step=step,event='growth',clone=nd,split=ns,semantic=self.enabled))
        return nd,ns

    @torch.no_grad()
    def _prune_gs(self,params,optimizers,state,step):
        opa=params['opacities'].sigmoid();scale=params['scales'].exp().max(-1).values
        invalid=(opa<self.prune_opa)|(scale>.3)
        n=len(opa);mask=invalid.clone();reallocated=0;protected=0
        # All variants get the same replacement opportunity and fraction.
        if step%300==0 and n>=self.cap*.95:
            contribution=opa*(scale/scale.median().clamp_min(1e-6)).clamp(.1,10.)
            rank=contribution/contribution.mean().clamp_min(1e-8)
            if self.enabled:
                rank=rank+.5*self.importance(state)
                a=state['anchor_ids'];ids=self.labels[:,a];c=self.confidence[:,a]
                keep=torch.zeros(n,device=opa.device,dtype=torch.bool)
                # One reliable representative per supported region is protected.
                for level in range(2):
                    q=(ids[level]>0)&(c[level]>=.75)&~invalid&(opa>.02)
                    maxima=torch.full((int(self.labels.max())+1,),-1.,device=opa.device)
                    maxima.scatter_reduce_(0,ids[level,q],opa[q],reduce='amax',include_self=True)
                    keep|=q&(opa>=maxima[ids[level]]-1e-8)
                rank[keep]=torch.inf;protected=int(keep.sum())
            eligible=torch.where(~invalid&torch.isfinite(rank))[0]
            count=min(int(n*.02),len(eligible));selected=eligible[torch.argsort(rank[eligible])[:count]]
            mask[selected]=True;reallocated=len(selected)
        count=int(mask.sum())
        if count:remove(params,optimizers,state,mask)
        self.events.append(dict(step=step,event='prune',removed=count,reallocated=reallocated,protected=protected))
        return count


def read_prefix(folder,stage,scene):
    folder=Path(folder);meta=json.loads((folder/f'prefix-{stage}.json').read_text())
    raw=np.load(folder/f'prefix-{stage}.npz');ids=raw['ids'];confidence=raw['confidence'].astype(np.float32)
    assert meta['max_teacher_frame']<meta['end']
    assert all(not scene['images'][i]['test'] for i in meta['consumed_teacher_frames'])
    assert not np.any(ids[:,scene['release']>=meta['end']])
    return meta,ids,confidence


def run(args):
    from .evaluation import evaluate
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    rng=np.random.default_rng(args.seed);flags=tuple(c=='1' for c in args.factors)
    assert len(flags)==3;S,B,R=flags
    scene=load_scene(args.data,args.width);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    params=None;opts={};state=None;released=np.zeros(len(scene['points']),bool);replay=[];seen=[]
    cache=OrderedDict();errors={};history=[];samples=[];global_step=0;controller_seconds=0.
    strategy=HierarchicalBudget(args.cap,enabled=B,refine_start_iter=50,refine_stop_iter=args.steps*args.stages-100,
        refine_every=100,reset_every=1000000000,grow_grad2d=.00025,absgrad=True,prune_scale3d=.3,verbose=False)
    start=time.perf_counter();total=len(scene['images']);positions=np.asarray([im['c2w'][:3,3] for im in scene['images']])
    def pixels(i):
        if i not in cache:
            im=scene['images'][i]
            cache[i]=(torch.from_numpy(image_rgb(im).astype(np.float32)/255),
                torch.from_numpy(depth_image(im)) if 'depth_path' in im else None)
            if len(cache)>32:cache.popitem(last=False)
        cache.move_to_end(i)
        rgb,depth=cache[i];return rgb.cuda(),None if depth is None else depth.cuda()
    for stage in range(1,args.stages+1):
        stage_start=time.perf_counter();tick=time.perf_counter()
        meta,ids,confidence=read_prefix(args.prefixes,stage,scene);end=meta['end'];prev=round(total*(stage-1)/args.stages)
        current=[i for i in range(prev,end) if not scene['images'][i]['test']]
        coverage={int(k):v for k,v in meta['coverage'].items()}
        if R:old_pool=coverage_replay(seen,coverage,48,positions)
        else:old_pool=replay.copy()
        stage_cap=round(args.cap*stage/args.stages);strategy.cap=stage_cap
        strategy.labels=torch.as_tensor(ids,device='cuda',dtype=torch.int64)
        strategy.confidence=torch.as_tensor(confidence,device='cuda')
        available=np.flatnonzero((scene['release']<end)&~released)
        room=stage_cap-(0 if params is None else len(params['means']))
        # Reserve half of the initial/expanded capacity for adaptive refinement.
        count=min(len(available),max(0,room//2))
        if count:
            if B:
                importance=semantic_importance(ids[:,available],confidence[:,available])
                weights=.25+importance;weights/=weights.sum()
                point_ids=rng.choice(available,count,replace=False,p=weights)
            else:point_ids=rng.choice(available,count,replace=False)
            values=initial_values(scene['points'][point_ids],scene['colors'][point_ids],args.degree)
            released[point_ids]=True
            if params is None:
                params=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.as_tensor(v,device='cuda')) for k,v in values.items()})
                rates=dict(means=.00016,scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.000125)
                opts={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in params.items()}
                state=strategy.initialize_state(scene_scale=1.);state['anchor_ids']=torch.as_tensor(point_ids,device='cuda',dtype=torch.int64)
                strategy.check_sanity(params,opts)
            else:
                anchors=state['anchor_ids'];append_points(params,opts,state,values)
                state['anchor_ids']=torch.cat([anchors,torch.as_tensor(point_ids,device='cuda',dtype=torch.int64)])
        assert params is not None
        for i in current:
            seen.append(i)
            if len(replay)<48:replay.append(i)
            else:
                j=int(rng.integers(len(seen)))
                if j<48:replay[j]=i
        controller_seconds+=time.perf_counter()-tick
        torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();times=[];compute_times=[];pixel_count=0
        # A separate fixed stream gives every variant the same replay/crop decisions.
        decisions=np.random.default_rng(args.seed*100+stage)
        replay_flags=decisions.random(args.steps)<.3;crop_flags=decisions.random(args.steps)<.5
        for local in range(args.steps):
            torch.cuda.synchronize();step_start=time.perf_counter();tick=time.perf_counter()
            pool=old_pool if old_pool and replay_flags[local] else current
            if S:
                weights=sampling_weights(pool,coverage,meta['births'],stage,errors)
                index=int(rng.choice(pool,p=weights))
            else:index=int(rng.choice(pool))
            im=scene['images'][index];target,depth=pixels(index)
            crop=None
            if crop_flags[local]:
                im,crop=choose_crop(im,meta['boxes'].get(str(index),[]),rng,semantic=S)
                x,y,w,h=crop;target=target[y:y+h,x:x+w];depth=depth[y:y+h,x:x+w] if depth is not None else None
            pixel_count+=im['width']*im['height'];controller_seconds+=time.perf_counter()-tick
            degree=min(args.degree,global_step//1000)
            opts['means'].param_groups[0]['lr']=.00016*(.1**(global_step/(args.stages*args.steps)))
            torch.cuda.synchronize();compute_start=time.perf_counter()
            pred,alpha,info=render(params,im,degree,mode='RGB+ED' if depth is not None else 'RGB',grad=True)
            strategy.step_pre_backward(params,opts,state,global_step,info)
            rgb_loss=(pred[0,...,:3]-target).abs().mean();loss=rgb_loss
            if depth is not None:
                valid=(depth>.2)&(depth<5.)&(alpha[0,...,0]>.1)
                if valid.any():loss=loss+.1*(pred[0,...,3][valid]-depth[valid]).abs().clamp_max(1.).mean()
            loss.backward()
            for opt in opts.values():opt.step();opt.zero_grad(set_to_none=True)
            strategy.step_post_backward(params,opts,state,global_step,info,packed=True)
            with torch.no_grad():params['scales'].clamp_(np.log(.00005),np.log(.3))
            errors[index]=.9*errors.get(index,float(rgb_loss.detach()))+.1*float(rgb_loss.detach())
            assert len(state['anchor_ids'])==len(params['means'])<=stage_cap
            torch.cuda.synchronize();compute_times.append((time.perf_counter()-compute_start)*1000)
            times.append((time.perf_counter()-step_start)*1000);global_step+=1
            if local%150==0:
                samples.append(dict(stage=stage,step=global_step,frame=index,crop=crop,old=pool is old_pool))
                print('UPDATE',args.factors,args.seed,stage,local,len(params['means']),float(loss.detach()),flush=True)
        torch.cuda.synchronize();training_seconds=time.perf_counter()-stage_start
        peak=torch.cuda.max_memory_allocated()/1024**2
        # Capture update memory before evaluation's temporary semantic one-hot buffers.
        anchors=state['anchor_ids'].cpu().numpy();gaussian_ids=ids[:,anchors]
        quality=evaluate(params,scene,end,args.masks,meta['targets'],gaussian_ids,out,stage,degree)
        row=dict(stage=stage,end=end,steps=args.steps,gaussians=len(anchors),cap=stage_cap,
            train_seconds=training_seconds,update_p50_ms=float(np.percentile(times,50)),update_p95_ms=float(np.percentile(times,95)),
            compute_p50_ms=float(np.percentile(compute_times,50)),gpu_update_peak_mib=peak,
            cpu_rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            controller_seconds_cumulative=controller_seconds,optimized_pixels=pixel_count,
            replay_pool=old_pool,teacher_max_frame=meta['max_teacher_frame'],**quality)
        history.append(row);(out/'metrics.json').write_text(json.dumps(history,indent=2))
        print('STAGE_COMPLETE',json.dumps({k:v for k,v in row.items() if k!='replay_pool'}),flush=True)
        # Full state is retained in Colab for every run; earlier work is untouched.
        ck=dict(params={k:v.detach().cpu() for k,v in params.items()},anchor_ids=state['anchor_ids'].cpu(),
            semantic_ids=gaussian_ids,args=vars(args),end=end,step=global_step)
        torch.save(ck,out/'checkpoint.pt')
    if args.seed==7 and args.factors in ('000','111'):
        export_map(params,out/'map.npz');np.save(out/'semantic-ids.npy',gaussian_ids)
    (out/'density-events.json').write_text(json.dumps(strategy.events))
    (out/'sampling-audit.json').write_text(json.dumps(samples,indent=2))
    (out/'run.json').write_text(json.dumps(dict(args=vars(args),dataset_kind=scene['kind'],units=scene['units'],
        seconds_including_evaluation_and_export=time.perf_counter()-start,device=torch.cuda.get_device_name(),
        optimizer_steps=global_step,cache_capacity=32,replay_capacity=48,
        qualifications=['Known input camera poses; no camera tracking','SAM pseudo region evaluation, not human part accuracy',
            'Teacher preprocessing is timed separately and shared; not an end-to-end real-time claim',
            'B uses a marginal coverage proxy, not an exact counterfactual query-loss oracle']),indent=2))
    print('RUN_COMPLETE',args.factors,args.seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--prefixes',required=True)
    p.add_argument('--masks',required=True);p.add_argument('--output',required=True);p.add_argument('--factors',default='111')
    p.add_argument('--seed',type=int,default=7);p.add_argument('--cap',type=int,default=300000)
    p.add_argument('--width',type=int,default=640);p.add_argument('--stages',type=int,default=6)
    p.add_argument('--steps',type=int,default=900);p.add_argument('--degree',type=int,default=2)
    run(p.parse_args())
