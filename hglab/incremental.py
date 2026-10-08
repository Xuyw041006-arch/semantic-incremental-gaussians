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
from .policies import semantic_importance,sampling_weights,coverage_replay,choose_crop,replay_probability
from .semantic_training import SemanticSupervisor,render_features
from .refinement import RefinementWindow,add_stability_arguments,validate_stability


class HierarchicalBudget(DefaultStrategy):
    def __init__(self,cap,enabled=False,**kw):
        super().__init__(**kw);self.cap=cap;self.enabled=enabled
        self.labels=None;self.confidence=None;self.events=[];self.strength=.1
        self.task_enabled=True;self.stage=0;self.stage_start=0;self.batch_pruned=False
        self.background_prune_fraction=.03;self.boundary_boost=.75;self.object_boost=.5
        self.stability_mode='mature';self.window=None
        self.growth_fraction=.1;self.min_child_age=200;self.min_fit_observations=12

    def begin_stage(self,stage,step,state=None,steps=2400,warmup=200,settle=800):
        self.stage=stage;self.stage_start=step;self.batch_pruned=False
        self.window=RefinementWindow(steps,warmup,settle,self.refine_every)
        if self.stability_mode!='legacy':
            if state is not None:
                for key in ('grad2d','count','radii'):
                    if isinstance(state.get(key),torch.Tensor):state[key].zero_()
            self.events.append(dict(stage=stage,step=step,event='batch_start',gradient_statistics_reset=True,
                warmup_steps=self.window.warmup_end,settle_steps=steps-self.window.settle_start))

    @torch.no_grad()
    def prepare_fit_state(self,params,state,step,admitted=0):
        if self.stability_mode!='mature':return
        n=len(params['means']);device=params['means'].device
        for key,fill in [('fit_observations',0.),('fit_view_a',-1),('fit_view_b',-1),('last_refined',-1000000)]:
            if key not in state:
                state[key]=torch.full((n,),fill,device=device,dtype=torch.float32 if key=='fit_observations' else torch.int64)
            if admitted:state[key][-admitted:]=fill

    @torch.no_grad()
    def _record_fit_observations(self,params,state,info):
        # A projected point counts only when RGB fitting supplied a nonzero gradient.
        # Two distinct frame IDs are required, without modulo/hash collisions.
        gid=info['gaussian_ids'];grad=info[self.key_for_gradient].absgrad if self.absgrad else info[self.key_for_gradient].grad
        good=torch.isfinite(grad).all(-1)&(grad.norm(dim=-1)>1e-10)
        gid=gid[good];gid=gid[params['opacities'][gid].sigmoid()>=self.prune_opa]
        if not len(gid):return
        state['fit_observations'][gid]+=1
        frame=int(info['frame_index']);a=state['fit_view_a'][gid]
        state['fit_view_a'][gid[a<0]]=frame
        second=(a>=0)&(a!=frame)&(state['fit_view_b'][gid]<0)
        state['fit_view_b'][gid[second]]=frame

    @torch.no_grad()
    def step_post_backward(self,params,optimizers,state,step,info,packed=False):
        if self.stability_mode=='legacy':
            return super().step_post_backward(params,optimizers,state,step,info,packed)
        if self.stability_mode=='mature':self._record_fit_observations(params,state,info)
        local=step-self.stage_start
        if self.window.phase(local)=='settle':return
        self._update_state(params,state,info,packed)
        if self.window.refine(local):
            self._grow_gs(params,optimizers,state,step)
            self._prune_gs(params,optimizers,state,step)
            state['grad2d'].zero_();state['count'].zero_()
            if isinstance(state.get('radii'),torch.Tensor):state['radii'].zero_()

    @torch.no_grad()
    def _reset_children(self,state,start,step):
        state['born_step'][start:]=step;state['observations'][start:]=0
        if self.stability_mode=='mature':
            state['fit_observations'][start:]=0
            state['fit_view_a'][start:]=-1;state['fit_view_b'][start:]=-1
            state['last_refined'][start:]=step

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
        factor=torch.ones_like(score)
        task=self.enabled and self.task_enabled
        if task:
            important=(state['semantic_priority']-1).clamp(0,2)/2
            boundary=state['boundary_score'].clamp(0,1)
            factor+=self.object_boost*important+self.boundary_boost*important*boundary
        eligible=(score*factor>self.grow_grad2d)&(state['count']>=2)
        if self.stability_mode=='mature':
            eligible&=(state['count']>=4)&(state['fit_observations']>=self.min_fit_observations)
            eligible&=(state['fit_view_b']>=0)&(step-state['born_step']>=self.min_child_age)
            eligible&=step-state['last_refined']>=self.min_child_age
        candidates=torch.where(eligible)[0]
        if not len(candidates):return 0,0
        rank=score[candidates]*factor[candidates]
        if self.enabled:rank*=1+self.strength*self.importance(state)[candidates].clamp_max(4)
        # Preserve opportunities for ordinary surfaces as well as task objects.
        fraction=self.growth_fraction if self.stability_mode=='mature' else .5
        budget=min(room,max(2048,int(len(score)*fraction)))
        general=min(int(budget*.3),len(candidates))
        ordinary=candidates[torch.argsort(score[candidates],descending=True)[:general]]
        chosen=torch.zeros_like(score,dtype=torch.bool);chosen[ordinary]=True
        remaining=candidates[~chosen[candidates]]
        selected_more=remaining[torch.argsort(rank[~chosen[candidates]],descending=True)[:budget-len(ordinary)]]
        chosen[selected_more]=True
        small=params['scales'].exp().max(-1).values<=self.grow_scale3d*state['scene_scale']
        if task:
            boundary_split=(state['semantic_priority']>=2)&(state['boundary_score']>=.5)&(params['scales'].exp().max(-1).values>.003)
            if self.stability_mode=='mature':boundary_split&=(score>self.grow_grad2d)&(state['semantic_conf']>=.8)
            small&=~boundary_split
        dup=chosen&small;spl=chosen&~small;nd=int(dup.sum());ns=int(spl.sum())
        important_count=int((chosen&(state['semantic_priority']>=2)).sum()) if task else 0
        edge_count=int((chosen&(state['semantic_priority']>=2)&(state['boundary_score']>=.5)).sum()) if task else 0
        if self.stability_mode=='mature':state['last_refined'][chosen]=step
        if nd:
            duplicate(params,optimizers,state,dup)
            self._reset_children(state,-nd,step)
        if ns:
            spl=torch.cat([spl,torch.zeros(nd,device=spl.device,dtype=torch.bool)])
            split(params,optimizers,state,spl,revised_opacity=self.revised_opacity)
        # New children inherit semantics, but need observations before budget pruning.
        fresh=2*ns if ns else nd
        if fresh:
            self._reset_children(state,-fresh,step)
        self.events.append(dict(stage=self.stage,step=step,event='growth',clone=nd,split=ns,
            important=important_count,important_boundary=edge_count,semantic=self.enabled,task=task,
            stability_mode=self.stability_mode,eligible=len(candidates),growth_budget=budget))
        return nd,ns

    @torch.no_grad()
    def _prune_gs(self,params,optimizers,state,step):
        opa=params['opacities'].sigmoid();scale=params['scales'].exp().max(-1).values
        invalid=(opa<self.prune_opa)|(scale>.3)|~torch.isfinite(params['means']).all(-1)
        n=len(opa);mask=invalid.clone();reallocated=0;protected=0
        task=self.enabled and self.task_enabled
        if task and not self.batch_pruned and step>=self.stage_start+300:
            background=(state['background_score']>=.75)&(state['semantic_priority']<2)&~invalid
            mature=(state['observations']>=3)&(step-state['born_step']>=200)
            # A measured contribution proxy, not an exact removal-loss estimate.
            rank=state['ema_contribution']*(.25+state['ema_error'].clamp(0,.3)/.05)
            keep=torch.zeros(n,device=opa.device,dtype=torch.bool)
            # Protect one strong background representative in every occupied 3D cell.
            bg=torch.where(background)[0]
            if len(bg):
                cells=torch.floor(params['means'][bg]/.035).long()
                _,inverse=torch.unique(cells,dim=0,return_inverse=True)
                maxima=torch.full((int(inverse.max())+1,),-1.,device=opa.device)
                maxima.scatter_reduce_(0,inverse,rank[bg],reduce='amax',include_self=True)
                keep[bg]=rank[bg]>=maxima[inverse]-1e-10
            protected=int(keep.sum())
            eligible=torch.where(background&mature&~keep&(state['ema_error']<.08))[0]
            # Only prune the low-contribution tail; a quota is a maximum, never an obligation.
            if len(eligible):
                threshold=torch.quantile(rank[background],.35)
                eligible=eligible[rank[eligible]<=threshold]
            count=min(int(int(background.sum())*self.background_prune_fraction),len(eligible))
            selected=eligible[torch.argsort(rank[eligible])[:count]];mask[selected]=True;reallocated=count
            self.batch_pruned=count>0
            self.events.append(dict(stage=self.stage,step=step,event='batch_background_prune',
                background_candidates=int(background.sum()),eligible=len(eligible),removed=count,
                fraction_limit=self.background_prune_fraction,spatially_protected=protected))
        elif not task and step%300==0 and n>=self.cap*.95:
            contribution=opa*(scale/scale.median().clamp_min(1e-6)).clamp(.1,10.)
            rank=contribution/contribution.mean().clamp_min(1e-8)
            if self.enabled:rank+=self.strength*self.importance(state).clamp_max(4)
            eligible=torch.where(~invalid&torch.isfinite(rank))[0]
            count=min(int(n*.02),len(eligible));selected=eligible[torch.argsort(rank[eligible])[:count]]
            mask[selected]=True;reallocated=count
        count=int(mask.sum())
        if count:remove(params,optimizers,state,mask)
        self.events.append(dict(stage=self.stage,step=step,event='prune',removed=count,reallocated=reallocated,protected=protected))
        return count


def read_prefix(folder,stage,scene):
    folder=Path(folder);meta=json.loads((folder/f'prefix-{stage}.json').read_text())
    raw=np.load(folder/f'prefix-{stage}.npz');ids=raw['ids'];confidence=raw['confidence'].astype(np.float32)
    assert meta['max_teacher_frame']<meta['end']
    assert all(not scene['images'][i]['test'] for i in meta['consumed_teacher_frames'])
    assert not np.any(ids[:,scene['release']>=meta['end']])
    return meta,ids,confidence,dict(raw)


def run(args):
    from .evaluation import evaluate
    validate_stability(args)
    torch.set_num_threads(2);torch.manual_seed(args.seed);np.random.seed(args.seed);random.seed(args.seed)
    rng=np.random.default_rng(args.seed);flags=tuple(c=='1' for c in args.factors)
    assert len(flags)==3;S,B,R=flags
    scene=load_scene(args.data,args.width);out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    params=None;opts={};state=None;released=np.zeros(len(scene['points']),bool);replay=[];seen=[]
    cache=OrderedDict();errors={};history=[];samples=[];global_step=0;controller_seconds=0.
    strategy=HierarchicalBudget(args.cap,enabled=B,refine_start_iter=50,refine_stop_iter=args.steps*args.stages-100,
        refine_every=100,reset_every=1000000000,grow_grad2d=.00025,absgrad=True,prune_scale3d=.3,verbose=False)
    strategy.strength=args.semantic_strength
    strategy.task_enabled=bool(args.semantic_densify)
    strategy.background_prune_fraction=args.background_prune_fraction
    strategy.boundary_boost=args.boundary_boost;strategy.object_boost=args.object_boost
    strategy.stability_mode=args.stability_mode;strategy.growth_fraction=args.growth_fraction
    strategy.min_child_age=args.min_child_age;strategy.min_fit_observations=args.min_fit_observations
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
        meta,ids,confidence,raw=read_prefix(args.prefixes,stage,scene);end=meta['end'];prev=round(total*(stage-1)/args.stages)
        semantic=SemanticSupervisor(args.prefixes,meta,raw,args.feature_dim)
        priorities=meta.get('priorities',{}) if args.semantic_densify else None
        strategy.begin_stage(stage,global_step,state,args.steps,args.batch_warmup,args.batch_settle)
        current=[i for i in range(prev,end) if not scene['images'][i]['test']]
        coverage={int(k):v for k,v in meta['coverage'].items()}
        if R:old_pool=coverage_replay(seen,coverage,48,positions,priorities)
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
                weights=1.+args.semantic_strength*np.minimum(importance,4)
                if args.semantic_densify:weights*=1+.5*np.maximum(raw['priority'][available].astype(float)-1,0)
                weights/=weights.sum()
                point_ids=rng.choice(available,count,replace=False,p=weights)
            else:point_ids=rng.choice(available,count,replace=False)
            values=initial_values(scene['points'][point_ids],scene['colors'][point_ids],args.degree)
            values['features']=semantic.initial_features(point_ids)
            released[point_ids]=True
            if params is None:
                params=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.as_tensor(v,device='cuda')) for k,v in values.items()})
                rates=dict(means=.00016,scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.000125,features=.002)
                opts={k:torch.optim.Adam([v],lr=rates[k],eps=1e-15) for k,v in params.items()}
                state=strategy.initialize_state(scene_scale=1.);state['anchor_ids']=torch.as_tensor(point_ids,device='cuda',dtype=torch.int64)
                strategy.check_sanity(params,opts)
            else:
                anchors=state['anchor_ids'];append_points(params,opts,state,values)
                state['anchor_ids']=torch.cat([anchors,torch.as_tensor(point_ids,device='cuda',dtype=torch.int64)])
        assert params is not None
        semantic.refresh(state,global_step)
        if count:state['born_step'][-count:]=global_step
        strategy.prepare_fit_state(params,state,global_step,count)
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
        replay_prob=replay_probability(stage,args.replay_max) if R else .3
        replay_flags=decisions.random(args.steps)<replay_prob;crop_flags=decisions.random(args.steps)<.5
        settle_order=[];warmup_order=[];phase_counts={};settle_counts={i:0 for i in seen}
        for local in range(args.steps):
            torch.cuda.synchronize();step_start=time.perf_counter();tick=time.perf_counter()
            phase=strategy.window.phase(local) if args.stability_mode!='legacy' else 'legacy'
            phase_counts[phase]=phase_counts.get(phase,0)+1
            pool=old_pool if old_pool and replay_flags[local] else current
            if phase=='settle':
                if not settle_order:settle_order=list(rng.permutation(seen))
                index=int(settle_order.pop());settle_counts[index]+=1
            elif phase=='warmup' and pool is current:
                if not warmup_order:warmup_order=list(rng.permutation(current))
                index=int(warmup_order.pop())
            elif S:
                weights=sampling_weights(pool,coverage,meta['births'],stage,errors,priorities)
                index=int(rng.choice(pool,p=weights))
            else:index=int(rng.choice(pool))
            im=scene['images'][index];target,depth=pixels(index)
            crop=None
            if crop_flags[local] and phase!='settle':
                im,crop=choose_crop(im,meta['boxes'].get(str(index),[]),rng,semantic=S,priorities=priorities)
                x,y,w,h=crop;target=target[y:y+h,x:x+w];depth=depth[y:y+h,x:x+w] if depth is not None else None
            pixel_count+=im['width']*im['height'];controller_seconds+=time.perf_counter()-tick
            degree=min(args.degree,global_step//1000)
            opts['means'].param_groups[0]['lr']=.00016*(.1**(global_step/(args.stages*args.steps)))
            torch.cuda.synchronize();compute_start=time.perf_counter()
            semantic_data=semantic.frame(index,crop) if local%args.semantic_every==0 else None
            pred,alpha,info=render(params,im,degree,mode='RGB+ED' if depth is not None or semantic_data is not None else 'RGB',grad=True)
            info['frame_index']=index
            strategy.step_pre_backward(params,opts,state,global_step,info)
            residual=(pred[0,...,:3]-target).abs().mean(-1);rgb_loss=residual.mean();loss=rgb_loss
            if semantic_data is not None:
                if args.semantic_densify:
                    weights=semantic.pixel_weights(semantic_data)
                    loss=(residual*weights).sum()/weights.sum()
                semantic.observe(params,state,info,pred,target,semantic_data,global_step)
                feature_target,valid=semantic.feature_target(semantic_data)
                if valid.any():
                    features,feature_alpha,_=render_features(params,im,degree)
                    valid=valid&(feature_alpha[0,...,0].detach()>.1)
                    if valid.any():
                        fitted=features[0]/feature_alpha[0].detach().clamp_min(.1)
                        loss=loss+args.feature_weight*(fitted[valid]-feature_target[valid]).square().mean()
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
                samples.append(dict(stage=stage,step=global_step,frame=index,crop=crop,old=index in old_pool,phase=phase))
                print('UPDATE',args.factors,args.seed,stage,local,len(params['means']),float(loss.detach()),flush=True)
        torch.cuda.synchronize();training_seconds=time.perf_counter()-stage_start
        peak=torch.cuda.max_memory_allocated()/1024**2
        # Capture update memory before evaluation's temporary semantic one-hot buffers.
        anchors=state['anchor_ids'].cpu().numpy();gaussian_ids=torch.stack([state['object_ids'],state['fine_ids']]).cpu().numpy()
        quality=evaluate(params,scene,end,args.masks,meta['targets'],gaussian_ids,out,stage,degree)
        row=dict(stage=stage,end=end,steps=args.steps,gaussians=len(anchors),cap=stage_cap,
            train_seconds=training_seconds,update_p50_ms=float(np.percentile(times,50)),update_p95_ms=float(np.percentile(times,95)),
            compute_p50_ms=float(np.percentile(compute_times,50)),gpu_update_peak_mib=peak,
            cpu_rss_peak_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
            controller_seconds_cumulative=controller_seconds,optimized_pixels=pixel_count,
            replay_pool=old_pool,replay_probability=replay_prob,teacher_max_frame=meta['max_teacher_frame'],
            stability_mode=args.stability_mode,phase_steps=phase_counts,settle_view_counts=settle_counts,**quality)
        row.update(important_gaussians=int((state['semantic_priority']>=2).sum()),
            important_boundary_gaussians=int(((state['semantic_priority']>=2)&(state['boundary_score']>=.5)).sum()),
            known_background_gaussians=int((state['background_score']>=.75).sum()),
            batch_background_pruned=sum(e['removed'] for e in strategy.events if e['stage']==stage and e['event']=='batch_background_prune'),
            important_densified=sum(e['important'] for e in strategy.events if e['stage']==stage and e['event']=='growth'),
            important_boundary_densified=sum(e['important_boundary'] for e in strategy.events if e['stage']==stage and e['event']=='growth'))
        history.append(row);(out/'metrics.json').write_text(json.dumps(history,indent=2))
        print('STAGE_COMPLETE',json.dumps({k:v for k,v in row.items() if k!='replay_pool'}),flush=True)
        # Full state is retained in Colab for every run; earlier work is untouched.
        ck=dict(params={k:v.detach().cpu() for k,v in params.items()},anchor_ids=state['anchor_ids'].cpu(),
            semantic_ids=gaussian_ids,args=vars(args),end=end,step=global_step,
            strategy_state={k:v.detach().cpu() if isinstance(v,torch.Tensor) else v for k,v in state.items()},
            optimizers={k:v.state_dict() for k,v in opts.items()})
        torch.save(ck,out/'checkpoint.pt')
        from .video_export import stage_preview
        stage_preview(params,scene,end,gaussian_ids,out,stage,degree)
    export_map(params,out/'map.npz');np.save(out/'semantic-ids.npy',gaussian_ids)
    np.savez_compressed(out/'semantic-state.npz',priority=state['semantic_priority'].cpu().numpy(),
        boundary=state['boundary_score'].cpu().numpy(),background=state['background_score'].cpu().numpy())
    (out/'density-events.json').write_text(json.dumps(strategy.events))
    (out/'sampling-audit.json').write_text(json.dumps(samples,indent=2))
    (out/'run.json').write_text(json.dumps(dict(args=vars(args),dataset_kind=scene['kind'],units=scene['units'],
        seconds_including_evaluation_and_export=time.perf_counter()-start,device=torch.cuda.get_device_name(),
        optimizer_steps=global_step,cache_capacity=32,replay_capacity=48,
        qualifications=['Input camera poses estimated offline by RGB SfM; not real-time camera tracking','SAM pseudo region evaluation, not human part accuracy',
            'Teacher preprocessing is timed separately and shared; not an end-to-end real-time claim',
            'B uses a marginal coverage proxy, not an exact counterfactual query-loss oracle']),indent=2))
    print('RUN_COMPLETE',args.factors,args.seed,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--prefixes',required=True)
    p.add_argument('--masks',required=True);p.add_argument('--output',required=True);p.add_argument('--factors',default='111')
    p.add_argument('--seed',type=int,default=7);p.add_argument('--cap',type=int,default=500000)
    p.add_argument('--width',type=int,default=640);p.add_argument('--stages',type=int,default=6)
    p.add_argument('--semantic-strength',type=float,default=.1);p.add_argument('--replay-max',type=float,default=.6)
    p.add_argument('--steps',type=int,default=600);p.add_argument('--degree',type=int,default=2)
    p.add_argument('--semantic-densify',type=int,choices=[0,1],default=1)
    p.add_argument('--background-prune-fraction',type=float,default=.03)
    p.add_argument('--boundary-boost',type=float,default=.75);p.add_argument('--object-boost',type=float,default=.5)
    p.add_argument('--feature-dim',type=int,default=16);p.add_argument('--feature-weight',type=float,default=.01)
    p.add_argument('--semantic-every',type=int,default=10)
    add_stability_arguments(p)
    run(p.parse_args())
