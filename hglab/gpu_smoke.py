"""Verify real CUDA gradients, crop rays and semantic-state inheritance."""
import numpy as np
import torch
from gsplat.strategy.ops import duplicate,split,remove
from .base_train import initial_values,render,append_points
from .incremental import HierarchicalBudget
from .semantic_training import render_features

def run():
    torch.manual_seed(7);xyz=np.array([[-.1,0,1],[.1,0,1],[0,.1,1]],np.float32)
    values=initial_values(xyz,np.full((3,3),128,np.uint8),2)
    values['features']=np.full((3,16),.2,np.float32)
    params=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.as_tensor(v,device='cuda')) for k,v in values.items()})
    opts={k:torch.optim.Adam([v],lr=.001) for k,v in params.items()}
    strategy=HierarchicalBudget(100,enabled=True,absgrad=True);state=strategy.initialize_state(scene_scale=1.)
    strategy.stability_mode='legacy'
    state['anchor_ids']=torch.tensor([0,1,2],device='cuda')
    state['object_ids']=torch.tensor([11,12,13],device='cuda')
    state['boundary_score']=torch.tensor([1.,0.,0.],device='cuda')
    im=dict(c2w=np.eye(4),K=np.array([[60.,0,32],[0,60,24],[0,0,1]]),width=64,height=48)
    pred,_,info=render(params,im,2,grad=True);strategy.step_pre_backward(params,opts,state,1,info)
    pred.mean().backward();assert params['means'].grad is not None and torch.isfinite(params['means'].grad).all()
    feature,_,_=render(params,im,colors=params['features']);feature.square().mean().backward()
    assert params['features'].grad is not None and params['features'].grad.abs().sum()>0
    for opt in opts.values():opt.step();opt.zero_grad(set_to_none=True)
    feature,_,_=render_features(params,im);feature.square().mean().backward()
    assert params['features'].grad.abs().sum()>0 and params['means'].grad is None
    for opt in opts.values():opt.zero_grad(set_to_none=True)
    duplicate(params,opts,state,torch.tensor([True,False,False],device='cuda'))
    assert state['anchor_ids'].tolist()==[0,1,2,0]
    split(params,opts,state,torch.tensor([False,True,False,False],device='cuda'))
    assert state['anchor_ids'].tolist()==[0,2,0,1,1]
    remove(params,opts,state,torch.tensor([False,True,False,False,False],device='cuda'))
    assert state['anchor_ids'].tolist()==[0,0,1,1]
    assert state['object_ids'].tolist()==[11,11,12,12]
    for k,v in params.items():assert len(v)==len(state['anchor_ids']) and torch.isfinite(v).all()
    # Exercise the actual cap controller, not merely the underlying gsplat ops.
    n=len(params['means']);strategy.labels=torch.ones((2,3),device='cuda',dtype=torch.long)
    strategy.confidence=torch.ones((2,3),device='cuda');strategy.cap=n+2
    state.update(grad2d=torch.ones(n,device='cuda'),count=torch.full((n,),3.,device='cuda'),
        semantic_priority=torch.full((n,),3.,device='cuda'),born_step=torch.zeros(n,device='cuda'),
        observations=torch.ones(n,device='cuda'))
    strategy._grow_gs(params,opts,state,100)
    assert len(params['means'])==n+2 and strategy.events[-1]['important_boundary']>0
    for k,v in state.items():
        if isinstance(v,torch.Tensor):assert len(v)==len(params['means']),(k,v.shape)
    # Confirm limited background removal cannot consume unknown or important Gaussians.
    n=64;values=initial_values(np.random.default_rng(7).normal(0,.001,(n,3)),np.full((n,3),128),2)
    p=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.as_tensor(v,device='cuda')) for k,v in values.items()})
    o={k:torch.optim.Adam([v],lr=.001) for k,v in p.items()}
    s=strategy.initialize_state();s.update(anchor_ids=torch.arange(n,device='cuda'),
        background_score=torch.cat([torch.ones(32),torch.zeros(32)]).cuda(),
        semantic_priority=torch.cat([torch.zeros(32),torch.full((16,),3.),torch.ones(16)]).cuda(),
        observations=torch.full((n,),4.,device='cuda'),born_step=torch.zeros(n,device='cuda'),
        ema_error=torch.full((n,),.01,device='cuda'),ema_contribution=torch.linspace(.001,.1,n,device='cuda'))
    strategy.begin_stage(1,0);strategy.background_prune_fraction=.1
    strategy._prune_gs(p,o,s,400)
    survivors=set(s['anchor_ids'].cpu().tolist());assert set(range(32,64))<=survivors
    assert 0<64-len(survivors)<=3
    print('CUDA_GRADIENT_FEATURE_SEMANTIC_INHERITANCE_CAP_AND_BACKGROUND_PRUNE_PASS',flush=True)
if __name__=='__main__':run()
