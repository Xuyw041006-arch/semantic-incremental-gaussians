"""CUDA controller regression tests for batch statistics and child maturity."""
from types import SimpleNamespace
import numpy as np
import torch
from .base_train import initial_values,append_points
from .incremental import HierarchicalBudget


def run():
    torch.manual_seed(7);n=6;device='cuda'
    xyz=np.column_stack([np.linspace(-.1,.1,n),np.zeros(n),np.ones(n)]).astype(np.float32)
    values=initial_values(xyz,np.full((n,3),128,np.uint8),2);values['features']=np.zeros((n,16),np.float32)
    p=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.as_tensor(v,device=device)) for k,v in values.items()})
    opts={k:torch.optim.Adam([v],lr=.001) for k,v in p.items()}
    strategy=HierarchicalBudget(n+1,enabled=True,absgrad=True)
    strategy.labels=torch.ones((2,n),dtype=torch.int64,device=device);strategy.confidence=torch.ones((2,n),device=device)
    state=strategy.initialize_state()
    state.update(anchor_ids=torch.arange(n,device=device),grad2d=torch.ones(n,device=device),count=torch.ones(n,device=device),
        born_step=torch.zeros(n,device=device),observations=torch.zeros(n,device=device),
        semantic_priority=torch.full((n,),3.,device=device),semantic_conf=torch.ones(n,device=device),boundary_score=torch.ones(n,device=device))
    strategy.begin_stage(2,2400,state)
    assert state['grad2d'].sum()==0 and state['count'].sum()==0
    strategy.prepare_fit_state(p,state,2400)
    info=dict(gaussian_ids=torch.arange(n,device=device),means2d=SimpleNamespace(absgrad=torch.ones((n,2),device=device)),frame_index=0)
    for _ in range(12):strategy._record_fit_observations(p,state,info)
    assert (state['fit_observations']==12).all() and (state['fit_view_b']==-1).all()
    state['grad2d'].fill_(1);state['count'].fill_(10)
    assert strategy._grow_gs(p,opts,state,2600)==(0,0)  # One frame is insufficient.
    info['frame_index']=1;strategy._record_fit_observations(p,state,info)
    state['born_step'].fill_(2500)
    assert strategy._grow_gs(p,opts,state,2600)==(0,0)  # Children too young.
    state['born_step'].zero_()
    nd,ns=strategy._grow_gs(p,opts,state,2600)
    assert nd+ns==1 and len(p['means'])==n+1
    fresh=2*ns if ns else nd
    assert (state['born_step'][-fresh:]==2600).all() and (state['fit_observations'][-fresh:]==0).all()
    assert (state['fit_view_a'][-fresh:]==-1).all() and (state['fit_view_b'][-fresh:]==-1).all()
    strategy.cap=100
    state['grad2d'].zero_();state['grad2d'][state['last_refined']==2600]=1
    state['count'].fill_(10)
    assert strategy._grow_gs(p,opts,state,2601)==(0,0)  # Parent cooldown and new children.
    for k,v in state.items():
        if isinstance(v,torch.Tensor):assert len(v)==len(p['means']),(k,v.shape)
    added={k:v[:2] for k,v in values.items()}
    anchors=state['anchor_ids'];append_points(p,opts,state,added)
    state['anchor_ids']=torch.cat([anchors,torch.tensor([0,1],device=device)])
    strategy.prepare_fit_state(p,state,2700,2)
    assert (state['fit_view_a'][-2:]==-1).all() and (state['fit_view_b'][-2:]==-1).all()
    # During the tail, no topology/statistic mutation runs even with high residuals.
    events=len(strategy.events);info['gaussian_ids']=torch.arange(len(p['means']),device=device)
    info['means2d']=SimpleNamespace(absgrad=torch.ones((len(p['means']),2),device=device))
    strategy.step_post_backward(p,opts,state,4000,info,packed=True)
    assert len(strategy.events)==events
    print('CUDA_BATCH_RESET_DISTINCT_VIEWS_CHILD_MATURITY_COOLDOWN_APPEND_AND_SETTLE_PASS',flush=True)


if __name__=='__main__':run()
