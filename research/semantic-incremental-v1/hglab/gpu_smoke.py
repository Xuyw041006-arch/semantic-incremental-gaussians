"""Verify real CUDA gradients, crop rays and semantic-state inheritance."""
import numpy as np
import torch
from gsplat.strategy.ops import duplicate,split,remove
from .base_train import initial_values,render,append_points
from .incremental import HierarchicalBudget

def run():
    torch.manual_seed(7);xyz=np.array([[-.1,0,1],[.1,0,1],[0,.1,1]],np.float32)
    values=initial_values(xyz,np.full((3,3),128,np.uint8),2)
    params=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.as_tensor(v,device='cuda')) for k,v in values.items()})
    opts={k:torch.optim.Adam([v],lr=.001) for k,v in params.items()}
    strategy=HierarchicalBudget(100,enabled=True,absgrad=True);state=strategy.initialize_state(scene_scale=1.)
    state['anchor_ids']=torch.tensor([0,1,2],device='cuda')
    im=dict(c2w=np.eye(4),K=np.array([[60.,0,32],[0,60,24],[0,0,1]]),width=64,height=48)
    pred,_,info=render(params,im,2,grad=True);strategy.step_pre_backward(params,opts,state,1,info)
    pred.mean().backward();assert params['means'].grad is not None and torch.isfinite(params['means'].grad).all()
    for opt in opts.values():opt.step();opt.zero_grad(set_to_none=True)
    duplicate(params,opts,state,torch.tensor([True,False,False],device='cuda'))
    assert state['anchor_ids'].tolist()==[0,1,2,0]
    split(params,opts,state,torch.tensor([False,True,False,False],device='cuda'))
    assert state['anchor_ids'].tolist()==[0,2,0,1,1]
    remove(params,opts,state,torch.tensor([False,True,False,False,False],device='cuda'))
    assert state['anchor_ids'].tolist()==[0,0,1,1]
    for k,v in params.items():assert len(v)==len(state['anchor_ids']) and torch.isfinite(v).all()
    print('CUDA_GRADIENT_AND_SEMANTIC_INHERITANCE_PASS',flush=True)
if __name__=='__main__':run()
