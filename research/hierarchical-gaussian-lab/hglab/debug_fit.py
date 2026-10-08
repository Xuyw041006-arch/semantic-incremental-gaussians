"""Diagnostic: verify one-view fitting before claiming reconstruction quality."""
from pathlib import Path
import numpy as np
import torch
from .colmap import read_scene,image_rgb
from .train import initial_values,render,ssim_loss,BudgetStrategy
import sys,random
from PIL import Image

scene=read_scene('/content/mip360/kitchen',800)
im=scene['images'][1]; ids=np.where(scene['release']<46)[0]
vals=initial_values(scene['points'][ids],scene['colors'][ids],2)
params=torch.nn.ParameterDict({k:torch.nn.Parameter(torch.as_tensor(v,device='cuda')) for k,v in vals.items()})
lr=dict(means=.00016,scales=.005,quats=.001,opacities=.05,sh0=.0025,shN=.000125)
opts={k:torch.optim.Adam([v],lr=lr[k],eps=1e-15) for k,v in params.items()}
use_ssim='--ssim' in sys.argv; use_strategy='--strategy' in sys.argv
strategy=BudgetStrategy(100000,refine_start_iter=50,refine_stop_iter=1000,refine_every=100,absgrad=True,grow_grad2d=.00035,prune_scale3d=.3)
state=strategy.initialize_state(); targets={i:torch.as_tensor(image_rgb(scene['images'][i])/255.,device='cuda',dtype=torch.float32) for i in range(1,46) if i%8!=0}
random.seed(7)
for step in range(501):
    i=random.choice(list(targets)); im=scene['images'][i]; target=targets[i]
    pred,alpha,info=render(params,im,0,grad=use_strategy)
    if use_strategy: strategy.step_pre_backward(params,opts,state,step,info)
    loss=(pred[0]-target).abs().mean()
    if use_ssim: loss=.8*loss+.2*ssim_loss(pred[0],target)
    if step%20==0:
        print('FIT',step,float(loss.detach()),float(((pred[0]-target)**2).mean().detach()),
              'range',float(pred.min().detach()),float(pred.max().detach()),'alpha',float(alpha.mean().detach()),flush=True)
        panel=torch.cat([target,pred[0].clamp(0,1)],dim=1).detach().cpu().numpy()
        Image.fromarray((panel*255).astype('uint8')).save('/content/fit-debug.jpg')
    loss.backward()
    if step==0:
        print({k:(float(v.grad.abs().mean()),float(v.grad.abs().max())) for k,v in params.items()},flush=True)
    for opt in opts.values(): opt.step(); opt.zero_grad(set_to_none=True)
    if use_strategy: strategy.step_post_backward(params,opts,state,step,info,packed=True)
print('FIT_COMPLETE')
