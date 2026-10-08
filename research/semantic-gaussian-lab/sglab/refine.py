"""Optional CUDA/gsplat local refinement with bounded old-keyframe replay.

This path is provided for Colab and has not been executed on the development Mac.
It optimizes RGB, depth, and semantic render losses for an isotropic Gaussian model.
"""
from collections import deque
import time
import numpy as np


class CudaRefiner:
    def __init__(self, steps=5, active_limit=6000, interval=5):
        import torch
        if not torch.cuda.is_available(): raise RuntimeError("--refine needs NVIDIA CUDA")
        from gsplat import rasterization
        self.torch=torch; self.rasterization=rasterization
        self.steps=steps; self.active_limit=active_limit; self.interval=interval
        self.recent=deque(maxlen=3); self.anchors=deque(maxlen=3)

    def observe(self, frame, frame_id):
        self.recent.append(frame)
        # Fixed early anchors preserve the old region and have bounded memory.
        if frame_id in (0,10,20): self.anchors.append(frame)

    def refine(self, mapping):
        torch=self.torch; cfg=mapping.config
        if mapping.frame_id%self.interval or not mapping.count: return None
        torch.cuda.synchronize(); begin=time.perf_counter(); torch.cuda.reset_peak_memory_stats()
        data=mapping.data[:mapping.count]
        active_mask=data["last"]>=mapping.frame_id-10
        if cfg.freeze_stable: active_mask &= data["hits"]<cfg.stable_hits
        active=np.flatnonzero(active_mask)
        active=active[-self.active_limit:]
        if not len(active): return None
        frozen=np.setdiff1d(np.arange(mapping.count),active,assume_unique=True)
        def tensor(x): return torch.tensor(np.asarray(x),device="cuda",dtype=torch.float32)
        # Keep optimized positions inside their original voxel so lookup stays valid.
        centers=tensor((np.floor(data["mean"][active]/cfg.voxel_size)+.5)*cfg.voxel_size)
        half=cfg.voxel_size*.4999
        position=torch.nn.Parameter(torch.atanh(((tensor(data["mean"][active])-centers)/half).clamp(-.999,.999)))
        colors=torch.nn.Parameter(torch.logit(tensor(data["color"][active]).clamp(.001,.999)))
        log_scale=torch.nn.Parameter(torch.log(tensor(data["sigma"][active])))
        opacity=torch.nn.Parameter(torch.logit(tensor(data["opacity"][active]).clamp(.001,.999)))
        # Evidence becomes trainable logits only for active splats; zero evidence stays diffuse.
        evidence=tensor(data["evidence"]+.02)
        semantic_logits=torch.nn.Parameter(torch.log(evidence[active]))
        fixed_mean=tensor(data["mean"][frozen]); fixed_color=tensor(data["color"][frozen])
        fixed_scale=tensor(data["sigma"][frozen]); fixed_opacity=tensor(data["opacity"][frozen])
        fixed_semantics=evidence[frozen]/evidence[frozen].sum(-1,keepdim=True)
        quats=torch.zeros((mapping.count,4),device="cuda"); quats[:,0]=1
        optimizer=torch.optim.Adam([position,colors,log_scale,opacity,semantic_logits],lr=.008)
        frames=list(self.recent)+list(self.anchors); losses=[]
        for step in range(self.steps):
            frame=frames[step%len(frames)]; h,w=frame.depth.shape
            means=torch.cat([centers+half*position.tanh(),fixed_mean])
            scales=torch.cat([log_scale.exp().clamp(.008,cfg.voxel_size),fixed_scale])[:,None].expand(-1,3).contiguous()
            rgb=torch.cat([colors.sigmoid(),fixed_color]); alpha=torch.cat([opacity.sigmoid(),fixed_opacity])
            semantics=torch.cat([semantic_logits.softmax(-1),fixed_semantics])
            args=dict(means=means,quats=quats,scales=scales,opacities=alpha,
                      viewmats=tensor(np.linalg.inv(frame.c2w))[None],Ks=tensor(frame.K)[None],width=w,height=h,packed=True)
            rendered,coverage,_=self.rasterization(colors=rgb,render_mode="RGB+ED",**args)
            pred=rendered[0]; target=tensor(frame.rgb/255); depth=tensor(frame.depth)
            valid=(depth>.15)&torch.isfinite(depth)&(coverage[0,...,0]>.1)
            # Only supported pixels: unobserved scene is not a black RGB target.
            if not valid.any(): continue
            loss=(pred[...,:3][valid]-target[valid]).abs().mean()+.15*(pred[...,3][valid]-depth[valid]).abs().mean()
            known=valid&(tensor(frame.semantic)>0)&(tensor(frame.confidence)>=.5)
            if known.any():
                rendered_sem,_,_=self.rasterization(colors=semantics,render_mode="RGB",**args)
                probability=rendered_sem[0]/coverage[0].clamp_min(1e-5)
                classes=torch.tensor(frame.semantic,device="cuda",dtype=torch.long)
                selected=probability[known].gather(1,classes[known].unsqueeze(1)).clamp_min(1e-6)
                loss=loss-.03*selected.log().mean()
            optimizer.zero_grad(); loss.backward(); optimizer.step(); losses.append(float(loss.detach()))
        with torch.no_grad():
            data["mean"][active]=(centers+half*position.tanh()).cpu().numpy()
            data["color"][active]=colors.sigmoid().cpu().numpy()
            data["sigma"][active]=log_scale.exp().clamp(.008,cfg.voxel_size).cpu().numpy()
            data["opacity"][active]=opacity.sigmoid().cpu().numpy()
            data["evidence"][active]=(semantic_logits.softmax(-1)*evidence[active].sum(-1,keepdim=True)).cpu().numpy()
        torch.cuda.synchronize()
        return {"refine_ms":round((time.perf_counter()-begin)*1000,3),"refine_steps":len(losses),
                "refine_loss":losses[-1] if losses else None,"active_gaussians":len(active),
                "gpu_allocated_mib":round(torch.cuda.memory_allocated()/1024**2,2),
                "gpu_peak_mib":round(torch.cuda.max_memory_allocated()/1024**2,2)}
