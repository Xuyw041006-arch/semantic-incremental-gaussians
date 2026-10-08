"""Portable, visibility-aware old-view probe using projected Gaussian centres.

This is NOT a full Gaussian image-quality benchmark. Missing pixels are reported
as missing; the probe must not be presented as standard rendered-view PSNR.
"""
import numpy as np


def probe_old_view(mapping, frame):
    d=mapping.data[:mapping.count]; h,w=frame.depth.shape
    if not len(d): return {"coverage":0.0,"covered_rgb_psnr":None,"covered_depth_mae_m":None}
    w2c=np.linalg.inv(frame.c2w); camera=d["mean"]@w2c[:3,:3].T+w2c[:3,3]
    valid=camera[:,2]>.15; camera=camera[valid]; original=np.flatnonzero(valid)
    uv=camera@frame.K.T
    uv=np.rint(uv[:,:2]/uv[:,2,None]).astype(int)
    inside=(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h)
    uv=uv[inside]; camera=camera[inside]; original=original[inside]
    order=np.argsort(camera[:,2],kind="stable"); flat=uv[order,1]*w+uv[order,0]
    _,first=np.unique(flat,return_index=True); chosen=order[first]
    u,v=uv[chosen].T; observed=frame.depth[v,u]
    # Reject occluded/geometry-inconsistent samples against the reference depth.
    visible=(observed>.15)&np.isfinite(observed)&(np.abs(camera[chosen,2]-observed)<mapping.config.voxel_size*2)
    chosen=chosen[visible]; u,v=uv[chosen].T
    coverage=len(chosen)/max(1,int(((frame.depth>.15)&np.isfinite(frame.depth)).sum()))
    if not len(chosen): return {"coverage":0.0,"covered_rgb_psnr":None,"covered_depth_mae_m":None}
    mse=float(np.mean((d["color"][original[chosen]]-frame.rgb[v,u]/255)**2))
    return {"coverage":round(coverage,5),"covered_rgb_psnr":round(-10*np.log10(max(mse,1e-10)),3),
            "covered_depth_mae_m":round(float(np.abs(camera[chosen,2]-frame.depth[v,u]).mean()),5)}
