"""Budgeted isotropic Gaussian fusion (not differentiable 3DGS training)."""
from dataclasses import dataclass, asdict
import sys
import time
import resource
from pathlib import Path
import numpy as np
from .data import backproject
from .tracking import InstanceTracker


@dataclass
class MapConfig:
    max_gaussians: int = 24000
    voxel_size: float = .065
    stride: int = 3
    max_samples: int = 4500
    target_update_ms: float = 45
    stable_hits: int = 5
    prune_after: int = 35
    dynamic_labels: tuple = ()
    freeze_stable: bool = True
    adaptive_budget: bool = True

    def __post_init__(self):
        if self.max_gaussians<1 or self.voxel_size<=0 or self.stride<1 or self.max_samples<1 or self.target_update_ms<=0 or self.stable_hits<1:
            raise ValueError("all budgets and sizes must be positive")


class GaussianMap:
    def __init__(self, labels, config=None):
        self.labels=labels; self.config=config or MapConfig(); c=len(labels)
        self.dtype=np.dtype([("mean","f4",3),("color","f4",3),("sigma","f4"),("opacity","f4"),
                             ("evidence","f4",c),("instance","i4"),("hits","u4"),("born","i4"),("last","i4")])
        self.data=np.zeros(self.config.max_gaussians,self.dtype)
        self.count=0; self.lookup={}; self.keys=[]; self.tracker=InstanceTracker()
        self.frame_id=0; self.sample_limit=self.config.max_samples; self.history=[]; self.trajectory=[]
        self.rng=np.random.default_rng(17)

    def _prune(self):
        d=self.data[:self.count]; cfg=self.config
        keep=~((d["hits"]<cfg.stable_hits)&(self.frame_id-d["last"]>cfg.prune_after))
        removed=self.count-int(keep.sum())
        if removed:
            self.data[:keep.sum()]=d[keep].copy()
            self.keys=[k for k,ok in zip(self.keys,keep) if ok]
            self.count=len(self.keys); self.lookup={k:i for i,k in enumerate(self.keys)}
        return removed

    def integrate(self, frame):
        start=time.perf_counter(); cfg=self.config; frame.validate(len(self.labels))
        xyz,color,sem,local,confidence,z=backproject(frame,cfg.stride)
        valid=~np.isin(sem,cfg.dynamic_labels)
        xyz,color,sem,local,confidence,z=[a[valid] for a in (xyz,color,sem,local,confidence,z)]
        total_valid=len(xyz)
        if total_valid>self.sample_limit:
            selection=self.rng.choice(total_valid,self.sample_limit,replace=False)
            xyz,color,sem,local,confidence,z=[a[selection] for a in (xyz,color,sem,local,confidence,z)]
        association_start=time.perf_counter()
        global_ids=self.tracker.associate(xyz,sem,local,self.frame_id,confidence)
        association_ms=(time.perf_counter()-association_start)*1000
        pruned=self._prune() if self.frame_id%10==0 else 0
        voxels=np.floor(xyz/cfg.voxel_size).astype(np.int32)
        # One observation per voxel per frame prevents false stability in a single frame.
        _,unique=np.unique(voxels,axis=0,return_index=True)
        unique=self.rng.permutation(unique)
        added=fused=frozen=rejected=processed=0
        changed=set(); fusion_start=time.perf_counter()
        deadline=start+cfg.target_update_ms/1000
        for j in unique:
            if cfg.adaptive_budget and processed%32==0 and time.perf_counter()>deadline: break
            processed+=1; key=tuple(voxels[j]); i=self.lookup.get(key)
            if i is None:
                if self.count>=cfg.max_gaussians: rejected+=1; continue
                i=self.count; self.count+=1; self.lookup[key]=i; self.keys.append(key)
                row=self.data[i]; row["mean"]=xyz[j]; row["color"]=color[j]
                row["sigma"]=np.clip(z[j]/frame.K[0,0]*cfg.stride*.7,.012,cfg.voxel_size*.8)
                row["opacity"]=.90; row["born"]=self.frame_id
                row["instance"]=global_ids[j]; added+=1
            else:
                row=self.data[i]
                # Do not mix two confident object identities at a voxel boundary.
                if row["instance"]>0 and global_ids[j]>0 and row["instance"]!=global_ids[j]:
                    rejected+=1; continue
                if cfg.freeze_stable and row["hits"]>=cfg.stable_hits:
                    row["last"]=self.frame_id; frozen+=1; continue
                weight=1/(min(int(row["hits"]),20)+1)
                row["mean"]=(1-weight)*row["mean"]+weight*xyz[j]
                row["color"]=(1-weight)*row["color"]+weight*color[j]
                if row["instance"]==0: row["instance"]=global_ids[j]
                fused+=1
            if sem[j]>0 and confidence[j]>=.5:
                row["evidence"][sem[j]]+=confidence[j]
            row["hits"]+=1; row["last"]=self.frame_id; changed.add(i)
        fusion_ms=(time.perf_counter()-fusion_start)*1000
        elapsed=(time.perf_counter()-start)*1000
        if cfg.adaptive_budget:
            ratio=np.clip(cfg.target_update_ms/max(elapsed,1),.65,1.15)
            self.sample_limit=int(np.clip(self.sample_limit*ratio, min(128,cfg.max_samples),cfg.max_samples))
        d=self.data[:self.count]
        rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/(1024**2 if sys.platform=="darwin" else 1024)
        stats={"frame":self.frame_id,"timestamp":frame.timestamp,"gaussians":self.count,
               "stable":int((d["hits"]>=cfg.stable_hits).sum()),"added":added,"fused":fused,"frozen":frozen,
               "pruned":pruned,"rejected":rejected,"valid_samples":total_valid,"processed":processed,
               "deferred":int(len(unique)-processed),"sample_limit":self.sample_limit,
               "association_ms":round(association_ms,3),"fusion_ms":round(fusion_ms,3),"update_ms":round(elapsed,3),
               "budget_overrun":elapsed>cfg.target_update_ms,"map_used_mib":round(self.count*self.dtype.itemsize/1024**2,3),
               "map_capacity_mib":round(self.data.nbytes/1024**2,3),"process_peak_rss_mib":round(rss,2),
               "gpu_allocated_mib":None,"segmentation_source":frame.segmentation_source}
        self.history.append(stats); self.trajectory.append(frame.c2w[:3,3].round(4).tolist()); self.frame_id+=1
        return stats, sorted(changed), bool(pruned)

    def packet(self, indices=None):
        ids=np.arange(self.count) if indices is None else np.array(indices,np.int32)
        d=self.data[ids]; evidence=d["evidence"]
        sums=evidence.sum(1); semantic=np.where(sums>0,evidence.argmax(1),0)
        conf=np.divide(evidence.max(1),sums,out=np.zeros(len(d),np.float32),where=sums>0)
        return {"ids":ids.tolist(),"xyz":d["mean"].round(4).ravel().tolist(),
                "rgb":(np.clip(d["color"],0,1)*255).astype(np.uint8).ravel().tolist(),
                "sigma":d["sigma"].round(4).tolist(),"semantic":semantic.tolist(),
                "opacity":d["opacity"].round(3).tolist(),
                "instance":d["instance"].tolist(),"confidence":conf.round(3).tolist()}

    def export(self, directory):
        import json
        path=Path(directory); path.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(path/"map.npz",gaussians=self.data[:self.count],trajectory=np.array(self.trajectory))
        (path/"metrics.json").write_text(json.dumps(self.history,indent=2))
        (path/"metadata.json").write_text(json.dumps({"labels":self.labels,"config":asdict(self.config),
            "backend":"cpu_isotropic_fusion","tracking":"provided_camera_poses","objects":self.tracker.summary()},indent=2,ensure_ascii=False))
        # Standard 3DGS fields: DC SH, log scales, logit opacity, wxyz quaternion.
        d=self.data[:self.count]; sem=self.packet()["semantic"]
        props=["x","y","z","nx","ny","nz","f_dc_0","f_dc_1","f_dc_2","opacity",
               "scale_0","scale_1","scale_2","rot_0","rot_1","rot_2","rot_3"]
        with (path/"map.ply").open("w") as f:
            f.write("ply\nformat ascii 1.0\ncomment sglab isotropic RGB-D Gaussian map; consult metadata.json for backend\n")
            f.write(f"element vertex {self.count}\n")
            for prop in props: f.write(f"property float {prop}\n")
            f.write("property int semantic_id\nproperty int instance_id\nend_header\n")
            for row,s in zip(d,sem):
                sh=(row["color"]-.5)/.28209479177387814
                opacity=float(np.clip(row["opacity"],1e-6,1-1e-6))
                values=[*row["mean"],0,0,0,*sh,np.log(opacity/(1-opacity)),*([np.log(row["sigma"])]*3),1,0,0,0]
                f.write(" ".join(f"{float(v):.6g}" for v in values)+f" {s} {int(row['instance'])}\n")
