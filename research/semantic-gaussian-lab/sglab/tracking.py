"""Bounded, class-gated 3D overlap association; not a learned tracker."""
from dataclasses import dataclass
import numpy as np


@dataclass
class Track:
    id: int
    label: int
    center: np.ndarray
    voxels: set
    last_seen: int
    observations: int = 1


class InstanceTracker:
    def __init__(self, voxel=.12, max_tracks=256, max_voxels=2048):
        self.voxel=voxel; self.max_tracks=max_tracks; self.max_voxels=max_voxels
        self.tracks={}; self.next_id=1

    def associate(self, xyz, semantic, local, frame_id, confidence):
        global_ids=np.zeros(len(xyz),np.int32); used=set()
        for local_id in np.unique(local):
            if local_id<=0: continue
            mask=(local==local_id)&(confidence>=.5)&(semantic>0)
            if mask.sum()<3: continue
            label=int(np.bincount(semantic[mask]).argmax())
            points=xyz[mask]; center=points.mean(0)
            voxels=set(map(tuple,np.floor(points/self.voxel).astype(np.int32)))
            # Match existing tracks before updating any metadata; one-to-one per frame.
            candidates=[]
            for t in self.tracks.values():
                if t.label!=label or t.id in used: continue
                overlap=len(voxels&t.voxels)/max(1,min(len(voxels),len(t.voxels)))
                distance=float(np.linalg.norm(center-t.center))
                if overlap>=.12 or distance<.28:
                    candidates.append((overlap-.15*distance,t.id))
            if candidates:
                tid=max(candidates)[1]; t=self.tracks[tid]
                t.center=.8*t.center+.2*center; t.last_seen=frame_id; t.observations+=1
                t.voxels=set(sorted(t.voxels|voxels)[:self.max_voxels])
            else:
                # At the metadata cap keep old IDs; leave new unmatched objects unknown.
                if len(self.tracks)>=self.max_tracks: continue
                tid=self.next_id; self.next_id+=1
                self.tracks[tid]=Track(tid,label,center,set(sorted(voxels)[:self.max_voxels]),frame_id)
            used.add(tid); global_ids[mask]=tid
        return global_ids

    def summary(self):
        return [{"id":t.id,"label":t.label,"center":t.center.round(3).tolist(),"last_seen":t.last_seen,
                 "observations":t.observations} for t in self.tracks.values()]
