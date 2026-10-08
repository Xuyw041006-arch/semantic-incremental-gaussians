"""Deterministic ray-cast indoor RGB-D sequence with explicitly oracle labels/poses."""
import numpy as np
from .data import Frame

LABELS = [
    {"name": "unknown", "zh": "未知", "color": [113, 123, 137], "thing": False},
    {"name": "wall", "zh": "墙面", "color": [124, 146, 207], "thing": False},
    {"name": "floor", "zh": "地板", "color": [200, 173, 127], "thing": False},
    {"name": "table", "zh": "桌子", "color": [239, 169, 76], "thing": True},
    {"name": "chair", "zh": "椅子", "color": [102, 195, 167], "thing": True},
    {"name": "sofa", "zh": "沙发", "color": [181, 131, 207], "thing": True},
    {"name": "cabinet", "zh": "柜子", "color": [221, 124, 131], "thing": True},
    {"name": "plant", "zh": "绿植", "color": [137, 193, 87], "thing": True},
]


class SyntheticSequence:
    def __init__(self, count=100, width=256, height=192):
        self.count, self.width, self.height = count, width, height
        self.labels = LABELS
        self.name = "Procedural indoor loop"
        self.description = "合成室内序列 · 深度 / 位姿 / 分割均为真值"
        self.K = np.array([[210*width/256, 0, (width-1)/2], [0, 210*width/256, (height-1)/2], [0, 0, 1]], np.float32)
        # World axes: Y up. Each box has one semantic class and object identity.
        self.boxes = [
            ([-3,-.12,-3],[3,0,3],2,0,[.57,.44,.31]),
            ([-3,0,-3.12],[3,2.8,-3],1,0,[.76,.78,.76]),
            ([-3.12,0,-3],[-3,2.8,3],1,0,[.70,.74,.76]),
            ([3,0,-3],[3.12,2.8,3],1,0,[.75,.71,.67]),
            ([-3,0,3],[3,2.8,3.12],1,0,[.73,.77,.76]),
            ([-.85,.72,-.65],[.85,.84,.5],3,1,[.62,.36,.18]),
            ([-.77,0,-.56],[-.65,.72,-.44],3,1,[.25,.21,.17]),
            ([.65,0,.29],[.77,.72,.41],3,1,[.25,.21,.17]),
            ([-1.45,.42,-.35],[-1.0,.52,.15],4,2,[.25,.52,.49]),
            ([-1.5,.5,-.35],[-1.39,1.15,.15],4,2,[.25,.52,.49]),
            ([1.0,.42,-.35],[1.45,.52,.15],4,3,[.29,.48,.59]),
            ([1.39,.5,-.35],[1.5,1.15,.15],4,3,[.29,.48,.59]),
            ([-2.65,.1,-2.7],[-.8,.55,-1.85],5,4,[.62,.58,.49]),
            ([-2.65,.55,-2.7],[-.8,1.0,-2.5],5,4,[.62,.58,.49]),
            ([1.55,0,-2.8],[2.75,1.65,-2.35],6,5,[.55,.33,.20]),
            ([2.25,0,1.8],[2.6,.5,2.15],7,6,[.48,.3,.2]),
            ([2.12,.5,1.66],[2.72,1.35,2.28],7,6,[.27,.46,.24]),
        ]

    def __len__(self):
        return self.count

    def get(self, index):
        angle = 2*np.pi*index/max(self.count-1, 1)
        origin = np.array([2.1*np.sin(angle), 1.5+.1*np.sin(2*angle), 2.1*np.cos(angle)])
        forward = np.array([0,.55,-.25])-origin
        forward /= np.linalg.norm(forward)
        right = np.cross(forward, [0,1,0]); right /= np.linalg.norm(right)
        down = np.cross(forward, right)
        rotation = np.stack([right,down,forward], axis=1)
        c2w = np.eye(4, dtype=np.float32); c2w[:3,:3]=rotation; c2w[:3,3]=origin
        v,u=np.mgrid[:self.height,:self.width]
        rays=np.stack([(u-self.K[0,2])/self.K[0,0],(v-self.K[1,2])/self.K[1,1],np.ones_like(u)],axis=-1)@rotation.T
        depth=np.full(u.shape,np.inf); semantic=np.zeros(u.shape,np.int32); instance=np.zeros(u.shape,np.int32)
        rgb=np.zeros((*u.shape,3),np.float32)
        safe=np.where(np.abs(rays)<1e-9,1e-9,rays)
        for low,high,label,obj,color in self.boxes:
            lo=(np.array(low)-origin)/safe; hi=(np.array(high)-origin)/safe
            entry=np.minimum(lo,hi).max(-1); leave=np.maximum(lo,hi).min(-1)
            t=np.where(entry>0,entry,leave)
            hit=(leave>=np.maximum(entry,0))&(t>0)&(t<depth)
            points=origin+rays*np.where(np.isfinite(t),t,0)[...,None]
            texture=.90+.08*np.sin(points[...,0]*12)*np.sin(points[...,2]*12)
            if label==2:
                texture=.84+.14*((np.floor(points[...,0]*2)+np.floor(points[...,2]*2))%2)
            elif label==6:
                texture=.85+.12*np.cos(points[...,1]*17)
            depth[hit]=t[hit]; semantic[hit]=label
            # Deliberately change frame-local IDs; tracking must recover global IDs.
            instance[hit]=obj+100*(index%7) if obj else 0
            rgb[hit]=(np.array(color)*texture[...,None])[hit]
        depth[~np.isfinite(depth)]=0
        return Frame((np.clip(rgb,0,1)*255).astype(np.uint8),depth.astype(np.float32),self.K,c2w,semantic,instance,
                     np.ones(u.shape,np.float32),index/10,"synthetic_oracle")
