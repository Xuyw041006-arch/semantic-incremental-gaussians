"""Full anisotropic splats: standard 32-byte record plus two int32 region IDs."""
from pathlib import Path
import numpy as np


def export_splats(path,values,ids):
    n=len(values['means']);dtype=np.dtype([('position','<f4',(3,)),('scale','<f4',(3,)),
        ('rgba','u1',(4,)),('rotation','u1',(4,)),('object','<i4'),('fine','<i4')])
    result=np.empty(n,dtype=dtype)
    result['position']=values['means'];result['scale']=values['scales']
    result['rgba'][:,:3]=np.rint((values['sh'][:,0].astype(np.float32)*.28209479177387814+.5).clip(0,1)*255).astype(np.uint8)
    result['rgba'][:,3]=np.rint(values['opacity'].clip(0,1)*255).astype(np.uint8)
    q=values['quats'].astype(np.float32);q/=np.linalg.norm(q,axis=1,keepdims=True).clip(1e-8)
    result['rotation']=np.rint(q*127+128).clip(0,255).astype(np.uint8)
    result['object']=ids[0];result['fine']=ids[1]
    Path(path).write_bytes(result.tobytes())
    xyz=values['means'];lo,hi=np.percentile(xyz,[5,95],axis=0)
    return dict(count=n,bytes=n*40,record_bytes=40,center=((lo+hi)/2).tolist(),
                radius=float(np.linalg.norm(hi-lo)/2),color='SH DC; CUDA comparison panels use full trained SH')
