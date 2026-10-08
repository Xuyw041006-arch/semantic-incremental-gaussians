"""TUM RGB-D -> generic manifest. Ground-truth poses, not estimated SLAM."""
import json
from pathlib import Path
import numpy as np
from PIL import Image


def quaternion_matrix(q):
    q=np.asarray(q,dtype=float)
    norm=np.linalg.norm(q)
    if not np.isfinite(norm) or norm<1e-8: raise ValueError("invalid pose quaternion")
    x,y,z,w=q/norm
    return np.array([[1-2*(y*y+z*z),2*(x*y-z*w),2*(x*z+y*w)],
        [2*(x*y+z*w),1-2*(x*x+z*z),2*(y*z-x*w)],
        [2*(x*z-y*w),2*(y*z+x*w),1-2*(x*x+y*y)]])


def read_rows(path):
    rows=[]
    for line in Path(path).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            tokens=line.split(); rows.append((float(tokens[0]),tokens[1:]))
    return sorted(rows)


def nearest(rows,stamp,tolerance=.025):
    if not rows: return None
    stamps=np.array([r[0] for r in rows]); i=int(np.searchsorted(stamps,stamp))
    choices=[j for j in (i-1,i) if 0<=j<len(rows)]
    j=min(choices,key=lambda j:abs(stamps[j]-stamp))
    return rows[j] if abs(stamps[j]-stamp)<=tolerance else None


def import_tum(dataset, output, max_frames=100, every=3, width=320):
    root=Path(dataset).resolve(); output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=True)
    rgb=read_rows(root/"rgb.txt"); depth=read_rows(root/"depth.txt"); poses=read_rows(root/"groundtruth.txt")
    frames=[]; initial=None
    # TUM recommends ROS default intrinsics for pre-registered, unrectified RGB-D.
    base_K=np.array([[525.,0,319.5],[0,525.,239.5],[0,0,1]])
    for stamp,row in rgb[::every]:
        d=nearest(depth,stamp); p=nearest(poses,stamp)
        if d is None or p is None: continue
        rgb_path=(root/row[0]).resolve(); depth_path=(root/d[1][0]).resolve()
        if not rgb_path.is_relative_to(root) or not depth_path.is_relative_to(root): raise ValueError("TUM path outside dataset")
        image=Image.open(rgb_path).convert("RGB"); old_w,old_h=image.size
        height=round(old_h*width/old_w); scale_x=width/old_w; scale_y=height/old_h
        K=base_K.copy(); K[0,0]*=scale_x; K[1,1]*=scale_y
        K[0,2]=(K[0,2]+.5)*scale_x-.5; K[1,2]=(K[1,2]+.5)*scale_y-.5
        image=image.resize((width,height),Image.Resampling.LANCZOS)
        depth_image=Image.open(depth_path).resize((width,height),Image.Resampling.NEAREST)
        values=np.array(p[1],float); pose=np.eye(4); pose[:3,:3]=quaternion_matrix(values[3:7]); pose[:3,3]=values[:3]
        if initial is None: initial=np.linalg.inv(pose)
        # Translate/rotate the world to the first camera, then choose Y-up for the viewer.
        y_up=np.diag([1,-1,-1,1]); pose=y_up@initial@pose
        prefix=f"{len(frames):06}"; image.save(output/(prefix+".png"))
        np.save(output/(prefix+"_depth.npy"),np.asarray(depth_image,dtype=np.float32)/5000)
        frames.append({"rgb":prefix+".png","depth":prefix+"_depth.npy","c2w":pose.tolist(),
            "K":K.tolist(),"timestamp":stamp-rgb[0][0],"segmentation_source":"none"})
        if len(frames)>=max_frames: break
    if not frames: raise ValueError("no RGB/depth/pose associations within 25 ms")
    manifest={"name":root.name,"description":"TUM 真实 RGB-D · 使用 groundtruth 位姿（未做相机跟踪）",
        "pose_convention":"opencv_c2w_meters","depth_scale":1,"labels":[{"name":"unknown","zh":"未知","color":[113,123,137],"thing":False}],
        "frames":frames,"provenance":{"dataset":"TUM RGB-D","pose":"groundtruth nearest association <=25ms",
            "intrinsics":"ROS default 525/525/319.5/239.5, resized with half-pixel correction"}}
    path=output/"manifest.json"; path.write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
    return path
