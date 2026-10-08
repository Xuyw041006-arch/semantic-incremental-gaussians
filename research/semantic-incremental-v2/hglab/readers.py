"""Posed RGB-D stream or explicitly offline posed COLMAP input."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
from .colmap import read_scene,image_rgb


def depth_image(im):
    d=np.load(im['depth_path']).astype(np.float32)*im.get('depth_scale',1.)
    return np.asarray(Image.fromarray(d).resize((im['width'],im['height']),Image.Resampling.NEAREST)).copy()


def load_scene(data,width=640):
    data=Path(data)
    if data.suffix!='.json':
        scene=read_scene(data,width)
        scene['kind']='offline_COLMAP';scene['units']='normalized offline COLMAP units'
        scene['depth_tolerance']=.035;return scene
    manifest=json.loads(data.read_text());images=[]
    assert manifest['pose_convention']=='opencv_c2w_meters'
    for i,f in enumerate(manifest['frames']):
        path=data.parent/f['rgb']
        with Image.open(path) as pic:ow,oh=pic.size
        w=min(width,ow);h=round(oh*w/ow);K=np.asarray(f.get('K',manifest.get('K')),float)
        K[0]*=w/ow;K[1]*=h/oh
        images.append(dict(index=i,name=path.name,path=str(path),width=w,height=h,K=K,
            c2w=np.asarray(f['c2w'],float),test=i%6==3,depth_path=str(data.parent/f['depth']),
            depth_scale=manifest.get('depth_scale',1.),timestamp=f.get('timestamp',i)))
    # Causal voxel admission: first available observation fixes an anchor's position.
    points=[];colors=[];release=[];seen=set()
    for im in images:
        if im['test']:continue
        d=depth_image(im);rgb=image_rgb(im);y,x=np.mgrid[0:im['height']:4,0:im['width']:4]
        z=d[y,x];valid=np.isfinite(z)&(z>.2)&(z<5.)
        x=x[valid];y=y[valid];z=z[valid];K=im['K']
        xyz=np.column_stack(((x-K[0,2])*z/K[0,0],(y-K[1,2])*z/K[1,1],z))
        xyz=xyz@im['c2w'][:3,:3].T+im['c2w'][:3,3]
        keys=np.floor(xyz/.012).astype(np.int32)
        for j,key in enumerate(map(tuple,keys)):
            if key in seen:continue
            seen.add(key);points.append(xyz[j]);colors.append(rgb[y[j],x[j]]);release.append(im['index'])
    return dict(images=images,points=np.asarray(points,np.float32),colors=np.asarray(colors,np.uint8),
        release=np.asarray(release,np.int32),kind='RGBD_groundtruth_poses',units='meters',
        depth_tolerance=.045,root=str(data),center=np.zeros(3),radius=1.,test_every=6)


def project(xyz,im,depth=None,tolerance=.04):
    inv=np.linalg.inv(im['c2w']);p=xyz@inv[:3,:3].T+inv[:3,3];z=p[:,2]
    uv=p@im['K'].T
    safe=np.where(np.abs(z)>1e-6,z,1e-6)
    xy=np.rint(np.nan_to_num(uv[:,:2]/safe[:,None],nan=-1e6).clip(-1e6,1e6)).astype(np.int32)
    good=(z>.05)&(xy[:,0]>=0)&(xy[:,0]<im['width'])&(xy[:,1]>=0)&(xy[:,1]<im['height'])
    ids=np.flatnonzero(good);x=xy[ids,0];y=xy[ids,1]
    if depth is not None:
        target=depth[y,x];valid=(target>0)&(np.abs(z[ids]-target)<tolerance+.025*target)
        ids=ids[valid];x=x[valid];y=y[valid]
    return ids,y,x,z[ids]


def sparse_depth(xyz,im):
    from scipy.ndimage import minimum_filter
    ids,y,x,z=project(xyz,im);d=np.full((im['height'],im['width']),np.inf,np.float32)
    np.minimum.at(d,(y,x),z);d=minimum_filter(d,size=5);d[~np.isfinite(d)]=0
    return d
