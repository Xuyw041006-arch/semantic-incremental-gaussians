"""Minimal COLMAP binary reader; no CUDA dependency.

COLMAP poses and triangulated coordinates are offline input, NOT online SLAM.
Points are released only after >=2 training observations in the arrived prefix.
Held-out observations never determine point release or training targets.
"""
from pathlib import Path
import struct
import json
import numpy as np
from PIL import Image


def unpack(f, fmt):
    fmt = '<' + fmt
    return struct.unpack(fmt, f.read(struct.calcsize(fmt)))


def qrot(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([[1-2*y*y-2*z*z, 2*x*y-2*w*z, 2*x*z+2*w*y],
                     [2*x*y+2*w*z, 1-2*x*x-2*z*z, 2*y*z-2*w*x],
                     [2*x*z-2*w*y, 2*y*z+2*w*x, 1-2*x*x-2*y*y]])


def read_scene(root, width=800, test_every=8):
    root = Path(root)
    sparse = root/'sparse/0'
    if not sparse.exists(): sparse = root/'sparse'
    frame_meta = json.loads((root/'frames.json').read_text()) if (root/'frames.json').exists() else None
    by_name = {r['name']:r for r in frame_meta['frames']} if frame_meta else {}
    cameras = {}
    with (sparse/'cameras.bin').open('rb') as f:
        for _ in range(unpack(f, 'Q')[0]):
            cid, model, w, h = unpack(f, 'iiQQ')
            counts = {0:3, 1:4, 2:4, 3:5, 4:8, 5:8, 6:12, 7:5, 8:4, 9:5, 10:12}
            p = unpack(f, 'd'*counts[model])
            if model == 1: fx, fy, cx, cy = p
            elif model in (0, 2, 3): fx, cx, cy = p[:3]; fy = fx
            else: raise ValueError('Use undistorted PINHOLE images, camera model='+str(model))
            cameras[cid] = (w, h, np.array([[fx,0,cx],[0,fy,cy],[0,0,1.]]))
    images = []
    with (sparse/'images.bin').open('rb') as f:
        for _ in range(unpack(f, 'Q')[0]):
            values = unpack(f, 'i'+'d'*7+'i')
            iid, q, t, cid = values[0], values[1:5], values[5:8], values[8]
            name = bytearray()
            while (b := f.read(1)) != b'\0':
                if not b: raise ValueError('Truncated image name')
                name.extend(b)
            n = unpack(f, 'Q')[0]
            obs = np.frombuffer(f.read(n*24), dtype=[('xy','<f8',(2,)),('pid','<i8')])
            w2c = np.eye(4); w2c[:3,:3] = qrot(q); w2c[:3,3] = t
            images.append(dict(id=iid, name=name.decode(), c2w=np.linalg.inv(w2c),
                               camera=cid, observed=obs['pid'][obs['pid']>=0].copy()))
    images.sort(key=lambda im:im['name'])
    if frame_meta:
        order={r['name']:i for i,r in enumerate(frame_meta['frames'])}
        images.sort(key=lambda im:order[im['name']])
    for i,im in enumerate(images):
        im['test']=bool(by_name.get(im['name'],{}).get('test',i%test_every==0))
    id_to_index = {im['id']:i for i,im in enumerate(images)}
    xyz, rgb, release = [], [], []
    with (sparse/'points3D.bin').open('rb') as f:
        for _ in range(unpack(f, 'Q')[0]):
            v = unpack(f, 'QdddBBBd')
            n = unpack(f, 'Q')[0]
            track = np.array(unpack(f, 'ii'*n)).reshape(n,2)
            arrived = sorted(id_to_index[int(i)] for i in track[:,0]
                             if int(i) in id_to_index and not images[id_to_index[int(i)]]['test'])
            if len(arrived)<2: continue
            xyz.append(v[1:4]); rgb.append(v[4:7]); release.append(arrived[1])
    centers = np.array([im['c2w'][:3,3] for im in images])
    center = np.median(centers,axis=0)
    radius = np.max(np.linalg.norm(centers-center,axis=1))
    if not np.isfinite(radius) or radius<1e-8: raise ValueError('Degenerate camera translation; cannot normalize monocular scene')
    for i, im in enumerate(images):
        im['c2w'][:3,3] = (im['c2w'][:3,3]-center)/radius
        raw_w,raw_h,K = cameras[im.pop('camera')]
        path = root/'images_4'/im['name']
        if not path.exists(): path = root/'images'/im['name']
        with Image.open(path) as pic: src_w,src_h = pic.size
        w = min(width, src_w); h = round(src_h*w/src_w)
        K = K.copy(); K[0,:] *= w/raw_w; K[1,:] *= h/raw_h
        im.update(path=str(path), width=w,height=h,K=K,index=i,
                  timestamp=by_name.get(im['name'],{}).get('timestamp',float(i)))
    if len(xyz)<4:raise ValueError('Fewer than four sparse anchors with two training observations')
    return dict(images=images, points=(np.asarray(xyz)-center)/radius,
                colors=np.asarray(rgb,dtype=np.uint8), release=np.asarray(release),
                center=center,radius=radius,root=str(root),test_every=test_every)


def metadata(scene):
    return {k:scene[k].tolist() if isinstance(scene[k],np.ndarray) else scene[k]
            for k in ('center','radius','root','test_every')}


def image_rgb(im):
    with Image.open(im['path']) as pic:
        return np.asarray(pic.convert('RGB').resize((im['width'],im['height']),Image.Resampling.LANCZOS)).copy()
