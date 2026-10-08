"""Export real experiment measurements, CUDA renders and bounded browser LOD."""
from pathlib import Path
import json,shutil,hashlib,zipfile,subprocess,sys
import numpy as np
from PIL import Image,ImageDraw
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .colmap import read_scene,image_rgb
from .train import render


def load_params(path):
    ck=torch.load(path,map_location='cpu',weights_only=False)
    return ck,{k:v.cuda().requires_grad_(False) for k,v in ck['params'].items()}


def run(root='/content/hierarchical-gaussian'):
    torch.set_num_threads(2)
    root=Path(root); main=root/'results/kitchen-300k-l1'; field=root/'semantic-field-300k'; out=root/'deliverable'; out.mkdir(exist_ok=True)
    for folder in ('assets','results','semantic-field','source'): (out/folder).mkdir(exist_ok=True)
    scene=read_scene('/content/mip360/kitchen',800); ims=scene['images']; n=len(ims)
    sem=np.load(field/'hierarchy.npz'); labels=sem['ids']
    nodes=json.loads((field/'nodes.json').read_text())
    palettes=[np.random.default_rng(13).uniform(.15,.95,(max(1,int(a.max())+1),3)).astype('float32') for a in labels]
    for p in palettes: p[0]=.15
    full,params=load_params(main/'checkpoint.pt'); means=params['means'].cpu().numpy()
    frames=[]; footprints={}
    center=np.median(means,axis=0); sample=means[::max(1,len(means)//6000)]
    _,_,basis=np.linalg.svd(sample-center,full_matrices=False); basis=basis[:2].T
    projected=(sample-center)@basis; lo,hi=np.percentile(projected,[1,99],axis=0)
    def footprint_points(xyz):
        a=((xyz-center)@basis-lo)/(hi-lo).clip(1e-6)
        return np.column_stack([a[:,0]*230+13,(1-a[:,1])*230+13])
    for im in ims:
        frame=dict(index=im['index'],name=im['name'],test=im['test'],width=im['width'],height=im['height'],c2w=im['c2w'].tolist(),
                   K=im['K'].tolist(),stage=next(s for s in range(1,7) if im['index']<round(n*s/6)))
        rgb=image_rgb(im); Image.fromarray(rgb).resize((400,round(im['height']*.5))).save(out/'assets'/f'input-{im["index"]:03d}.jpg',quality=83)
        frames.append(frame)
    for stage in range(1,7):
        ck,p=load_params(main/f'checkpoint-{stage}.pt'); count=len(p['means'])
        idx=torch.linspace(0,count-1,min(60000,count),device='cuda').long()
        points=p['means'][idx].cpu().numpy(); scales=p['scales'][idx].exp().cpu().numpy()
        quats=p['quats'][idx].cpu().numpy(); colors=(p['sh0'][idx,0]*.28209479177387814+.5).clamp(0,1).cpu().numpy()
        opacity=p['opacities'][idx].sigmoid().cpu().numpy()[:,None]
        node_ids=labels[:,idx.cpu().numpy()].T if stage==6 else np.zeros((len(idx),3),np.int32)
        a=np.concatenate([points,scales,quats,colors,opacity,node_ids],axis=1).astype('<f4')
        a.tofile(out/'assets'/f'map-{stage}.bin')
        footprint=Image.new('RGB',(256,267),(18,26,35)); pen=ImageDraw.Draw(footprint)
        xy=footprint_points(points)
        for xy0,color in zip(xy,colors):
            if 0<=xy0[0]<256 and 0<=xy0[1]<267: pen.point(tuple(xy0),fill=tuple((color*180).astype(int)))
        footprints[stage]=footprint
        if stage==6:
            np.save(out/'assets/preview-indices.npy',idx.cpu().numpy())
            np.save(out/'semantic-field/preview-affinity.npy',np.load(field/'affinity-fp16.npy')[idx.cpu().numpy()])
        # Actual CUDA rendering of the map available at the end of this chunk.
        start=round(n*(stage-1)/6); end=round(n*stage/6)
        for im in ims[start:end]:
            with torch.no_grad(): pred,_,_=render(p,im,min(2,(ck['step']-1)//1000))
            img=(pred[0].clamp(0,1).cpu().numpy()*255).astype('uint8')
            Image.fromarray(img).resize((400,round(im['height']*.5))).save(out/'assets'/f'render-{im["index"]:03d}.jpg',quality=84)
        print('BUNDLE_STAGE',stage,count,flush=True)
        del p
    selected=[0,40,88,136,184,232,272]
    for i in selected:
        im=ims[i]; panels=[image_rgb(im)]; titles=['Real photo','Full 300k Gaussian render','Coarse regions','Object-scale regions','Fine regions / candidate parts']
        with torch.no_grad(): pred,_,_=render(params,im,2)
        panels.append((pred[0].clamp(0,1).cpu().numpy()*255).astype('uint8'))
        for level in range(3):
            with torch.no_grad(): pred,_,_=render(params,im,colors=torch.as_tensor(palettes[level][labels[level]],device='cuda'))
            pic=(pred[0].clamp(0,1).cpu().numpy()*255).astype('uint8'); panels.append(pic)
            Image.fromarray(pic).resize((400,round(im['height']*.5))).save(out/'assets'/f'level-{level}-{i:03d}.jpg',quality=85)
        panel=Image.fromarray(np.concatenate(panels,axis=1)); panel.thumbnail((2000,600))
        band=Image.new('RGB',(panel.width,panel.height+36),(18,24,33)); band.paste(panel,(0,36)); draw=ImageDraw.Draw(band)
        for j,t in enumerate(titles): draw.text((j*panel.width/5+8,10),t,fill=(230,239,246))
        band.save(out/'assets'/f'comparison-{i:03d}.jpg',quality=92)
    # Presentation-speed replay of real photographs, not original capture FPS.
    ffmpeg=shutil.which('ffmpeg')
    if ffmpeg:
        movie=root/'work/movie'; movie.mkdir(parents=True,exist_ok=True)
        history=json.loads((main/'metrics.json').read_text())
        for i,frame in enumerate(frames):
            stage=frame['stage']; s=history[stage-1]
            panel=Image.new('RGB',(1056,340),(12,18,26)); pen=ImageDraw.Draw(panel)
            panel.paste(Image.open(out/'assets'/f'input-{i:03d}.jpg'),(0,36))
            panel.paste(Image.open(out/'assets'/f'render-{i:03d}.jpg'),(400,36))
            fp=footprints[stage].copy(); d=ImageDraw.Draw(fp)
            traj=footprint_points(np.array([im['c2w'][:3,3] for im in ims[:i+1]]))
            if len(traj)>1: d.line([tuple(t) for t in traj],fill=(130,237,191),width=2)
            panel.paste(fp,(800,36))
            pen.text((12,10),'REAL PHOTO',fill=(220,235,243)); pen.text((412,10),'CUDA GAUSSIAN RENDER / END-OF-CHUNK MAP',fill=(220,235,243)); pen.text((810,10),'MAP + CAMERA PATH',fill=(220,235,243))
            pen.text((12,312),f'Photo {i+1}/{n} | batch {stage}/6 | {s["gaussians"]:,} Gaussians | tensor peak {s["gpu_peak_mib"]:.0f} MiB | presentation 15 FPS',fill=(161,218,191))
            panel.save(movie/f'{i:04d}.jpg',quality=88)
        pos=n
        for i in selected:
            panel=Image.new('RGB',(1056,340),(12,18,26)); p=Image.open(out/'assets'/f'comparison-{i:03d}.jpg'); p.thumbnail((1056,280)); panel.paste(p,(0,45))
            ImageDraw.Draw(panel).text((12,12),'FINAL MAP / MULTI-SCALE REGIONS AND CANDIDATE PARTS / SEMANTIC POST-PASS',fill=(165,235,202))
            for _ in range(15): panel.save(movie/f'{pos:04d}.jpg',quality=88); pos+=1
        subprocess.run([ffmpeg,'-nostdin','-y','-loglevel','error','-framerate','15','-i',str(movie/'%04d.jpg'),'-c:v','libx264','-threads','2','-pix_fmt','yuv420p','-crf','22',str(out/'real-kitchen-demo.mp4')],stdin=subprocess.DEVNULL,check=True)
    rows=[]; histories={}
    for name in ('kitchen-300k-l1','kitchen-600k-l1','kitchen-600k-no-replay'):
        src=root/'results'/name
        if not (src/'metadata.json').exists(): continue
        dest=out/'results'/name; dest.mkdir(exist_ok=True)
        for p in src.iterdir():
            if p.suffix in ('.json','.jpg'): shutil.copy2(p,dest/p.name)
        history=json.loads((src/'metrics.json').read_text()); histories[name]=history
        last=history[-1]
        rows.append(dict(variant=name,gaussians=last['gaussians'],psnr=last['psnr'],ssim=last['ssim'],
            old_psnr_first=history[0]['old_view_psnr'],old_psnr_final=last['old_view_psnr'],
            old_change_db=last['old_view_psnr']-history[0]['old_view_psnr'],
            training_seconds=sum(r['chunk_seconds'] for r in history),
            gpu_peak_mib=max(r['gpu_peak_mib'] for r in history),
            last_step_p50_ms=last['update_p50_ms'],last_step_p95_ms=last['update_p95_ms']))
    (out/'comparison.json').write_text(json.dumps(rows,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(11,7))
    for name,h in histories.items():
        x=[r['arrived'] for r in h]
        for ax,key in zip(axes.ravel(),['gaussians','psnr','old_view_psnr','gpu_peak_mib']): ax.plot(x,[r[key] for r in h],'-o',label=name)
    for ax,title in zip(axes.ravel(),['Gaussian count','Held-out PSNR (dB)','Fixed old-view PSNR (dB)','PyTorch peak allocated MiB']):
        ax.set_title(title); ax.set_xlabel('Arrived photographs'); ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'experiment-curves.png',dpi=180); plt.close(fig)
    for p in field.iterdir():
        if p.suffix in ('.json','.npy','.npz','.jpg'): shutil.copy2(p,out/'semantic-field'/p.name)
    for name in ('text-features.npy','vocabulary.json','clip-upgrade.json','index.json'):
        p=root/'semantics'/name
        if p.exists(): shutil.copy2(p,out/'semantic-field'/name)
    teachers=out/'teacher-masks'; teachers.mkdir(exist_ok=True)
    for p in (root/'semantics').glob('masks-*'):
        if p.suffix in ('.json','.npz'): shutil.copy2(p,teachers/p.name)
    raw=Path('/content/mip360/kitchen')
    (out/'dataset-provenance.json').write_text(json.dumps(dict(
        official_archive='https://storage.googleapis.com/gresearch/refraw360/360_v2.zip',
        files=[dict(path=str(p.relative_to(raw)),bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest())
               for p in sorted(raw.rglob('*')) if p.is_file()]),indent=2))
    shutil.copy2(main/'map.npz',out/'map.npz')
    shutil.copy2(root/'results/kitchen-600k-l1/map.npz',out/'alternative-600k-map.npz')
    shutil.copytree(root/'hglab',out/'source/hglab',dirs_exist_ok=True)
    semantic_metrics=json.loads((field/'metrics.json').read_text())
    metadata=dict(dataset='Mip-NeRF 360 kitchen',real_capture=True,photographs=n,
        training_photographs=sum(not im['test'] for im in ims),heldout_photographs=sum(im['test'] for im in ims),
        map_cap=300000,maximum_tested_map_cap=600000,display_lod_cap=60000,stages=histories.get('kitchen-300k-l1',[]),frames=frames,
        semantic=semantic_metrics['summary'],nodes=nodes,palettes=[p.tolist() for p in palettes],view_target=center.tolist(),
        semantic_qualification='Final-map semantic post-pass; historical stages show geometry only. Fine masks are candidate parts. No human semantic GT.',
        protocol='Offline COLMAP poses and triangulated coordinates; prefix-restricted RGB and point release; filename-ordered chunk replay, not original video timestamps.',
        device=torch.cuda.get_device_name())
    (out/'viewer.json').write_text(json.dumps(metadata))
    (out/'environment.json').write_text(json.dumps(dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,gpu=torch.cuda.get_device_name()),indent=2))
    (out/'pip-freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True))
    for src in ('kitchen-train-l1.log','kitchen-field.log','kitchen-masks.log','kitchen-300k-l1.log','kitchen-600k-no-replay.log','fit-ssim.log','fit-strategy.log'):
        p=Path('/content')/src
        if p.exists(): shutil.copy2(p,out/p.name)
    archive=Path('/content/hierarchical-kitchen-results.zip')
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for p in out.rglob('*'):
            if p.is_file() and '__pycache__' not in str(p): z.write(p,p.relative_to(out))
    print('BUNDLE_READY',archive,archive.stat().st_size,hashlib.sha256(archive.read_bytes()).hexdigest(),flush=True)


if __name__=='__main__': run()
