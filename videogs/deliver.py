"""Portable 3DGS PLY, semantic IDs, camera track and measured playback."""
from pathlib import Path
import base64,json,shutil
import numpy as np
from .video import write_json,encode_preview
from .splats import export_splats


def export_ply(path,values,ids):
    means=values['means'];n=len(means)
    sh=values['sh'].astype(np.float32)
    fields=['x','y','z','nx','ny','nz']+[f'f_dc_{i}' for i in range(3)]+[
        f'f_rest_{i}' for i in range((sh.shape[1]-1)*3)]+['opacity']+[
        f'scale_{i}' for i in range(3)]+[f'rot_{i}' for i in range(4)]
    dtype=[(f,'<f4') for f in fields]+[('object_id','<i4'),('fine_id','<i4')]
    arr=np.zeros(n,dtype=dtype)
    opacity=values['opacity'].astype(float).clip(1e-6,1-1e-6)
    scale=np.log(values['scales'].astype(float).clip(1e-8))
    rotation=values['quats'].astype(float);rotation/=np.linalg.norm(rotation,axis=1,keepdims=True).clip(1e-8)
    columns=np.column_stack([means,np.zeros((n,3)),sh[:,0],sh[:,1:].transpose(0,2,1).reshape(n,-1),
                             np.log(opacity/(1-opacity)),scale,rotation])
    for j,f in enumerate(fields):arr[f]=columns[:,j]
    arr['object_id']=ids[0];arr['fine_id']=ids[1]
    header='ply\nformat binary_little_endian 1.0\ncomment Trained anisotropic Gaussian splats; normalized monocular scale\n'
    header+=f'element vertex {n}\n'+''.join(f'property float {f}\n' for f in fields)
    header+='property int object_id\nproperty int fine_id\nend_header\n'
    with Path(path).open('wb') as f:f.write(header.encode());f.write(arr.tobytes())


def b64(array,dtype):
    return base64.b64encode(np.asarray(array,dtype=dtype).tobytes()).decode()


def deliver(out,scene,state):
    out=Path(out);train=out/'training';metrics=json.loads((train/'metrics.json').read_text())
    values=dict(np.load(train/'map.npz'));ids=np.load(train/'semantic-ids.npy')
    export_ply(out/'gaussians.ply',values,ids)
    graph=json.loads((out/'prefixes'/f'prefix-{len(metrics)}.json').read_text())
    write_json(out/'objects.json',dict(target_scales=graph['targets'],nodes=graph['nodes'],
        qualification='SAM region instances and candidate fine regions, with CLIP vocabulary names; no human semantic/part ground truth'))
    np.savez_compressed(out/'gaussian-semantic-labels.npz',object_ids=ids[0],fine_ids=ids[1])
    track=[dict(index=im['index'],name=im['name'],timestamp=im['timestamp'],
                c2w=im['c2w'].tolist(),K=im['K'].tolist(),width=im['width'],height=im['height'],test=im['test']) for im in scene['images']]
    write_json(out/'camera-trajectory.json',dict(units='normalized monocular SfM scale',cameras=track))
    original=json.loads((out/'sfm/scene/frames.json').read_text())['frames']
    encode_preview(original,out/'sfm/scene/images',out/'input-preview.mp4')
    display=[]
    for row in metrics:
        p=dict(np.load(train/f'preview-{row["stage"]}.npz'))
        display.append(dict(stage=row['stage'],end=row['end'],points=b64(p['points'],'<f4'),
            colors=b64(p['colors'],'u1'),objects=b64(p['ids'][0],'<i4'),fine=b64(p['ids'][1],'<i4'),
            count=len(p['points']),metrics=row))
        splat=train/f'stage-{row["stage"]}.splat'
        if splat.exists():
            display[-1]['splat']=f'training/{splat.name}';display[-1]['splat_count']=splat.stat().st_size//40
    # Upgrade existing results without inventing earlier-stage full maps.
    if 'splat' not in display[-1]:
        export_splats(train/f'stage-{metrics[-1]["stage"]}.splat',values,ids)
        display[-1].update(splat=f'training/stage-{metrics[-1]["stage"]}.splat',splat_count=len(ids[0]))
    sfm=json.loads((out/'sfm/sfm-quality.json').read_text())
    summary=dict(input_video=state['config']['video'],input_video_sha256=state['config']['video_sha256'],
        source_sha256=state['config']['source_sha256'],sfm=sfm,gaussians=len(values['means']),
        stages=len(metrics),optimizer_steps=sum(r['steps'] for r in metrics),final=metrics[-1],
        stage_seconds={k:v.get('seconds') for k,v in state['stages'].items() if v['state']=='complete'},
        qualifications=['Monocular RGB only; no sensor depth or groundtruth poses',
            'Offline COLMAP uses the full video; incremental Gaussian fitting follows temporal prefixes',
            'RGB evaluation views are excluded from Gaussian fitting but included in SfM',
            'Interactive viewer renders full anisotropic splats with SH DC colors when stage .splat exists; old stages without saved full maps use labelled center LOD',
            'SAM/CLIP labels and fine regions are predictions, not verified named parts',
            'Playback is a recorded reconstruction process; not live real-time SLAM'])
    write_json(out/'summary.json',summary)
    payload=dict(stages=display,track=track,summary=summary,base_time=original[0]['timestamp'])
    template=(Path(__file__).with_name('viewer.html')).read_text()
    (out/'viewer.html').write_text(template.replace('__PAYLOAD__',json.dumps(payload,ensure_ascii=False).replace('</','<\\/')))
    print('VIDEO_VIEWER_AND_PLY_READY',len(values['means']),flush=True)
