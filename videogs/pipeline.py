"""Resumable stages keyed by input, configuration and source fingerprint."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
import time
import traceback
from .video import extract, write_json, sha256, encode_preview

ROOT=Path(__file__).resolve().parents[1]


def source_hash():
    h=hashlib.sha256()
    for folder in ('videogs','hglab'):
        for p in sorted(p for p in (ROOT/folder).iterdir() if p.suffix in ('.py','.html','.js')):
            h.update(p.relative_to(ROOT).as_posix().encode());h.update(p.read_bytes())
    return h.hexdigest()


def run_module(module,args,log):
    env=os.environ.copy();env['PYTHONUNBUFFERED']='1';env.setdefault('MAX_JOBS','2')
    with Path(log).open('w') as f:
        subprocess.run([sys.executable,'-m',module,*map(str,args)],cwd=ROOT,env=env,
                       stdout=f,stderr=subprocess.STDOUT,check=True)


def run(args):
    from .sfm import reconstruct,usable_stages
    from hglab.refinement import validate_stability
    validate_stability(args)
    video=Path(args.video).resolve();out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=True)
    config={k:v for k,v in vars(args).items() if k not in ('command','output','prepare_only')}
    config['video']=str(video);config['video_sha256']=sha256(video);config['source_sha256']=source_hash()
    if args.steps<1 or args.stages<1 or args.cap<1000 or args.width<64 or args.teacher_stride<1:
        raise ValueError('Invalid training steps/stages/cap/width/teacher stride.')
    if not 0<=args.background_prune_fraction<=.1 or args.semantic_every<1 or min(args.boundary_boost,args.object_boost,args.feature_weight)<0:
        raise ValueError('Background pruning must be between 0 and 0.1; semantic interval must be positive and boosts nonnegative.')
    fingerprint=hashlib.sha256(json.dumps(config,sort_keys=True).encode()).hexdigest()
    old=out/'pipeline.json'
    state=json.loads(old.read_text()) if old.exists() else dict(fingerprint=fingerprint,config=config,stages={})
    if state['fingerprint']!=fingerprint:
        raise ValueError('Output belongs to a different video/configuration/source version. Use a new --output directory; old results are preserved.')
    logs=out/'logs';logs.mkdir(exist_ok=True);started=time.perf_counter()
    state['state']='running';write_json(old,state)
    def stage(name,fn,required):
        if state['stages'].get(name,{}).get('state')=='complete' and all(Path(p).exists() for p in required):
            print('REUSE_STAGE',name,flush=True);return
        state.update(current_stage=name,state='running');state['stages'][name]=dict(state='running')
        write_json(old,state);begin=time.perf_counter();print('START_STAGE',name,flush=True)
        fn()
        if not all(Path(p).exists() for p in required):raise RuntimeError(f'{name} did not produce required artifacts')
        state['stages'][name]=dict(state='complete',seconds=time.perf_counter()-begin)
        write_json(old,state);print('COMPLETE_STAGE',name,flush=True)
    try:
        frames=out/'frames';sfm=out/'sfm';scene=sfm/'scene'
        stage('extract',lambda:extract(video,frames,args.fps,args.max_frames,args.extract_width,args.start,args.duration,
              args.blur_threshold,args.duplicate_threshold,args.rotate),[frames/'frames.json'])
        # pycolmap writes native stderr; the outer pipeline log also captures it.
        stage('sfm',lambda:reconstruct(frames,sfm,args.camera_model,args.camera_params,args.overlap,
              args.min_registration,args.threads),[scene/'sparse/cameras.bin',sfm/'sfm-quality.json',scene/'frames.json'])
        from hglab.readers import load_scene
        loaded=load_scene(scene,args.width);stages=usable_stages(loaded,args.stages)
        state['effective_stages']=stages;write_json(old,state)
        if args.prepare_only:
            state.update(state='ready_for_gpu',current_stage=None);write_json(old,state)
            print('VIDEO_PREPARATION_COMPLETE',scene,flush=True);return
        import torch
        if not torch.cuda.is_available():raise RuntimeError('Gaussian training and SAM need NVIDIA CUDA. Preparation finished; run the same command in Colab L4.')
        checkpoint=Path(args.checkpoint).resolve()
        if not checkpoint.is_file():raise FileNotFoundError(f'SAM2 checkpoint missing: {checkpoint}; run install_colab.py first.')
        teachers=out/'teachers';prefix=out/'prefixes';train=out/'training'
        stage('teachers',lambda:run_module('hglab.teachers',['--data',scene,'--output',teachers,'--width',args.width,
            '--checkpoint',checkpoint,'--train-stride',args.teacher_stride,'--test-stride',1],logs/'teachers.log'),
            [teachers/'index.json',teachers/'text-features.npy'])
        stage('semantics',lambda:run_module('hglab.online_semantics',['--data',scene,'--masks',teachers,
            '--output',prefix,'--width',args.width,'--stages',stages,'--important',args.important],logs/'semantics.log'),[prefix/f'prefix-{stages}.json'])
        stage('train',lambda:run_module('hglab.incremental',['--data',scene,'--masks',teachers,'--prefixes',prefix,
            '--output',train,'--width',args.width,'--stages',stages,'--steps',args.steps,'--cap',args.cap,
            '--factors',args.factors,'--seed',args.seed,'--semantic-densify',args.semantic_densify,
            '--background-prune-fraction',args.background_prune_fraction,'--boundary-boost',args.boundary_boost,
            '--object-boost',args.object_boost,'--feature-weight',args.feature_weight,'--semantic-every',args.semantic_every,
            '--stability-mode',args.stability_mode,'--batch-warmup',args.batch_warmup,'--batch-settle',args.batch_settle,
            '--growth-fraction',args.growth_fraction,'--min-child-age',args.min_child_age,
            '--min-fit-observations',args.min_fit_observations],logs/'training.log'),
            [train/'checkpoint.pt',train/'map.npz',train/'run.json',train/f'preview-{stages}.npz'])
        from .deliver import deliver
        stage('export',lambda:deliver(out,loaded,state),[out/'viewer.html',out/'gaussians.ply',out/'summary.json'])
        state.update(state='complete',current_stage=None,last_invocation_seconds=time.perf_counter()-started)
        write_json(old,state)
        print('VIDEO_TO_SEMANTIC_GAUSSIANS_COMPLETE',out,flush=True)
    except Exception as e:
        state.update(state='failed',error=str(e));write_json(old,state)
        (logs/'failure.txt').write_text(traceback.format_exc())
        current=state.get('current_stage')
        log=logs/({'train':'training','semantics':'semantics','teachers':'teachers'}.get(current,'')+'.log')
        if log.is_file():print(log.read_text(errors='replace')[-4000:],flush=True)
        raise


def doctor():
    packages={}
    for name in ['av','pycolmap','numpy','Pillow','scipy','torch','gsplat','open_clip_torch','sam2']:
        try:packages[name]=importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:packages[name]=None
    import shutil
    result=dict(python=sys.version,packages=packages,nvcc=shutil.which('nvcc'))
    try:
        import torch
        result['cuda']=torch.cuda.is_available()
        if result['cuda']:result['gpu']=torch.cuda.get_device_name()
    except ImportError:result['cuda']=False
    print(json.dumps(result,indent=2))
