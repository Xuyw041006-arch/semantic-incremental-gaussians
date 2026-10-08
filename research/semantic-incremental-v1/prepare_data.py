"""Prepare two real scenes and immutable prefix semantics in isolated processes."""
from pathlib import Path
import subprocess,sys,json,time,traceback,os
ROOT=Path(__file__).resolve().parent
def run():
    os.chdir(ROOT);data=ROOT/'data';data.mkdir(exist_ok=True)
    sys.path.insert(0,'/content/semantic-gaussian-real')
    from sglab.tum import import_tum
    manifest=data/'tum640/manifest.json'
    if not manifest.exists():
        import_tum('/content/tum-desk/rgbd_dataset_freiburg1_desk',manifest.parent,max_frames=120,every=5,width=640)
    config=dict(width=640,cap=300000,stages=6,steps=600,seeds=[7,17,27],
        datasets=[dict(name='tum-desk',data=str(manifest)),dict(name='kitchen',data='/content/mip360/kitchen')],
        factors=['semantic_sampling','hierarchical_budget','coverage_replay'],
        replay_probability=.3,replay_capacity=48,rgb_cache_capacity=32,crop_width=384,crop_height=256,
        semantic_supervision='SAM2/CLIP pseudo labels, not human ground truth')
    (ROOT/'experiment-plan.json').write_text(json.dumps(config,indent=2))
    for ds in config['datasets']:
        folder=data/ds['name'];folder.mkdir(exist_ok=True)
        command=[sys.executable,'-m','hglab.teachers','--data',ds['data'],'--output',str(folder/'teachers')]
        if ds['name']=='kitchen':command+=['--reuse','/content/hierarchical-gaussian/semantics']
        subprocess.run(command,check=True)
        subprocess.run([sys.executable,'-m','hglab.online_semantics','--data',ds['data'],
            '--masks',str(folder/'teachers'),'--output',str(folder/'prefixes')],check=True)
    print('ALL_DATA_READY',flush=True)
if __name__=='__main__':
    try:run()
    except Exception:
        traceback.print_exc();sys.exit(1)
