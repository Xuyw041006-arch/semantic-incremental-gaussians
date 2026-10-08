"""Official-source dependencies/data for a fresh NVIDIA Colab runtime.

Exact replay uses archived teacher/prefix caches. Regenerating them is a new run.
"""
import os,subprocess,sys,tarfile,urllib.request
from pathlib import Path

def main():
    import torch
    assert torch.cuda.is_available(),'Select a Colab NVIDIA GPU runtime.'
    os.environ['MAX_JOBS']='2';os.environ['SAM2_BUILD_CUDA']='0'
    subprocess.run([sys.executable,'-m','pip','install','-q','gsplat==1.5.3','scipy','scikit-image','matplotlib','open_clip_torch','remotezip','psutil'],check=True)
    subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation',
        'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'],check=True)
    weights=Path('/content/hierarchical-gaussian/checkpoints');weights.mkdir(parents=True,exist_ok=True)
    checkpoint=weights/'sam2.1_hiera_tiny.pt'
    if not checkpoint.exists():urllib.request.urlretrieve('https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',checkpoint)
    tum=Path('/content/tum-desk');tum.mkdir(exist_ok=True)
    if not (tum/'rgbd_dataset_freiburg1_desk/rgb.txt').exists():
        archive=tum/'desk.tgz'
        urllib.request.urlretrieve('https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz',archive)
        with tarfile.open(archive) as f:f.extractall(tum,filter='data')
    if not Path('/content/mip360/kitchen/sparse').exists():
        from remotezip import RemoteZip
        with RemoteZip('https://storage.googleapis.com/gresearch/refraw360/360_v2.zip') as z:
            files=[f for f in z.infolist() if f.filename.startswith('kitchen/') and ('/images_4/' in f.filename or '/sparse/' in f.filename)]
            for f in files:z.extract(f,'/content/mip360')
    print('Official dependencies/data ready; GPU',torch.cuda.get_device_name())

if __name__=='__main__':main()
