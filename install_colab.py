"""Install official CUDA/teacher packages without downloading benchmark datasets."""
from pathlib import Path
import os,subprocess,sys,urllib.request

def main():
    import torch
    if not torch.cuda.is_available():raise RuntimeError('Select an NVIDIA Colab GPU (L4 recommended for this project).')
    os.environ['MAX_JOBS']='2';os.environ['SAM2_BUILD_CUDA']='0'
    subprocess.run([sys.executable,'-m','pip','install','-q','pycolmap==3.13.0','av','numpy','pillow',
        'scipy','scikit-image','gsplat==1.5.3','open_clip_torch','psutil'],check=True)
    subprocess.run([sys.executable,'-m','pip','install','-q','--no-build-isolation',
        'git+https://github.com/facebookresearch/sam2.git@2b90b9f5ceec907a1c18123530e92e794ad901a4'],check=True)
    folder=Path(__file__).resolve().parent/'checkpoints';folder.mkdir(exist_ok=True)
    checkpoint=folder/'sam2.1_hiera_tiny.pt'
    if not checkpoint.exists():urllib.request.urlretrieve(
        'https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt',checkpoint)
    print('VIDEO_PIPELINE_DEPENDENCIES_READY',torch.cuda.get_device_name())

if __name__=='__main__':main()
