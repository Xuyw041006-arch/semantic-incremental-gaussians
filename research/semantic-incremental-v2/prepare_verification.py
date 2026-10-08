"""An untouched recording for final verification; do not tune against its outcomes."""
from pathlib import Path
import urllib.request,tarfile,sys,subprocess,argparse
ROOT=Path(__file__).resolve().parent
INPUT=Path('/content/semantic-incremental')
def main():
 p=argparse.ArgumentParser();p.add_argument('--download-only',action='store_true');args=p.parse_args()
 base=Path('/content/tum-desk2');base.mkdir(exist_ok=True);raw=base/'rgbd_dataset_freiburg1_desk2'
 if not (raw/'rgb.txt').exists():
  archive=base/'desk2.tgz'
  urllib.request.urlretrieve('https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk2.tgz',archive)
  with tarfile.open(archive) as z:z.extractall(base,filter='data')
 print('DESK2_PUBLIC_RECORDING_READY',flush=True)
 if args.download_only:return
 from sglab.tum import import_tum
 manifest=INPUT/'data/desk2-640/manifest.json'
 import_tum(raw,manifest.parent,max_frames=120,every=5,width=640)
 folder=INPUT/'data/tum-desk2'
 subprocess.run([sys.executable,'-m','hglab.teachers','--data',str(manifest),'--output',str(folder/'teachers')],cwd=ROOT,check=True)
 subprocess.run([sys.executable,'-m','hglab.online_semantics','--data',str(manifest),'--masks',str(folder/'teachers'),'--output',str(folder/'prefixes')],cwd=ROOT,check=True)
 print('UNTOUCHED_VERIFICATION_INPUTS_READY',flush=True)
if __name__=='__main__':main()
