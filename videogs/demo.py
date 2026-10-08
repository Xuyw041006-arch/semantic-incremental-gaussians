"""Public real RGB recording -> MP4, explicitly discarding depth and GT poses."""
from pathlib import Path
import tarfile
import urllib.request
import numpy as np
from PIL import Image
from .video import encode_preview,write_json,sha256

URL='https://webshare.cvg.cit.tum.de/g/rgbd/dataset/freiburg1/rgbd_dataset_freiburg1_desk.tgz'


def make_demo(output,seconds=24.):
    output=Path(output).resolve();output.parent.mkdir(parents=True,exist_ok=True)
    cache=output.parent/'tum-rgb-source';cache.mkdir(exist_ok=True)
    archive=cache/'desk.tgz'
    if not archive.exists():
        print('DOWNLOAD_REAL_RGB_RECORDING',URL,flush=True)
        partial=archive.with_suffix('.part');urllib.request.urlretrieve(URL,partial);partial.replace(archive)
    pictures=cache/'rgb';pictures.mkdir(exist_ok=True)
    rows=[]
    # Only RGB entries are read. groundtruth.txt/depth are never extracted.
    with tarfile.open(archive,mode='r|gz') as tar:
        stamps=[float(Path(m.name).stem) for m in tar if m.isfile() and '/rgb/' in m.name and m.name.endswith('.png')]
    if not stamps:raise RuntimeError('Official archive contains no RGB frames')
    base=min(stamps)
    # The source tar is not necessarily chronological. Stream it once instead
    # of repeatedly seeking backwards through gzip for time-sorted members.
    with tarfile.open(archive,mode='r|gz') as tar:
        for member in tar:
            if not member.isfile() or '/rgb/' not in member.name or not member.name.endswith('.png'):continue
            timestamp=float(Path(member.name).stem)-base
            if timestamp>seconds:continue
            name=Path(member.name).name
            with tar.extractfile(member) as f,Image.open(f) as pic:pic.convert('RGB').save(pictures/name)
            rows.append(dict(name=name,timestamp=timestamp))
    rows.sort(key=lambda r:r['timestamp'])
    encode_preview(rows,pictures,output)
    write_json(output.with_suffix('.provenance.json'),dict(source=URL,source_archive_sha256=sha256(archive),
        video_sha256=sha256(output),frames=len(rows),seconds=rows[-1]['timestamp'],
        explanation='MP4 encoded from a real TUM RGB recording; pipeline receives only this MP4. Depth and groundtruth poses are not used.'))
    print('RGB_ONLY_DEMO_VIDEO_READY',output,flush=True)
