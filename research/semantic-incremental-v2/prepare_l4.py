"""Restore identical posed RGB-D input without regenerating frozen teachers."""
from pathlib import Path
from sglab.tum import import_tum
from hglab.readers import load_scene
import numpy as np,hashlib,json
r=Path('/content/semantic-incremental')
manifest=r/'data/tum640/manifest.json'
original=json.loads(manifest.read_text())
import_tum('/content/tum-desk/rgbd_dataset_freiburg1_desk',manifest.parent,max_frames=120,every=5,width=640)
assert json.loads(manifest.read_text())==original
scene=load_scene(manifest);cached=np.load(r/'data/tum-desk/prefixes/anchors.npz')
for key in ['points','colors','release']:assert np.array_equal(scene[key],cached[key]),key
print('RESTORED_TUM_MANIFEST_AND_ALL_ANCHORS_IDENTICAL',len(scene['points']))
