import tempfile,struct,unittest
from pathlib import Path
from PIL import Image
import numpy as np
from hglab.colmap import read_scene


class ColmapProtocolTests(unittest.TestCase):
    def test_release_requires_two_arrived_nonheldout_observations(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); sparse=root/'sparse/0'; sparse.mkdir(parents=True); (root/'images_4').mkdir()
            def pack(f,fmt,*values): f.write(struct.pack('<'+fmt,*values))
            with (sparse/'cameras.bin').open('wb') as f:
                pack(f,'Q',1); pack(f,'iiQQ',1,1,100,60); pack(f,'dddd',80,80,50,30)
            with (sparse/'images.bin').open('wb') as f:
                pack(f,'Q',10)
                for i in range(10):
                    name=f'{i:03d}.png'; Image.new('RGB',(25,15)).save(root/'images_4'/name)
                    pack(f,'i'+'d'*7+'i',i,1,0,0,0,-i,0,0,1); f.write(name.encode()+b'\0'); pack(f,'Q',0)
            with (sparse/'points3D.bin').open('wb') as f:
                pack(f,'Q',3)
                for pid,track in [(1,[0,1,5,8]),(2,[0,8]),(3,[2,3])]:
                    pack(f,'QdddBBBd',pid,2.,0.,4.,120,130,140,.1); pack(f,'Q',len(track))
                    for iid in track: pack(f,'ii',iid,0)
            scene=read_scene(root,width=25)
            np.testing.assert_array_equal(scene['release'],[5,3])
            self.assertEqual(len(scene['points']),2)
            np.testing.assert_allclose(scene['images'][0]['K'],[[20,0,12.5],[0,20,7.5],[0,0,1]])
            self.assertTrue(scene['images'][0]['test']); self.assertTrue(scene['images'][8]['test'])
            self.assertFalse(scene['images'][1]['test'])
            np.testing.assert_allclose(scene['images'][0]['c2w'][:3,3],[-1,0,0])

if __name__=='__main__': unittest.main()
