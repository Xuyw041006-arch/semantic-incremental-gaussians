import unittest,tempfile,json
from pathlib import Path
import numpy as np
from PIL import Image
from hglab.readers import load_scene

class CausalityTest(unittest.TestCase):
    def test_future_rgb_depth_and_pose_do_not_change_prefix_anchors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frames=[]
            for i in range(4):
                Image.fromarray(np.full((16,16,3),40+i,np.uint8)).save(root/f'{i}.png')
                np.save(root/f'{i}.npy',np.ones((16,16),np.float32))
                pose=np.eye(4);pose[0,3]=i*.1
                frames.append(dict(rgb=f'{i}.png',depth=f'{i}.npy',c2w=pose.tolist(),K=[[20,0,8],[0,20,8],[0,0,1]]))
            manifest=dict(pose_convention='opencv_c2w_meters',frames=frames,depth_scale=1.)
            path=root/'manifest.json';path.write_text(json.dumps(manifest));before=load_scene(path,16)
            np.save(root/'2.npy',np.full((16,16),3,np.float32));frames[2]['c2w'][0][3]=100
            Image.fromarray(np.full((16,16,3),255,np.uint8)).save(root/'2.png')
            path.write_text(json.dumps(manifest));after=load_scene(path,16)
            a=before['release']<2;b=after['release']<2
            np.testing.assert_array_equal(before['points'][a],after['points'][b])
            np.testing.assert_array_equal(before['colors'][a],after['colors'][b])

    def test_heldout_depth_is_never_admitted(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);frames=[]
            for i in range(4):
                Image.fromarray(np.zeros((16,16,3),np.uint8)).save(root/f'{i}.png')
                np.save(root/f'{i}.npy',np.ones((16,16),np.float32)*(i+1))
                frames.append(dict(rgb=f'{i}.png',depth=f'{i}.npy',c2w=np.eye(4).tolist(),K=[[20,0,8],[0,20,8],[0,0,1]]))
            p=root/'manifest.json';p.write_text(json.dumps(dict(pose_convention='opencv_c2w_meters',frames=frames)))
            scene=load_scene(p,16);self.assertNotIn(3,scene['release'].tolist())

if __name__=='__main__':unittest.main()
