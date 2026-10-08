import tempfile,unittest
from pathlib import Path
import numpy as np
from hglab.priority import node_priorities,anchor_priority,boundary_band,descriptor_table
from hglab.policies import sampling_weights
from videogs.splats import export_splats


class SemanticPriorityTest(unittest.TestCase):
    def test_task_importance_is_separate_from_background_and_unknown(self):
        nodes={'1':dict(names=['keyboard'],confidence=.9,parent=0),
               '2':dict(names=['floor']*3,confidence=.95,parent=0),
               '3':dict(names=['keyboard key'],confidence=.5,parent=1)}
        p=node_priorities(nodes,'keyboard')
        self.assertEqual(p['1']['priority'],3);self.assertTrue(p['2']['background'])
        self.assertEqual(p['3']['priority'],3)
        ids=np.array([[1,2,0],[3,0,0]]);confidence=np.array([[.9,.95,0],[.5,0,0]])
        importance,background=anchor_priority(ids,confidence,p)
        np.testing.assert_array_equal(importance,[3,0,1]);np.testing.assert_array_equal(background,[False,True,False])

    def test_boundary_is_local_and_empty_semantics_has_no_edges(self):
        self.assertFalse(boundary_band(np.zeros((20,20),int)).any())
        labels=np.ones((20,20),int);labels[:,10:]=2
        edge=boundary_band(labels,1)
        self.assertTrue(edge[:,9:11].all());self.assertFalse(edge[:,0].any());self.assertFalse(edge[:,-1].any())

    def test_priority_sampling_keeps_unknown_exploration(self):
        weights=sampling_weights([0,1,2],{0:[2],1:[4],2:[]},{},1,
            priorities={'1':{'priority':3},'2':{'priority':0}})
        self.assertGreater(weights[0],weights[1]);self.assertGreaterEqual(weights[2],1/6)
        self.assertAlmostEqual(weights.sum(),1)

    def test_descriptor_projection_has_no_future_graph_dependence(self):
        n={'1':dict(descriptors=[np.ones(512).tolist()])};before=descriptor_table(n)
        n['2']=dict(descriptors=[np.arange(512).tolist()]);after=descriptor_table(n)
        np.testing.assert_array_equal(before[1],after[1]);self.assertAlmostEqual(np.linalg.norm(before[1]),1,places=6)

    def test_full_splat_preserves_scale_opacity_quaternion_and_region_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'map.splat';values=dict(means=np.array([[1,2,3]],np.float32),
                scales=np.array([[.1,.2,.3]],np.float32),quats=np.array([[1,0,0,0]],np.float32),
                opacity=np.array([.8]),sh=np.zeros((1,9,3)))
            meta=export_splats(path,values,np.array([[13],[29]]));raw=path.read_bytes()
            self.assertEqual(len(raw),40);self.assertEqual(meta['count'],1)
            np.testing.assert_allclose(np.frombuffer(raw[:24],'<f4'),[1,2,3,.1,.2,.3])
            self.assertEqual(raw[27],204);self.assertEqual(list(raw[28:32]),[255,128,128,128])
            np.testing.assert_array_equal(np.frombuffer(raw[32:],'<i4'),[13,29])

if __name__=='__main__':unittest.main()
