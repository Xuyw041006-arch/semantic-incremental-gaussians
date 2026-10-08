import unittest
import numpy as np
from hglab.hierarchy import containment_tree,scale_layers,multiview_descriptors,infer_parent_ids,mask_physical_scale
from hglab.colmap import qrot

class HierarchyTests(unittest.TestCase):
    def test_nested_parent_is_smallest_containing_mask(self):
        m=np.zeros((4,20,20),bool); m[0]=True; m[1,2:18,2:18]=True
        m[2,5:10,5:10]=True; m[3,0:4,15:20]=True
        np.testing.assert_array_equal(containment_tree(m),[-1,0,1,0])

    def test_scale_changes_object_to_part_and_preserves_unknown(self):
        m=np.zeros((2,20,20),bool); m[0,2:18,2:18]=True; m[1,5:10,5:10]=True
        layers=scale_layers(m,[1.,.15],targets=(.7,.25,.08))
        self.assertEqual(layers[0,6,6],1); self.assertEqual(layers[2,6,6],2)
        self.assertEqual(layers[2,0,0],0)

    def test_view_dependent_clusters_are_retained(self):
        x=np.array([[1.,0,0],[.99,.01,0],[0,1.,0],[0,.99,.01]])
        centers,weights=multiview_descriptors(x)
        self.assertEqual(len(centers),2); self.assertAlmostEqual(weights.sum(),1.,places=6)
        self.assertGreater((centers@np.eye(3)[:2].T).max(0).min(),.99)

    def test_ambiguous_parent_not_forced(self):
        p=infer_parent_ids(np.array([1,1,2,2,2,2]),np.array([7,7,7,7,8,8]))
        self.assertEqual(p[7],0); self.assertEqual(p[8],2)

    def test_mask_scale_respects_depth_and_rotation(self):
        m=np.ones((10,10),bool); K=np.array([[10,0,5],[0,10,5],[0,0,1.]])
        s=mask_physical_scale(m,np.ones((10,10)),K)
        self.assertAlmostEqual(mask_physical_scale(m,np.ones((10,10))*2,K),2*s)

    def test_colmap_wxyz_rotation(self):
        np.testing.assert_allclose(qrot([1,0,0,0]),np.eye(3))
        np.testing.assert_allclose(qrot([0,0,0,1]),np.diag([-1,-1,1]))

if __name__=='__main__': unittest.main()
