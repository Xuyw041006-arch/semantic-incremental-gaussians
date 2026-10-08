import unittest
import numpy as np
from hglab.policies import semantic_importance,coverage_replay,sampling_weights,crop_camera


class PoliciesTest(unittest.TestCase):
    def test_unknown_does_not_gain_confidence_bonus(self):
        ids=np.array([[1,1,0],[2,2,3]]);c=np.array([[.9,.9,1.],[.9,.9,.1]])
        s=semantic_importance(ids,c);self.assertEqual(s[2],0);self.assertGreater(s[0],0)

    def test_small_supported_part_receives_more_marginal_weight(self):
        ids=np.zeros((2,101),int);ids[1,:100]=1;ids[1,100]=2
        s=semantic_importance(ids,np.ones_like(ids,float));self.assertGreater(s[-1],s[0])

    def test_replay_capacity_and_rare_region(self):
        coverage={i:[2] for i in range(80)};coverage[37]=[2,3]
        pool=coverage_replay(range(80),coverage,capacity=8)
        self.assertEqual(len(set(pool)),8);self.assertIn(37,pool)

    def test_empty_semantics_still_explores(self):
        weights=sampling_weights([1,2,3],{}, {},1)
        self.assertTrue(np.all(weights>0));self.assertAlmostEqual(weights.sum(),1)
        self.assertEqual(len(coverage_replay(range(100),{},8)),8)

    def test_crop_ray_equivalence(self):
        im=dict(K=np.array([[500.,0,320],[0,500,240],[0,0,1]]),width=640,height=480)
        cropped=crop_camera(im,100,80,384,256)
        a=np.linalg.inv(im['K'])@np.array([150,120,1])
        b=np.linalg.inv(cropped['K'])@np.array([50,40,1])
        np.testing.assert_allclose(a,b)


if __name__=='__main__':unittest.main()
