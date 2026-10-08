import unittest
import numpy as np
from hglab.policies import sampling_weights,replay_probability,coverage_replay
class BalanceTests(unittest.TestCase):
 def test_unlabelled_frames_retain_exploration_floor(self):
  indices=list(range(100));coverage={0:[2]};births={'2':6};errors={0:10.}
  p=sampling_weights(indices,coverage,births,6,errors)
  self.assertAlmostEqual(p.sum(),1.)
  self.assertGreaterEqual(p[1:].min(),.5/100)
  self.assertLess(p.max()/p.min(),4.)
 def test_long_stream_replay_keeps_new_observations(self):
  p=[replay_probability(s,.6) for s in range(1,1000)]
  self.assertTrue(all(a<=b for a,b in zip(p,p[1:])))
  self.assertLess(max(p),.6)
 def test_unknown_stream_temporal_reserve(self):
  selected=coverage_replay(list(range(1000)),{},48)
  self.assertEqual(len(set(selected)),48)
  self.assertIn(0,selected);self.assertIn(999,selected)
if __name__=='__main__':unittest.main()
