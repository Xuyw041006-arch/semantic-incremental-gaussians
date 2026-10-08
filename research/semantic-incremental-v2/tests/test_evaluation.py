import unittest
import numpy as np
from hglab.evaluation import matched_iou

class EvaluationTest(unittest.TestCase):
    def test_unknowns_are_misses(self):
        target=np.array([[1,1],[2,2]])
        result=matched_iou(np.zeros_like(target),target)
        self.assertEqual(result['miou'],0.)
        self.assertEqual(result['coverage'],0.)

    def test_merge_cannot_match_two_reference_regions(self):
        target=np.array([[1,1],[2,2]])
        # The one predicted blob matches only one of the two regions, IoU .5.
        result=matched_iou(np.full_like(target,9),target)
        self.assertAlmostEqual(result['miou'],.25)
        self.assertEqual(result['regions'],2)

    def test_region_numbering_does_not_change_agreement(self):
        target=np.array([[1,1,0],[2,2,0]])
        pred=np.array([[9,9,0],[6,6,0]])
        self.assertAlmostEqual(matched_iou(pred,target)['miou'],1.)

if __name__=='__main__':unittest.main()
