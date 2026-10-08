import unittest
from analyze_results import factorial_contrasts,METRICS,VARIANTS

class AnalysisTest(unittest.TestCase):
    def test_paired_factor_effect_and_interaction(self):
        rows=[]
        for seed in (7,17,27):
            for f in VARIANTS:
                s,b,r=map(int,f)
                # A known non-additive response, with a nuisance seed offset.
                response=seed+2*s+3*b+5*r+7*s*b
                rows.append(dict(dataset='fixture',factors=f,seed=seed,**{k:response for k in METRICS}))
        result={r['contrast']:r for r in factorial_contrasts(rows,[7,17,27],'fixture') if r['metric']=='psnr'}
        self.assertAlmostEqual(result['main_S']['mean'],5.5)
        self.assertAlmostEqual(result['main_B']['mean'],6.5)
        self.assertAlmostEqual(result['main_R']['mean'],5.)
        self.assertAlmostEqual(result['interaction_SB']['mean'],7.)
        self.assertAlmostEqual(result['111_minus_011_S']['mean'],9.)
        self.assertAlmostEqual(result['111_minus_000']['mean'],17.)
        self.assertEqual(result['main_S']['sd'],0.)

if __name__=='__main__':unittest.main()
