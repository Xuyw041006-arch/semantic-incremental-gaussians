import unittest
from hglab.refinement import RefinementWindow


class RefinementTest(unittest.TestCase):
    def test_every_batch_has_warmup_and_densification_free_tail(self):
        w=RefinementWindow(2400)
        self.assertEqual([i for i in range(2400) if w.refine(i)],list(range(200,1600,100)))
        self.assertEqual(sum(w.phase(i)=='warmup' for i in range(2400)),200)
        self.assertEqual(sum(w.phase(i)=='settle' for i in range(2400)),800)
        self.assertFalse(w.refine(0))
        # New children born at the last growth get 899 subsequent updates.
        self.assertEqual(2399-max(i for i in range(2400) if w.refine(i)),899)

    def test_short_runs_do_not_start_with_growth(self):
        for steps in [1,2,10,50,600]:
            w=RefinementWindow(steps)
            self.assertFalse(w.refine(0))
            self.assertLessEqual(w.warmup_end,w.settle_start)
            self.assertEqual(len([w.phase(i) for i in range(steps)]),steps)

    def test_invalid_window_fails_explicitly(self):
        for kw in [dict(steps=0),dict(steps=2400,warmup=-1),dict(steps=2400,interval=0)]:
            with self.assertRaises(ValueError):RefinementWindow(**kw)


if __name__=='__main__':unittest.main()
