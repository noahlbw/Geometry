import unittest
from dinotool.development_readout import Profile
from dinotool.task_read_strength import select_read_strength


class ReadStrengthTests(unittest.TestCase):
    def test_only_strength_changes(self):
        p=Profile(bank='semantic_segmentation',strength=2.,coupling=2.)
        d={'selected_gain':2.,'selected_strength':2.}
        out,decision=select_read_strength(p,d,'natural_strength3')
        self.assertEqual(out.strength,3.)
        for k,v in p.record().items():
            if k!='strength':self.assertEqual(out.record()[k],v)
        self.assertEqual(d['selected_strength'],2.)
        self.assertEqual(decision['selected_strength'],3.)

    def test_original_is_retained(self):
        p=Profile(strength='original',coupling=.5)
        out,_=select_read_strength(p,{},'natural_strength3')
        self.assertIs(out,p)

    def test_remote_image_selector_unchanged(self):
        self.assertEqual(select_read_strength(None,None,'natural_strength3'),(None,None))

    def test_default_exact_objects(self):
        p,d=Profile(),{}
        out,decision=select_read_strength(p,d,'retained')
        self.assertIs(out,p);self.assertIs(decision,d)


if __name__=='__main__':unittest.main()
