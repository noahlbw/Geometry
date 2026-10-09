import unittest
from dinotool.kev_alias_selection import candidate_groups, admit, restored, question


class AliasAdmissionTests(unittest.TestCase):
    def test_rotation_restores_category_meaning(self):
        _,order=question('rooftop',2)
        self.assertEqual(restored([.1,.2,.3,.4],order),[.3,.4,.1,.2])
        with self.assertRaises(ValueError):restored([float('nan'),0,0,1],order)

    def test_variable_length_canonical_and_residual_protection(self):
        names=['other','wall','roof'];groups=[['other','mixed'],['wall','facade','roof'],['roof','rooftop']]
        scores=[{}, {'facade':{'probabilities':[.9,.1,0,0]},'roof':{'probabilities':[0,0,1,0]}},
                {'rooftop':{'probabilities':[.2,.8,0,0]}}]
        result=admit(names,groups,scores,.8,0)
        self.assertEqual([len(c['synonyms']) for c in result],[2,2,1])
        self.assertEqual(result[1]['synonyms'],['wall','facade'])

    def test_union_preserves_ontology_and_deduplicates(self):
        e={'banks':{'original':{'classes':[{'name':'car','synonyms':['car','Car','auto']}]},
                    'new':{'classes':[{'name':'car','synonyms':['automobile','auto']}]}}}
        self.assertEqual(candidate_groups(e),[['car','auto','automobile']])
        e['banks']['new']['classes'][0]['name']='truck'
        with self.assertRaises(ValueError):candidate_groups(e)


if __name__=='__main__':unittest.main()
