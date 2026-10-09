import unittest
from dinotool.variable_alias_vocabulary import parse_reply,remove_shared_additions,freeze_pool


class VariableAliasTests(unittest.TestCase):
    def test_no_minimum_and_canonical_duplicate(self):
        self.assertEqual(parse_reply('[]','car',['road'])[0],[])
        a,r=parse_reply('[{"phrase":" CAR ","kind":"synonym"},{"phrase":"sedan","kind":"subtype"}]','car',['road'])
        self.assertEqual([x['phrase'] for x in a],['sedan']);self.assertEqual(r[0]['reason'],'duplicate_or_anchor')

    def test_exact_competitor_and_kind_contract(self):
        a,r=parse_reply('[{"phrase":"LOW-vegetation","kind":"synonym"}]','tree',['low vegetation'])
        self.assertFalse(a);self.assertEqual(r[0]['reason'],'exact_competitor_concept')
        with self.assertRaises(ValueError):parse_reply('[{"phrase":"green","kind":"attribute"}]','tree',[])

    def test_shared_additions_are_removed_symmetrically(self):
        records=[dict(accepted=[dict(phrase=p,kind='synonym')],rejected=[]) for p in ('seat','Seat')]
        remove_shared_additions(records)
        self.assertTrue(all(not r['accepted'] for r in records))

    def test_invalid_class_falls_back_without_changing_source_or_background(self):
        classes=[dict(name='background',synonyms=['sky','floor']),dict(name='bicycle',synonyms=['bicycle','bike'])]
        protocol=dict(datasets={'task':dict(family='natural',background_index=0,banks={'semantic_segmentation':dict(classes=classes)})})
        source=dict(status='failed',target_images_loaded=False,target_masks_loaded=False,target_label_tuning=False,
            records=[dict(dataset='task',index=1,name='bicycle',anchor='bicycle',format_complete=False,
                accepted=[dict(phrase='wheel',kind='subtype')],rejected=[])])
        pool=freeze_pool(protocol,source);out=pool['datasets']['task']['semantic_segmentation']['classes']
        self.assertEqual(out[0],classes[0]);self.assertEqual(out[1]['synonyms'],['bicycle'])
        self.assertEqual(source['records'][0]['accepted'][0]['phrase'],'wheel')
        source['target_label_tuning']=True
        with self.assertRaises(ValueError):freeze_pool(protocol,source)
