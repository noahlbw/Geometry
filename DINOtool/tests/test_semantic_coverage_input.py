import json
import unittest

from dinotool.semantic_coverage_input import parse_reply,lexical_filter,compile_banks


class CoverageInputTests(unittest.TestCase):
    def record(self,name='vehicle'):
        return dict(dataset='toy',class_index=0,class_name=name,queries=[name]+[f'{name} old {i}' for i in range(1,20)],
            protected_slots=list(range(19)),eligible_slots=[19],competitor_names=['road'],accepted=[],rejected=[])

    def item(self,phrase='delivery van',slot=19):
        return dict(slot_id=slot,phrase=phrase,relation='subtype',meaning_added='A cargo vehicle subtype',basis='A van is a road vehicle')

    def test_protected_and_competitor_proposals_are_rejected(self):
        r=self.record()
        for item in (self.item(slot=0),self.item(phrase='road')):
            accepted,rejected=parse_reply(json.dumps(dict(replacements=[item])),r)
            self.assertEqual(accepted,[]);self.assertEqual(len(rejected),1)

    def test_unfilled_slot_is_inherited_not_padded(self):
        r=self.record();pool=compile_banks([r],dict(complete=True,decisions=[]))
        for classes in pool['datasets']['toy'].values():self.assertEqual(classes[0]['synonyms'],r['queries'])

    def test_subtype_is_kept_in_candidate_but_not_synonym_control(self):
        r=self.record();r['accepted']=[self.item()]
        review=dict(complete=True,decisions=[dict(dataset='toy',class_index=0,slot_id=19,
            phrase='delivery van',relation='subtype',relation_confirmed=True,
            accept=True,reason='A concrete road vehicle subtype')])
        pool=compile_banks([r],review)['datasets']['toy']
        self.assertEqual(pool['Coverage20'][0]['synonyms'][19],'delivery van')
        self.assertEqual(pool['SynonymOnly20'][0]['synonyms'],r['queries'])

    def test_shared_additions_reject_both_owners(self):
        a,b=self.record(),self.record('roof');b['class_index']=1
        a['accepted']=[self.item()];b['accepted']=[self.item()]
        lexical_filter([a,b])
        self.assertEqual(a['accepted'],[]);self.assertEqual(b['accepted'],[])

    def test_entire_reply_must_be_one_json_object(self):
        answer=json.dumps(dict(replacements=[self.item()]))
        for reply in (answer+answer,'```json\n'+answer+'\n```','Explanation '+answer):
            with self.assertRaises(ValueError):parse_reply(reply,self.record())

    def test_strided_shard_concatenation_requires_restored_class_order(self):
        records=[self.record('vehicle '+str(i)) for i in range(8)]
        for i,record in enumerate(records):record['class_index']=i
        concatenated=[r for shard in range(4) for r in records[shard::4]]
        with self.assertRaises(ValueError):compile_banks(concatenated,dict(complete=True,decisions=[]))
        restored=sorted(concatenated,key=lambda r:r['class_index'])
        self.assertEqual(len(compile_banks(restored,dict(complete=True,decisions=[]))['datasets']['toy']['Existing20']),8)

    def test_unconfirmed_or_changed_relation_cannot_enter_synonym_control(self):
        r=self.record();r['accepted']=[self.item()]
        decision=dict(dataset='toy',class_index=0,slot_id=19,phrase='delivery van',
            relation='subtype',relation_confirmed=False,accept=True,reason='Valid subtype')
        with self.assertRaises(ValueError):compile_banks([r],dict(complete=True,decisions=[decision]))
        decision.update(relation='synonym',relation_confirmed=True)
        with self.assertRaises(ValueError):compile_banks([r],dict(complete=True,decisions=[decision]))


if __name__=='__main__':unittest.main()
