import json
import unittest

from dinotool.semantic_coverage_transport import parse_transport


class TransportTests(unittest.TestCase):
    def test_one_entire_fence_is_the_same_as_raw_json(self):
        record=dict(eligible_slots=[1],queries=['wall'],competitor_names=['roof'])
        answer=json.dumps(dict(replacements=[]))
        self.assertEqual(parse_transport(answer,record),parse_transport(' \n```json\n'+answer+'\n```\n',record))

    def test_extra_objects_fences_or_prose_are_rejected(self):
        record=dict(eligible_slots=[1],queries=['wall'],competitor_names=['roof'])
        answer=json.dumps(dict(replacements=[]));fenced='```json\n'+answer+'\n```'
        for reply in (fenced+fenced,fenced+' trailing','explanation '+fenced,
                      '```json\n'+answer+answer+'\n```','```json\n'+answer,'```\n'+answer+'\n```'):
            with self.assertRaises(ValueError):parse_transport(reply,record)


if __name__=='__main__':unittest.main()
