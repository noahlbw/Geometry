import copy
from dataclasses import asdict
import unittest

from eval_vip_paper_distillation import verify_references
from dinotool.vip_official_adapter import VIPSettings


class CandidateReferenceTest(unittest.TestCase):
    def fixture(self):
        settings = VIPSettings()
        manifest = {'dinotxt': {'bytes': 1}}
        ref = dict(status='complete', coverage_verified=True, processed_images=2, total_images=2,
            signature=dict(dataset='vdd', checkpoint_manifest=manifest, settings=asdict(settings),
                upstream_commit='pinned', vocabulary_sha256='official', sample_keys=['a', 'b']))
        candidate = dict(status='complete', coverage_verified=True, processed_images=2, total_images=2,
            signature=dict(dataset='vdd', checkpoints=manifest, sample_keys=['a', 'b'], classes={'vdd': ['road']},
                vocabulary=dict(sha256='twenty', aliases={'vdd': ['road', 'street']}, counts={'vdd': [2]})))
        args = (ref, manifest, 'twenty', settings, 'pinned', {'vdd': [('road', 'street')]}, {'vdd': ('road',)})
        return args, candidate

    def test_short_reference_requires_explicit_candidate_reference(self):
        args, candidate = self.fixture()
        with self.assertRaises(RuntimeError):
            verify_references(*args)
        verify_references(*args, candidate)

    def test_changed_candidate_identity_rejected(self):
        args, candidate = self.fixture()
        for field, value in (('sample_keys', ['b', 'a']), ('checkpoints', {}),
                             ('classes', {'vdd': ['wrong']}), ('vocabulary', {})):
            changed = copy.deepcopy(candidate)
            changed['signature'][field] = value
            with self.assertRaises(RuntimeError):
                verify_references(*args, changed)

    def test_existing_same_bank_reference_unchanged(self):
        args, _ = self.fixture()
        args[0]['signature']['vocabulary_sha256'] = 'twenty'
        verify_references(*args)


if __name__ == '__main__':
    unittest.main()
