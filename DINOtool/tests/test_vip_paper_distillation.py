import unittest
from types import SimpleNamespace

import torch

from dinotool.vip_paper_distillation import (
    BackboneAttentionCapture, PaperAliasAccumulator, PaperDistillationConfig,
    aggregate_logits, canonical_indices, image_crops, random_walk,
    replacement_probabilities, select_aliases, valid_patches,
)


class VIPPaperDistillationTest(unittest.TestCase):
    def fixture(self):
        return torch.tensor([0, 0, 1, 1]), torch.tensor([0, 2])

    def test_replaces_only_own_class(self):
        parents, canonical = self.fixture()
        logits = torch.tensor([[1., 8., 3., -2.], [2., 4., 5., 9.]])
        actual = replacement_probabilities(logits, parents, canonical, torch.arange(4))
        expected = torch.stack([logits[:, [0, 2]], logits[:, [1, 2]],
                                logits[:, [0, 2]], logits[:, [0, 3]]]).softmax(-1)
        self.assertTrue(torch.equal(actual, expected))
        self.assertTrue(torch.equal(actual[0], actual[2]))

    def test_random_walk_excludes_padded_keys(self):
        valid = torch.tensor([True, True, False])
        walk = random_walk(torch.ones(3, 3), valid, PaperDistillationConfig())
        self.assertTrue(torch.allclose(walk.sum(-1), torch.ones(3)))
        self.assertEqual(float(walk[:, 2].abs().max()), 0.)
        self.assertTrue(torch.equal(valid_patches((0, 0, 16, 32), 'cpu').reshape(21, 21)[0, :3],
                                    torch.tensor([True, True, False])))

    def test_image_balanced_not_patch_balanced(self):
        parents, canonical = self.fixture()
        config = PaperDistillationConfig()
        collector = PaperAliasAccumulator(4, 'cpu')
        image_scores = []
        for copies, logits in ((1, torch.tensor([[1., 2., 0., -1.]])),
                               (10, torch.tensor([[3., 2., 0., 4.]]))):
            p = replacement_probabilities(logits, parents, canonical, torch.arange(4))
            own = p[..., 0].flatten()[0]
            image_scores.append(float(own / (2 - own)))
            for _ in range(copies):
                collector.update_crop(logits, parents, canonical, torch.ones(1, 1),
                                      torch.ones(1, dtype=torch.bool), config)
            collector.finalize_image()
        state = collector.state()
        self.assertEqual(state['observed_images'][0], 2)
        self.assertAlmostEqual(state['vg_sum'][0] / 2, sum(image_scores) / 2, places=6)
        self.assertEqual(state['observed_images'][2], 0)

    def test_skip_no_high_region_and_keep_canonical(self):
        queries = SimpleNamespace(class_names=('a', 'b'), aliases=('a', 'a2', 'b', 'b2'),
                                  parents=torch.tensor([0, 0, 1, 1]),
                                  features=torch.tensor([[[1., 0.]], [[1., 0.]], [[0., 1.]], [[0., 1.]]]))
        state = dict(vg_sum=[.3, .4, 0., .9], sc_sum=[.5, .4, 0., .1], observed_images=[1, 1, 0, 1])
        selected = select_aliases(state, queries, PaperDistillationConfig())
        self.assertEqual(selected['keep'], [True, True, True, False])
        state['vg_sum'][1] = .3
        self.assertEqual(select_aliases(state, queries, PaperDistillationConfig())['selected_counts'], [1, 1])

    def test_text_prefilter_is_independent(self):
        queries = SimpleNamespace(class_names=('a',), aliases=('a', 'wrong'), parents=torch.zeros(2, dtype=torch.long),
                                  features=torch.eye(2)[:, None])
        state = dict(vg_sum=[.3, .9], sc_sum=[.5, .1], observed_images=[1, 1])
        selected = select_aliases(state, queries, PaperDistillationConfig())
        self.assertEqual(selected['selected_counts'], [1])
        self.assertEqual(selected['aliases'][1]['reason'], 'text_prefilter')

    def test_aggregation_recomputes_survivor_salience_exactly(self):
        parents, _ = self.fixture()
        logits = torch.arange(4.).reshape(4, 1, 1).expand(4, 21, 21)
        salience = torch.tensor([1., 2., 3., 4.])
        settings = SimpleNamespace(tau=4.)
        keep = torch.tensor([True, False, True, False])
        actual = aggregate_logits(logits, salience, parents, 2, settings, keep)
        self.assertTrue(torch.equal(actual[:, 0, 0], torch.tensor([0., 2.])))

    def test_panorama_crops_keep_exact_pinned_resize(self):
        settings = SimpleNamespace(resize_long_edge=448, slide_crop=336, slide_stride=112)
        original, resized, crops = image_crops(torch.zeros(3, 100, 500), settings)
        self.assertEqual(original, (100, 500))
        self.assertEqual(resized, (90, 448))
        self.assertEqual(len(crops), 2)
        self.assertEqual(crops[0][0].shape, (3, 336, 336))

    def test_capture_preserves_forward_and_removes_hooks(self):
        class Attention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.qkv = torch.nn.Linear(2, 6, bias=False)
                self.num_heads, self.scale = 1, 2 ** -.5

            def forward(self, x, rope=None):
                return x * 2

        attention = Attention()
        model = SimpleNamespace(blocks=[SimpleNamespace(attn=attention)])
        inputs = torch.ones(1, 3, 2)
        plain = attention(inputs)
        with BackboneAttentionCapture(model, patch_count=2) as capture:
            self.assertTrue(torch.equal(attention(inputs), plain))
            self.assertEqual(capture.count, 1)
            self.assertTrue(torch.allclose(capture.mean().sum(-1), torch.full((2,), 2/3)))
        self.assertFalse(attention._forward_pre_hooks)
        with self.assertRaises(RuntimeError):
            BackboneAttentionCapture(model, patch_count=2).mean()

    def test_canonical_lookup_rejects_missing_name(self):
        with self.assertRaises(ValueError):
            canonical_indices(('a',), ('a2',), torch.tensor([0]))


if __name__ == '__main__':
    unittest.main()
