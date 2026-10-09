from dataclasses import fields, replace
import unittest

import torch
from torch import nn
import torch.nn.functional as F

from dinotool.geometry_execution import GeometryExecution, prepare_image_pruned
from dinotool.tcpr import TCPRConfig, TCPRSegmenter
from test_tcpr import _TinyBackbone, _TinyBlock


def fixture(device='cpu'):
    backbone = _TinyBackbone()
    backbone.device = torch.device(device)
    visual = backbone.model.visual_model
    visual.head.blocks = nn.ModuleList([_TinyBlock(), _TinyBlock()])
    backbone.model.eval().requires_grad_(False).to(device)
    backbone._imagenet_mean = backbone._imagenet_mean.to(device)
    backbone._imagenet_std = backbone._imagenet_std.to(device)
    def features(rgb):
        raw = F.adaptive_avg_pool2d(rgb, (2, 2)).flatten(2).transpose(1, 2)
        raw = torch.cat((raw, torch.ones_like(raw[..., :1])), -1)
        return raw.mean(1), raw, raw[:, :1] * .5
    visual.get_backbone_features = features
    return TCPRSegmenter(backbone, TCPRConfig(geometry_depth=2))


def require_prepared_equal(test, actual, expected):
    for field in fields(expected):
        a, b = getattr(actual, field.name), getattr(expected, field.name)
        if isinstance(b, torch.Tensor):
            test.assertTrue(torch.equal(a, b), field.name)
        else:
            test.assertEqual(a, b, field.name)


class GeometryExecutionTests(unittest.TestCase):
    def test_pruned_all_prepared_fields_cpu_exact(self):
        torch.manual_seed(20261005)
        model = fixture()
        state = {name: value.clone() for name, value in model.backbone.model.state_dict().items()}
        for depth in (1, 2):
            for prefix in ('preserve', 'block'):
                for relation in ('dense', 'sparse'):
                    model.config = replace(model.config, geometry_depth=depth, prefix_policy=prefix, relation_policy=relation)
                    image = torch.rand(1, 3, 32, 32)
                    require_prepared_equal(self, prepare_image_pruned(model, image), model.prepare_image(image))
        self.assertTrue(all(torch.equal(state[name], value) for name, value in model.backbone.model.state_dict().items()))

    def test_changed_config_and_training_head_rejected(self):
        model = fixture()
        view = GeometryExecution(model)
        model.config = replace(model.config, prefix_policy='block')
        with self.assertRaises(ValueError):
            view.prepare_image(torch.zeros(1, 3, 32, 32))
        model.backbone.model.visual_model.head.train()
        with self.assertRaises(ValueError):
            prepare_image_pruned(model, torch.zeros(1, 3, 32, 32))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for graph ownership/equality.')
    def test_pruned_and_graph_cuda_exact_and_results_survive_replay(self):
        torch.manual_seed(20261005)
        model = fixture('cuda')
        graph = GeometryExecution(model, graph=True)
        a, b = torch.rand(1, 3, 32, 32, device='cuda'), torch.rand(1, 3, 32, 32, device='cuda')
        expected_a, expected_b = model.prepare_image(a), model.prepare_image(b)
        require_prepared_equal(self, prepare_image_pruned(model, a), expected_a)
        actual_a, actual_b = graph.prepare_image(a), graph.prepare_image(b)
        require_prepared_equal(self, actual_a, expected_a)
        require_prepared_equal(self, actual_b, expected_b)
        self.assertFalse(torch.equal(actual_a.backbone_tokens, actual_b.backbone_tokens))
        self.assertEqual(graph.replays, 2)
        self.assertGreater(graph.setup_seconds, 0.)
        with self.assertRaises(ValueError):
            graph.prepare_image(torch.zeros(1, 3, 32, 32, device='cuda', dtype=torch.float64))


if __name__ == '__main__':
    unittest.main()
