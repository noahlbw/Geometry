import unittest
from types import SimpleNamespace

import torch

from dinotool.bounded_alias_path_audit import AuditTile, routed_change
from dinotool.bounded_alias_write_match import MATCHED, WRITERS, field_power, writer_changes


class WriteMatchTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(20261006)

    def source(self, count=2):
        tiles = []
        for index in range(count):
            relation = torch.rand(7, 7, dtype=torch.float64)
            valid = torch.tensor([True]*5+[False]*2)
            relation = relation.masked_fill(~valid[None], 0.)
            relation = (relation/relation.sum(-1, keepdim=True)).masked_fill(~valid[:, None], 0.)
            gram = relation.T@relation
            h = torch.linalg.solve(torch.eye(7)+gram, gram)
            tiles.append(AuditTile(index, index, {}, torch.empty(0), torch.empty(0), h, valid, relation))
        return SimpleNamespace(tiles=tiles)

    def test_h_and_two_h_preserve_original_action_exactly(self):
        source = self.source()
        changes = [torch.randn(7, 6) for _ in source.tiles]
        result, _ = writer_changes(source, changes)
        for tile, value, actual, twice in zip(source.tiles, changes, result['H'], result['TwoH']):
            expected = routed_change(torch.zeros_like(value), value, tile.operator, 'Wide')
            self.assertTrue(torch.equal(actual, expected))
            self.assertTrue(torch.equal(twice, expected*2))

    def test_matching_uses_whole_image_not_separate_tile_scales(self):
        source = self.source()
        changes = [torch.randn(7, 6)*scale for scale in (1., 20.)]
        result, diagnostics = writer_changes(source, changes)
        goal = field_power(result['H'], source.tiles)
        for name in MATCHED:
            torch.testing.assert_close(field_power(result[name], source.tiles), goal, atol=1e-12, rtol=1e-12)
            self.assertLess(diagnostics[name+'_norm_max_relative_error'], 1e-12)
        first = result['DirectMatched'][0][:5]/changes[0][:5]
        second = result['DirectMatched'][1][:5]/changes[1][:5]
        torch.testing.assert_close(first[0], second[0], atol=1e-7, rtol=1e-7)

    def test_matched_paths_exclude_padding_but_legacy_direct_is_exact(self):
        source = self.source()
        changes = [torch.randn(7, 6) for _ in source.tiles]
        result, _ = writer_changes(source, changes)
        for name in (*MATCHED, 'H', 'TwoH'):
            for value in result[name]:
                self.assertTrue(torch.equal(value[5:], torch.zeros_like(value[5:])))
        for value, original in zip(result['DirectLegacy'], changes):
            self.assertTrue(torch.equal(value, original.double()))
        altered = [value.clone() for value in changes]
        for value in altered:
            value[5:] += 1000
        second, _ = writer_changes(source, altered)
        for name in (*MATCHED, 'H', 'TwoH'):
            for a, b in zip(result[name], second[name]):
                torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_nonnegative_paths_do_not_raise_a_negative_source_action(self):
        source = self.source()
        result, diagnostics = writer_changes(source, [-torch.rand(7, 6) for _ in source.tiles])
        for name in MATCHED:
            self.assertEqual(diagnostics[name+'_fallback_columns'], 0)
            self.assertTrue(all(bool((field <= 0).all()) for field in result[name]))

    def test_zero_action_replays_identity_and_has_no_fallback(self):
        source = self.source()
        result, diagnostics = writer_changes(source, [torch.zeros(7, 6) for _ in source.tiles])
        self.assertTrue(all(torch.equal(field, torch.zeros_like(field)) for fields in result.values() for field in fields))
        self.assertTrue(all(value == 0 for value in diagnostics.values()))

    def test_geometry_row_scaling_does_not_change_support(self):
        source = self.source()
        changes = [torch.randn(7, 6) for _ in source.tiles]
        first, _ = writer_changes(source, changes)
        for tile in source.tiles:
            tile.relation = tile.relation*torch.arange(1, 8)[:, None]
        second, _ = writer_changes(source, changes)
        for a, b in zip(first['GeometryMatched'], second['GeometryMatched']):
            torch.testing.assert_close(a, b, atol=1e-12, rtol=1e-12)

    def test_zero_template_falls_back_to_h_with_explicit_count(self):
        source = self.source(1)
        source.tiles[0].relation.zero_()
        result, diagnostics = writer_changes(source, [torch.ones(7, 6)])
        self.assertEqual(diagnostics['GeometryMatched_fallback_columns'], 6)
        self.assertTrue(torch.equal(result['GeometryMatched'][0], result['H'][0]))

    def test_bad_source_or_action_is_rejected(self):
        source = self.source(1)
        with self.assertRaises(ValueError):
            writer_changes(source, [torch.full((7, 6), torch.nan)])
        source.tiles[0].relation = None
        with self.assertRaises(ValueError):
            writer_changes(source, [torch.ones(7, 6)])
        self.assertEqual(len(WRITERS), 6)


if __name__ == '__main__':
    unittest.main()
