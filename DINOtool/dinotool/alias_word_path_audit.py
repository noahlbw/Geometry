"""Label-free decomposition of the existing soft action; never changes its output."""
import torch

from .fine_alias_view import CONFIG
from .native_alias_noise import signed_potential
from .native_query_alias import native_risk
from .reciprocal_alias_admission import (CLASS_MEAN, DIAGNOSTICS, METHODS,
    OBSERVATION_MEAN, ONE_SIDED_SOFT, continuous_controls, fixed_slot_action, reciprocal_scores)
from .rival_alias_fast import sampled_cached
from .supported_positive_alias import predict_image as source_predict


IMPLEMENTATION = "frozen-soft-alias-word-path-audit-v1-20261006"


def ratio(numerator, denominator):
    numerator, denominator = float(numerator), float(denominator)
    if denominator == 0:
        if numerator != 0:
            raise ValueError("Nonzero numerator with zero reference power.")
        return 0.
    return numerator / denominator


def decomposition(risk, members, canonical, valid, directed, class_directed, operator, positive):
    n, c, k = len(valid), *members.shape
    if (risk.shape != (n, members.numel(), c) or canonical.shape != (c,)
            or directed.shape != (n, c, c) or class_directed.shape != directed.shape
            or operator.shape != (n, n) or positive.shape != (n, c)
            or valid.dtype != torch.bool or not bool(valid.any())
            or not all(bool(torch.isfinite(v).all()) for v in (risk, directed, class_directed, operator, positive))
            or bool(((risk < 0) | (risk > 1)).any())
            or len(members.flatten().unique()) != members.numel()
            or not bool((members == canonical[:, None]).any(-1).all()) or k < 2):
        raise ValueError("Matching finite source fields and actual class membership required.")
    grouped = risk[:, members].double()
    noncanonical = members != canonical[:, None]
    rival = ~torch.eye(c, device=risk.device, dtype=torch.bool)
    slots = valid[:, None, None, None] & noncanonical[None, :, :, None] & rival[None, :, None, :]
    average = (grouped * noncanonical[None, :, :, None]).sum(2) / noncanonical.sum(1)[None, :, None]
    centered = (grouped - average[:, :, None]).masked_fill(~slots, 0.)
    selected = grouped.masked_fill(~slots, 0.)
    full, _ = signed_potential(directed, valid)
    common, _ = signed_potential(class_directed, valid)
    word = full - common
    pair = ((directed - class_directed) - (directed - class_directed).transpose(-1, -2)).double()
    gradient = word[:, :, None] - word[:, None, :]
    cycle = pair - gradient
    pair_power, gradient_power, cycle_power = (v.square().sum() for v in (pair, gradient, cycle))
    error = float((pair_power - gradient_power - cycle_power).abs())
    if error > 1e-10 * max(1., float(pair_power)):
        raise RuntimeError("Complete-graph gradient/cycle orthogonality failed.")
    full_write, common_write, word_write = (operator.double() @ v for v in (full, common, word))
    logits, class_logits = positive.double() + .5 * full_write, positive.double() + .5 * common_write
    return dict(
        noncanonical_active_fraction=float((grouped[slots] > 0).double().mean()),
        within_class_risk_power_fraction=ratio(centered.square().sum(), selected.square().sum()),
        word_directed_relative_norm=ratio((directed - class_directed).double().norm(), directed.double().norm()),
        word_pair_gradient_power_fraction=ratio(gradient_power, pair_power),
        word_pair_cycle_power_fraction=ratio(cycle_power, pair_power),
        projection_orthogonality_absolute_error=error,
        word_potential_relative_norm=ratio(word.norm(), full.norm()),
        word_H_power_retention=ratio(word_write.square().sum(), word.square().sum()),
        common_H_power_retention=ratio(common_write.square().sum(), common.square().sum()),
        word_write_relative_norm=ratio(word_write.norm(), full_write.norm()),
        soft_vs_class_patch_argmax_fraction=float((logits[valid].argmax(-1) != class_logits[valid].argmax(-1)).double().mean()),
        valid_queries=int(valid.sum()), classes=c, aliases_per_class=k)


class WordPathAudit:
    def __init__(self):
        self.rows = []

    def begin(self, sample_key, banks):
        self.sample_key = sample_key
        self.protocols = {bank.class_count: p for p, bank in banks.items()}
        if len(self.protocols) != len(banks):
            raise ValueError("This audit requires distinguishable protocol class counts.")
        self.tiles = dict.fromkeys(banks, 0)

    @torch.inference_mode()
    def score(self, local, operator, broad, fine_field, wide, fine, coordinates,
              relation, valid, members, canonical, parents, *, methods):
        if tuple(methods) != (ONE_SIDED_SOFT,):
            raise ValueError("Audit only the existing single-sided soft prediction.")
        values, diagnostics = reciprocal_scores(local, operator, broad, fine_field, wide, fine,
            coordinates, relation, valid, members, canonical, parents, methods=methods)
        wm, fm = sampled_cached(wide, coordinates), sampled_cached(fine, coordinates)
        fm = fm.masked_fill(~valid[:, None, None], 0.)
        risk = native_risk(wm, fm, fm, parents, canonical, valid, CONFIG)
        controls, _ = continuous_controls(risk, members, canonical)
        directed, _ = fixed_slot_action(wide, members, risk, valid)
        common, _ = fixed_slot_action(wide, members, controls[CLASS_MEAN], valid)
        # Match the deployed source's exact arithmetic ordering, not an algebraic rewrite.
        positive = local.double() + operator.double() @ (broad.double() - local.double())
        innovation = .5 * (fine_field.double() - broad.double())
        innovation.masked_fill_(~valid[:, None], 0.)
        positive = positive + operator.double() @ innovation
        full, _ = signed_potential(directed, valid)
        replay = positive + .5 * (operator.double() @ full)
        if not torch.equal(values[ONE_SIDED_SOFT], replay):
            raise RuntimeError("Audit source does not replay the exact deployed score.")
        protocol = self.protocols[len(members)]
        self.rows.append(dict(sample_key=self.sample_key, protocol=protocol, tile_index=self.tiles[protocol],
            **decomposition(risk, members, canonical, valid, directed, common, operator, positive)))
        self.tiles[protocol] += 1
        return values, diagnostics

    def predict(self, *args, **kwargs):
        return source_predict(*args, **kwargs, methods=(ONE_SIDED_SOFT,), tile_scorer=self.score,
            allowed_methods=METHODS, observation_method=OBSERVATION_MEAN, diagnostic_fields=DIAGNOSTICS)
