"""Isolate top-two pair actions without changing the frozen fine witness."""
from types import FunctionType

import torch

from . import fine_responsibility_reader as reference
from .rival_competition_admission import posterior_potential


IMPLEMENTATION = 'frozen-fine-witness-top2-action-diagnostic-v1-20261005'
PRIMARY = 'FineResponsibilityTop2_Exact'
FULL = reference.PRIMARY
METHODS = ('Geometry', 'NoAdmission_Exact', 'RivalFineHard_Exact', FULL, PRIMARY)


def restrict_actions(actions, posterior, valid, contenders=2):
    n, classes = posterior.shape
    if (actions.shape != (n, classes, classes) or valid.shape != (n,)
            or not 2 <= contenders <= classes):
        raise ValueError('Matching pair actions, posterior and a valid contender budget required.')
    pairs = posterior.argsort(dim=-1, descending=True, stable=True)[:, :contenders]
    selected = torch.zeros_like(posterior, dtype=torch.bool).scatter_(1, pairs, True)
    mask = selected[:, :, None] & selected[:, None] & valid[:, None, None]
    return actions.masked_fill(~mask, 0.), mask


@torch.inference_mode()
def diagnostic_scores(*inputs, methods=METHODS):
    if not methods or not set(methods).issubset(METHODS):
        raise ValueError('Declared fine-source diagnostic endpoints required.')
    captured = []
    def capture(actions, posterior, valid):
        captured.append((actions, posterior, valid))
        return posterior_potential(actions, posterior, valid)
    original = reference.responsibility_scores.__wrapped__
    reader = FunctionType(original.__code__, dict(original.__globals__, posterior_potential=capture),
                          original.__name__, original.__defaults__)
    requested = tuple(m for m in methods if m != PRIMARY)
    if PRIMARY in methods and FULL not in requested:
        requested += (FULL,)
    values, risk, diagnostics = reader(*inputs, methods=requested)
    if PRIMARY in methods:
        if len(captured) != 1:
            raise RuntimeError('Exactly one frozen fine-only action source required.')
        action, posterior, valid = captured[0]
        restricted, mask = restrict_actions(action, posterior, valid)
        potential, consistency = posterior_potential(restricted, posterior, valid)
        local, operator, broad = inputs[:3]
        base = local.double() + operator.double() @ (broad.double() - local.double())
        values[PRIMARY] = base + operator.double() @ potential
        denominator = action.abs().sum()
        diagnostics.update(consistency,
            same_fine_observations_and_alias_allocation=True,
            diagnostic_dense_reader_retained=True,
            full_action_l1=float(denominator),
            top2_action_l1_fraction=float(restricted.abs().sum() / denominator) if denominator > 0 else 1.,
            posterior_top2_mass_mean=float(posterior.sort(-1, descending=True).values[valid, :2].sum(-1).mean())
                if bool(valid.any()) else 1.,
            retained_pair_posterior_mass_mean=float((posterior[:, :, None] * posterior[:, None] * mask).sum((-1, -2))[valid].mean())
                if bool(valid.any()) else 1.)
    return {m: values[m] for m in methods}, risk, diagnostics
