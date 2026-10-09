"""Fixed-score expansion interventions, not an alias selector."""
import torch


IMPLEMENTATION = 'frozen-expansion-class-field-attribution-v1-20261005'
METHODS = ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact')


def centered_restorations(old, expanded):
    if (old.ndim != 2 or old.shape != expanded.shape or old.shape[-1] < 2
            or not bool(torch.isfinite(old).all() and torch.isfinite(expanded).all())):
        raise ValueError('Matching finite query/class endpoint scores required.')
    # Fix the otherwise arbitrary common class-logit gauge before attribution.
    old = old.double()-old.double().mean(-1, keepdim=True)
    expanded = expanded.double()-expanded.double().mean(-1, keepdim=True)
    output = {'All20': old, 'All40': expanded}
    for c in range(old.shape[-1]):
        own20 = expanded.clone()
        own20[:, c] = old[:, c]
        rivals20 = old.clone()
        rivals20[:, c] = expanded[:, c]
        output['Own20_Rivals40__'+str(c)] = own20
        output['Own40_Rivals20__'+str(c)] = rivals20
    return output


def common_count_shift(operator, ratio=2.):
    if (operator.ndim != 2 or operator.shape[0] != operator.shape[1]
            or not bool(torch.isfinite(operator).all()) or ratio <= 0):
        raise ValueError('Finite square original reconstruction and positive ratio required.')
    return operator.double().sum(-1, keepdim=True)*operator.new_tensor(ratio, dtype=torch.float64).log()
