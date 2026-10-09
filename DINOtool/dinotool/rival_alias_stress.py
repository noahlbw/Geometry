"""Diagnostics only for a frozen selector's uncontradicted alias evidence."""
import torch


SCENARIOS = ('k20', 'k30', 'k40', 'wrong_parent', 'paraphrase')
METHODS = ('NoAdmission_Exact', 'RivalFineHard_Exact', 'FineRivalProjected_Exact', 'FineBudgetOnly_Exact')
IMPLEMENTATION = 'frozen-rival-projected-vocabulary-stress-v1-20261005'


def evidence_audit(broad_positive, fine_nonnegative, rejected, parents, canonical, known,
                   alias_subset=None, query_target=None):
    if (broad_positive.ndim != 3 or fine_nonnegative.shape != broad_positive.shape
            or rejected.shape != broad_positive.shape or parents.shape != broad_positive.shape[1:2]
            or canonical.shape != broad_positive.shape[2:] or known.shape != broad_positive.shape[:1]
            or any(x.dtype != torch.bool for x in (broad_positive, fine_nonnegative, rejected, known))):
        raise ValueError('Matching boolean source flags and frozen class identities required.')
    aliases = torch.ones(len(parents), dtype=torch.bool, device=parents.device) if alias_subset is None else alias_subset.clone()
    if aliases.shape != parents.shape or aliases.dtype != torch.bool:
        raise ValueError('Boolean audit alias subset required.')
    aliases[canonical] = False
    rivals = torch.arange(broad_positive.shape[-1], device=parents.device)
    valid = known[:, None, None] & aliases[None, :, None] & (parents[:, None] != rivals)[None]
    broad = broad_positive & valid
    blind = broad & fine_nonnegative
    def fraction(count, total):
        return float(count/total) if total else None
    result = {'audited_comparisons': int(valid.sum()), 'broad_advantage_comparisons': int(broad.sum()),
        'uncontradicted_advantage_comparisons': int(blind.sum()),
        'uncontradicted_given_broad_fraction': fraction(int(blind.sum()), int(broad.sum())),
        'uncontradicted_but_rejected_comparisons': int((blind & rejected).sum()),
        'retained_fraction': fraction(int((valid & ~rejected).sum()), int(valid.sum()))}
    if query_target is not None:
        if query_target.shape != known.shape:
            raise ValueError('One audit-only centre target per query required.')
        target_valid = known & (query_target >= 0) & (query_target < len(canonical))
        rival_is_target = rivals[None] == query_target[:, None]
        wrong_parent = valid & target_valid[:, None, None] & rival_is_target[:, None]
        wrong_broad = broad_positive & wrong_parent
        wrong_blind = wrong_broad & fine_nonnegative
        result['target_audit'] = {'wrong_parent_comparisons': int(wrong_parent.sum()),
            'wrong_parent_broad_advantage': int(wrong_broad.sum()),
            'wrong_parent_uncontradicted_advantage': int(wrong_blind.sum()),
            'wrong_parent_uncontradicted_fraction': fraction(int(wrong_blind.sum()), int(wrong_parent.sum())),
            'uncontradicted_given_wrong_parent_broad_fraction': fraction(int(wrong_blind.sum()), int(wrong_broad.sum()))}
    return result
