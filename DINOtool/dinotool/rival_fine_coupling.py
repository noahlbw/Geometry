"""Write already-admitted pair evidence into the unchanged Geometry reconstruction."""
import torch


IMPLEMENTATION = 'geometry-physical8-rival-admission-coupled-v1-20261003'
PRIMARY = 'RivalFineHard_Exact'
REPLAY = ('Geometry', 'NoAdmission_Exact', 'FineAliasViewJoint_Exact',
    'ObserverAll20', 'ObserverFinePairHard')
WRITES = {PRIMARY: 'ObserverFinePairHard', 'RivalFineSoft_Exact': 'ObserverFinePairSoft',
    'RivalNative16Hard_Exact': 'ObserverNativePairHard',
    **{'RivalFineRandom'+str(i)+'_Exact': 'ObserverFinePairRandom'+str(i) for i in range(3)}}
RANDOM = tuple('RivalFineRandom'+str(i)+'_Exact' for i in range(3))
METHODS = (*REPLAY, *WRITES, 'RivalFineHard_MeanLogit')
GATE = {'minimum_clean_gain_pp': .1, 'minimum_clean_domain_wins': 5,
    'maximum_clean_protocol_loss_pp': 1., 'clean_above_count_matched_random': list(RANDOM),
    'wrong_parent_gain_required': True, 'no_automatic_full_rollout': True,
    'previous_failed_gates_unchanged': True}


def write_admitted_pairs(baseline, local, operator, observations):
    broad = observations['ObserverAll20'].double()
    if (baseline.shape != local.shape or broad.shape != baseline.shape
            or operator.shape != (len(broad), len(broad))
            or not bool(torch.isfinite(operator).all() and torch.isfinite(baseline).all())):
        raise ValueError('Matching original scores and finite Geometry reconstruction required.')
    output = {}
    for name, source in WRITES.items():
        admitted = observations[source].double()
        if admitted.shape != broad.shape or not bool(torch.isfinite(admitted).all()):
            raise ValueError('Matching finite admitted observer scores required.')
        output[name] = baseline.double()+operator.double() @ (admitted-broad)
    output['RivalFineHard_MeanLogit'] = .5*(local.double()+observations['ObserverFinePairHard'].double())
    return output
