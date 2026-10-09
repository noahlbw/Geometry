"""Opt-in natural-task read-strength prior derived from development controls."""
from dataclasses import replace

IMPLEMENTATION='natural-rank-menu-strength3-prior-v1-20261007'


def select_read_strength(profile,decision,policy):
    if policy not in ('retained','natural_strength3'):
        raise ValueError('Declared retained or natural-strength3 policy required.')
    if policy=='retained' or profile is None:
        return profile,decision
    updated=dict(decision)
    if profile.strength==2.:
        profile=replace(profile,strength=3.)
    updated.update(read_policy=policy,selected_strength=profile.strength)
    return profile,updated
