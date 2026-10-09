"""Bounded temporaries for the SAME frozen alias equations on large taxonomies."""
from contextlib import contextmanager
import math

import torch

from .calibrated_competitive_alias import profiled_logits
from .fine_alias_view import CONFIG
from .native_query_alias import native_risk as original_risk
from .rival_alias_fast import CachedCrop, sampled_cached as original_sample
from .stratified_soft_alias import crop_stencil

QUERY_CHUNK = 32


class LazyMargins:
    def __init__(self,evidence,beta):
        self.alias=evidence.flatten(1)
        self.rival=(beta*evidence).logsumexp(-1)/beta-math.log(evidence.shape[-1])/beta
        self.shape=(len(evidence),self.alias.shape[-1],self.rival.shape[-1])

    def __getitem__(self,indices):
        # Identical subtraction, on the same selected source entries.
        return self.alias[indices][...,None]-self.rival[indices][...,None,:]


def cached_observations(crops,count,coordinates,image_size,members,beta=CONFIG.beta):
    result=[]
    for crop in crops:
        evidence=profiled_logits(crop,members)
        indices,coefficients=crop_stencil(crop,count,coordinates,image_size)
        result.append(CachedCrop(evidence,LazyMargins(evidence,beta),indices,coefficients))
    return result


def sampled_observations(observations,coordinates):
    if not observations:raise ValueError('At least one observation is required.')
    result=coordinates.new_zeros((len(coordinates),*observations[0].margins.shape[1:]))
    for start in range(0,len(coordinates),QUERY_CHUNK):
        sl=slice(start,start+QUERY_CHUNK)
        part=tuple(CachedCrop(c.evidence,c.margins,c.indices[sl],c.coefficients[sl]) for c in observations)
        result[sl]=original_sample(part,coordinates[sl])
    return result


def chunked_risk(broad,native,supported,parents,canonical,known,config=CONFIG):
    result=torch.empty_like(broad)
    for start in range(0,len(broad),QUERY_CHUNK):
        sl=slice(start,start+QUERY_CHUNK)
        result[sl]=original_risk(broad[sl],native[sl],supported[sl],parents,canonical,known[sl],config)
    return result


@contextmanager
def execution():
    """Process-local single-thread execution substitution, restored after call.

    The source predictors expose cached observation/risk globals. No backbone,
    text, weights, writer, readout, geometry or model configuration is changed.
    """
    from . import shared_local_alias as light
    from . import one_sided_alias_stream as retained
    replacements=((light,'cache_observations',cached_observations),
        (light,'sampled_cached',sampled_observations),(light,'native_risk',chunked_risk),
        (retained,'sampled_cached',sampled_observations),(retained,'native_risk',chunked_risk))
    previous=[(module,name,getattr(module,name)) for module,name,_ in replacements]
    try:
        for module,name,value in replacements:setattr(module,name,value)
        yield
    finally:
        for module,name,value in previous:setattr(module,name,value)
