"""Experimental spatially anchored residual readout of whole-image slots.

This opt-in module reuses the frozen global slot head and local DonorBefore
iterator. No deployment defaults are changed. The coordinate map intentionally
matches ``query_slot_memory._sample_global`` for all controls: normalized local
coordinates span the padded global patch lattice, with border extrapolation.
It does not introduce a corrected padding/valid-region map; any incumbent
geometric approximation is shared by the compared readers.
"""
import torch
import torch.nn.functional as F

from .context_memory_observer import context_assignments, read_context_head
from .query_slot_memory import _sample_global, query_slot_features, slot_prototypes


READ_MODES = ('coordinate', 'legacy', 'anchored', 'no_prior', 'no_reference',
              'absolute_mix', 'shuffled')


@torch.inference_mode()
def anchored_slot_features(query_raw, reference_raw, centers, values, baseline,
                           spatial_prior, *, temperature=.1, beta=.5,
                           mode='anchored'):
    """Return normalized wide features using an explicit spatial baseline.

    ``anchored`` updates pi with (normalize(q)-normalize(qbar)) C^T/tau,
    then adds beta*(posterior-pi)V to the unnormalized sampled baseline.
    ``no_prior`` substitutes uniform pi in BOTH the posterior and residual.
    ``no_reference`` uses normalize(q) instead of the descriptor difference.
    ``absolute_mix`` uses the anchored posterior but convex feature mixing.
    ``shuffled`` rolls only q, leaving qbar, pi and the baseline fixed.
    ``legacy`` calls the incumbent fixed raw-query reader without changes.
    ``coordinate`` and beta=0 return the normalized baseline before any
    routing or query normalization; unused inputs may consequently be None.

    centers/values are the normalized prototypes from ``slot_prototypes``.
    The positivity floor on pi is a numerical constant, not a tuned gate.
    """
    if mode not in READ_MODES:
        raise ValueError('Unknown anchored slot readout mode: ' + str(mode))
    if temperature <= 0 or not 0 <= beta <= 1:
        raise ValueError('Positive temperature and beta in [0,1] required.')
    if baseline is None or baseline.ndim != 3:
        raise ValueError('Expected baseline with shape [B,N,D_sem].')
    if mode == 'coordinate' or beta == 0:
        with torch.autocast(device_type=baseline.device.type, enabled=False):
            return F.normalize(baseline.float(), dim=-1)
    if mode == 'legacy':
        return query_slot_features(query_raw, centers, values, baseline,
                                   temperature=temperature, beta=beta)
    if (query_raw is None or centers is None or values is None
            or query_raw.ndim != 3 or centers.ndim != 3 or values.ndim != 3):
        raise ValueError('Expected query [B,N,D_raw], centers and values [B,K,D].')
    if (centers.shape[:2] != values.shape[:2] or query_raw.shape[0] != centers.shape[0]
            or query_raw.shape[-1] != centers.shape[-1]
            or baseline.shape[:2] != query_raw.shape[:2]
            or baseline.shape[-1] != values.shape[-1] or centers.shape[1] < 1):
        raise ValueError('Query, slot and baseline shapes do not match.')
    if mode != 'no_reference' and (reference_raw is None
                                   or reference_raw.shape != query_raw.shape):
        raise ValueError('The coordinate raw reference must match the local query.')
    if mode != 'no_prior' and (spatial_prior is None
            or spatial_prior.shape != (*query_raw.shape[:2], centers.shape[1])):
        raise ValueError('Expected coordinate spatial prior with shape [B,N,K].')
    with torch.autocast(device_type=query_raw.device.type, enabled=False):
        query = F.normalize(query_raw.float(), dim=-1)
        if mode == 'shuffled':
            query = query.roll(query.shape[1]//2, dims=1)
        evidence = query
        if mode != 'no_reference':
            evidence = evidence-F.normalize(reference_raw.float(), dim=-1)
        evidence = (evidence @ centers.float().transpose(-1, -2))/float(temperature)
        if mode == 'no_prior':
            prior = torch.full_like(evidence, 1./centers.shape[1])
        else:
            prior = spatial_prior.float().clamp_min(1e-20)
            prior = prior/prior.sum(-1, keepdim=True)
        posterior = (prior.log()+evidence).softmax(-1)
        if mode == 'absolute_mix':
            mixed = (1.-float(beta))*baseline.float()+float(beta)*(posterior @ values.float())
        else:
            mixed = baseline.float()+float(beta)*((posterior-prior) @ values.float())
        return F.normalize(mixed, dim=-1)


@torch.inference_mode()
def anchored_slot_wide_observations(image, geometry, observer, strengths, *,
                                    wide_policy='long448', temperature=.1, beta=.5,
                                    reconstruction=None, mode='anchored', modes=None,
                                    audit=False):
    """Encode global/local views once, optionally evaluating multiple readers.

    The experiment supplies the frozen K24, tau=.1, five-iteration, uncentered
    observer config. It is reused verbatim so every reader sees identical head
    features. Local observations are yielded by the incumbent iterator with
    unchanged strengths, reconstruction and geometry objects (DonorBefore).

    If ``modes`` is None, return the usual observation source for ``mode``.
    Otherwise return ``{name: source}``; all sources share identical local
    rows and operators but have independent wide-feature lists. Multi-mode
    feature retention is for accuracy evaluation, NOT efficiency measurement.
    Costs must use one mode in an independent process. Optional audit checks
    execute outside that measurement and synchronize when reading scalars.
    """
    from .compact_deployment import _iter_local_observations
    from .bounded_physical_coupling import resize_geometry
    from .context_memory_deployment import memory_rgb

    names = [mode] if modes is None else list(modes)
    if (not names or len(set(names)) != len(names)
            or any(name not in READ_MODES for name in names)):
        raise ValueError('Provide nonempty unique anchored readout modes.')
    if temperature <= 0 or not 0 <= beta <= 1:
        raise ValueError('Positive temperature and beta in [0,1] required.')
    routed = beta != 0 and any(name != 'coordinate' for name in names)
    relative = routed and any(name not in ('coordinate', 'legacy') for name in names)
    resized = resize_geometry(image)
    h, w = resized.shape[-2:]
    rgb, (wide_h, wide_w) = memory_rgb(image, wide_policy=wide_policy)
    ph, pw = rgb.shape[-2:]
    grid_hw = (ph//16, pw//16)
    raw = observer.raw_patch_features(rgb)
    assignment = context_assignments(raw, observer.config)
    with observer.backbone._autocast():
        semantic = read_context_head(observer.backbone.model.visual_model.head, raw, assignment)
    slot_count = int(assignment.shape[-1])
    centers, values = slot_prototypes(raw, assignment, semantic) if routed else (None, None)
    global_raw = F.normalize(raw.float(), dim=-1) if relative else None
    global_prior = assignment if relative else None
    del raw, assignment

    local = []
    sources = {name: dict(local=local, size=(h, w), output_size=tuple(image.shape[-2:]),
                         wide=[], wide_size=(h, w), query_slot_count=slot_count,
                         query_wide_size=(wide_h, wide_w), query_transport=name,
                         anchored_readout=name) for name in names}
    checks = dict(legacy_match=True, coordinate_beta0_match=True,
                  shared_local_identity=True, checked_tiles=0,
                  coordinate_mapping='incumbent _sample_global over padded global lattice')
    for tile in _iter_local_observations(resized, (h, w), geometry, strengths,
                                         reconstruction=reconstruction, include_raw=routed):
        query = tile.pop('raw') if routed else None
        baseline = _sample_global(semantic, grid_hw, (h, w), tile['top'], tile['left'])
        reference = (_sample_global(global_raw, grid_hw, (h, w), tile['top'], tile['left'])
                     if relative else None)
        prior = (_sample_global(global_prior, grid_hw, (h, w), tile['top'], tile['left'])
                 if relative else None)
        outputs = {}
        for name in names:
            features = anchored_slot_features(query, reference, centers, values, baseline, prior,
                                               temperature=temperature, beta=beta, mode=name)
            outputs[name] = features
            ah, aw = min(512, h-tile['top']), min(512, w-tile['left'])
            sources[name]['wide'].append(dict(top=tile['top'], left=tile['left'], ah=ah, aw=aw,
                grid_hw=(32, 32), output_hw=(512, 512), features=features))
        local.append(tile)
        if audit:
            coordinate = anchored_slot_features(None, None, None, None, baseline, None,
                                                 mode='coordinate')
            beta0 = anchored_slot_features(None, None, None, None, baseline, None, beta=0.)
            checks['coordinate_beta0_match'] &= torch.equal(coordinate, beta0)
            if routed:
                legacy = anchored_slot_features(query, None, centers, values, baseline, None,
                                                temperature=temperature, beta=beta, mode='legacy')
                incumbent = query_slot_features(query, centers, values, baseline,
                                                temperature=temperature, beta=beta)
                checks['legacy_match'] &= torch.equal(legacy, incumbent)
            checks['shared_local_identity'] &= all(source['local'][-1] is tile
                                                    for source in sources.values())
            checks['checked_tiles'] += 1
    if audit:
        if not all(checks[key] for key in ('legacy_match', 'coordinate_beta0_match',
                                           'shared_local_identity')):
            raise AssertionError('Anchored readout integration audit failed: ' + str(checks))
        for source in sources.values():
            source['anchored_audit'] = dict(checks)
    return sources[names[0]] if modes is None else sources
