"""Local-query-conditioned routing over a single whole-image slot bank.

The wide image is encoded once.  Its raw patch assignments form a compact
slot bank, while each already-computed local Geometry patch token selects the
slot mixture used for that local output tile.  This changes the wide donor
routing without adding an RGB encoder or a learned parameter.
"""
import torch
import torch.nn.functional as F

from .context_memory_observer import context_assignments, read_context_head


@torch.inference_mode()
def slot_prototypes(raw, assignment, features):
    """Return normalized raw centers and semantic Value prototypes."""
    if raw.ndim != 3 or assignment.ndim != 3 or features.ndim != 3:
        raise ValueError('Expected batch-one raw, assignment and semantic tensors.')
    if raw.shape[:2] != assignment.shape[:2] or features.shape[:2] != raw.shape[:2]:
        raise ValueError('Slot tensors must share the patch lattice.')
    with torch.autocast(device_type=raw.device.type, enabled=False):
        p = assignment.float()
        mass = p.sum(1).clamp_min(1e-20)
        raw_center = (p.transpose(1, 2) @ F.normalize(raw.float(), dim=-1))/mass[..., None]
        semantic = (p.transpose(1, 2) @ features.float())/mass[..., None]
        return F.normalize(raw_center, dim=-1), F.normalize(semantic, dim=-1)


@torch.inference_mode()
def slot_coordinate_prototypes(assignment, grid_hw):
    """Return normalized image coordinates carried by each soft slot."""
    if assignment.ndim != 3 or assignment.shape[0] != 1:
        raise ValueError('Expected batch-one slot assignments.')
    height, width = grid_hw
    if height < 1 or width < 1 or assignment.shape[1] != height * width:
        raise ValueError('Slot assignment count must match grid_hw.')
    with torch.autocast(device_type=assignment.device.type, enabled=False):
        yy = (torch.arange(height, device=assignment.device, dtype=torch.float32)+.5)/height*2-1
        xx = (torch.arange(width, device=assignment.device, dtype=torch.float32)+.5)/width*2-1
        y, x = torch.meshgrid(yy, xx, indexing='ij')
        coordinates = torch.stack((x, y), -1).reshape(1, -1, 2)
        mass = assignment.float().sum(1).clamp_min(1e-20)
        return assignment.float().transpose(1, 2) @ coordinates / mass[..., None]


@torch.inference_mode()
def query_slot_features(query_raw, centers, values, baseline=None, *, temperature=.1, beta=.5,
                        routing='fixed', query_coords=None, slot_coords=None,
                        spatial_weight=0., spatial_sigma=.75,
                        secondary_query=None, secondary_centers=None, hybrid_weight=.5,
                        slot_mass=None, mass_weight=0., transport='absolute', residual_anchor=None):
    """Select semantic slot prototypes with a local raw-token query.

    ``beta`` controls the amount of routed context.  ``transport='absolute'``
    is the incumbent convex interpolation between the coordinate baseline and
    the selected slot values.  Residual transports keep that baseline intact
    and move only the slot value minus ``residual_anchor``; this is useful when
    the compressed slots are treated as context corrections rather than a
    replacement feature.
    """
    if query_raw.ndim != 3 or centers.ndim != 3 or values.ndim != 3:
        raise ValueError('Expected batch-one query and slot tensors.')
    if centers.shape[:2] != values.shape[:2] or query_raw.shape[0] != centers.shape[0]:
        raise ValueError('Query and slot batch dimensions do not match.')
    if temperature <= 0 or not 0 <= beta <= 1:
        raise ValueError('Positive temperature and beta in [0,1] required.')
    if routing not in ('fixed', 'entropy', 'max'):
        raise ValueError('routing must be fixed, entropy, or max.')
    if spatial_weight and (query_coords is None or slot_coords is None):
        raise ValueError('Spatial routing requires query_coords and slot_coords.')
    if (secondary_query is None) != (secondary_centers is None):
        raise ValueError('Secondary query and centers must be provided together.')
    if secondary_query is not None and not 0 <= hybrid_weight <= 1:
        raise ValueError('hybrid_weight must be in [0,1].')
    if mass_weight and slot_mass is None:
        raise ValueError('Mass-calibrated routing requires slot_mass.')
    if transport not in ('absolute', 'global_residual', 'coordinate_residual'):
        raise ValueError('transport must be absolute, global_residual, or coordinate_residual.')
    if transport != 'absolute' and (baseline is None or residual_anchor is None):
        raise ValueError('Residual transport requires baseline and residual_anchor.')
    if residual_anchor is not None and residual_anchor.shape != values.shape:
        raise ValueError('Residual anchor must match slot values.')
    if spatial_sigma <= 0:
        raise ValueError('spatial_sigma must be positive.')
    with torch.autocast(device_type=query_raw.device.type, enabled=False):
        weights = (F.normalize(query_raw.float(), dim=-1) @ centers.float().transpose(-1, -2))
        if secondary_query is not None:
            secondary = (F.normalize(secondary_query.float(), dim=-1)
                         @ secondary_centers.float().transpose(-1, -2))
            weights = float(hybrid_weight)*weights + (1.-float(hybrid_weight))*secondary
        weights = weights/temperature
        if mass_weight:
            if slot_mass.ndim != 2 or slot_mass.shape != centers.shape[:2]:
                raise ValueError('slot_mass must have shape [B,K].')
            prior = (slot_mass.float()/slot_mass.float().sum(-1, keepdim=True)).clamp_min(1e-20).log()
            weights = weights + float(mass_weight)*prior[:, None, :]
        if spatial_weight:
            if query_coords.shape[:2] != query_raw.shape[:2] or slot_coords.shape[:2] != centers.shape[:2]:
                raise ValueError('Spatial coordinates must match query and slot lattices.')
            distance = (query_coords.float()[:, :, None, :] - slot_coords.float()[:, None, :, :]).square().sum(-1)
            weights = weights - float(spatial_weight) * distance / (2.*float(spatial_sigma)**2)
        probability = weights.softmax(-1)
        if transport == 'absolute':
            routed = probability @ values.float()
        else:
            routed = probability @ (values.float() - residual_anchor.float())
        if baseline is None:
            return F.normalize(routed, dim=-1)
        if baseline.shape != routed.shape:
            raise ValueError('Baseline and routed query shapes must match.')
        if routing == 'fixed':
            local_beta = torch.full_like(weights[..., :1], float(beta))
        elif routing == 'max':
            uniform = 1.0 / probability.shape[-1]
            confidence = ((probability.max(-1).values - uniform) / (1. - uniform)).clamp(0., 1.)
            local_beta = float(beta) * confidence[..., None]
        elif routing == 'entropy':
            entropy = -(probability * probability.clamp_min(1e-20).log()).sum(-1)
            normalized = entropy / torch.log(torch.tensor(float(probability.shape[-1]), device=entropy.device))
            local_beta = float(beta) * (1. - normalized).clamp(0., 1.)[..., None]
        if transport == 'absolute':
            mixed = (1-local_beta)*baseline.float()+local_beta*routed
        else:
            mixed = baseline.float()+local_beta*routed
        return F.normalize(mixed.float(), dim=-1)


@torch.inference_mode()
def _sample_global(features, grid_hw, source_hw, top, left):
    """Sample a full-image feature grid at local 32x32 patch centers."""
    gh, gw = grid_hw
    h, w = source_hw
    yy = (top+(torch.arange(32, device=features.device, dtype=torch.float32)+.5)*16)/h*2-1
    xx = (left+(torch.arange(32, device=features.device, dtype=torch.float32)+.5)*16)/w*2-1
    y, x = torch.meshgrid(yy, xx, indexing='ij')
    image = features.transpose(1, 2).reshape(1, -1, gh, gw)
    grid = torch.stack((x, y), -1)[None]
    sampled = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)
    return sampled.permute(0, 2, 3, 1).reshape(1, 32*32, -1)


@torch.inference_mode()
def _sample_grid_points(features, grid_hw, coordinates):
    """Sample a normalized feature grid at a compact set of [-1, 1] points."""
    if coordinates.ndim != 3 or coordinates.shape[-1] != 2:
        raise ValueError('Expected [B,K,2] normalized coordinates.')
    gh, gw = grid_hw
    image = features.transpose(1, 2).reshape(features.shape[0], -1, gh, gw)
    grid = coordinates[:, None].float()
    sampled = F.grid_sample(image.float(), grid, mode='bilinear', padding_mode='border', align_corners=False)
    return sampled[:, :, 0].transpose(1, 2)


@torch.inference_mode()
def query_slot_wide_observations(image, geometry, observer, strengths, *, wide_policy='long448',
                                 temperature=.1, beta=.5, reconstruction=None,
                                 shuffle_query=False, routing='fixed', spatial_weight=0.,
                                 spatial_sigma=.75, query_mode='raw', hybrid_weight=.5,
                                 mass_weight=0., transport='absolute'):
    """Build local-tile wide features from one global slot memory."""
    # Keep the algebraic slot functions importable in CPU-only test and
    # analysis environments.  These deployment helpers bring in OpenCV and
    # the full VIP adapter, which are only needed for an actual image pass.
    from .compact_deployment import _iter_local_observations
    from .bounded_physical_coupling import resize_geometry
    from .context_memory_deployment import memory_rgb
    if query_mode not in ('raw', 'semantic', 'hybrid'):
        raise ValueError('query_mode must be raw, semantic, or hybrid.')
    if transport not in ('absolute', 'global_residual', 'coordinate_residual'):
        raise ValueError('transport must be absolute, global_residual, or coordinate_residual.')
    resized = resize_geometry(image)
    h, w = resized.shape[-2:]
    source = dict(local=[], size=(h, w), output_size=tuple(image.shape[-2:]))
    rgb, (wide_h, wide_w) = memory_rgb(image, wide_policy=wide_policy)
    ph, pw = rgb.shape[-2:]
    raw = observer.raw_patch_features(rgb)
    assignment = context_assignments(raw, observer.config)
    with observer.backbone._autocast():
        semantic = read_context_head(observer.backbone.model.visual_model.head, raw, assignment)
    centers, values = slot_prototypes(raw, assignment, semantic)
    if query_mode == 'semantic':
        centers = values
    slot_coords = slot_coordinate_prototypes(assignment, (ph//16, pw//16))
    global_grid = semantic
    if transport == 'global_residual':
        anchor = values.mean(1, keepdim=True).expand_as(values)
    elif transport == 'coordinate_residual':
        anchor = _sample_grid_points(global_grid, (ph//16, pw//16), slot_coords)
    else:
        anchor = None
    slot_mass = assignment.sum(1)
    slot_count = int(assignment.shape[-1])
    # The global raw lattice and assignment matrix have served their purpose;
    # retain only the compact slot bank before encoding local tiles.
    del raw, assignment
    wide = []
    for tile in _iter_local_observations(resized, (h, w), geometry, strengths,
                                         reconstruction=reconstruction, include_raw=True):
        raw_query = tile.pop('raw')
        semantic_query = tile['features'][strengths[0]].unsqueeze(0)
        if query_mode == 'raw':
            q, query_centers, secondary_query, secondary_centers = raw_query, centers, None, None
        elif query_mode == 'semantic':
            q, query_centers, secondary_query, secondary_centers = semantic_query, centers, None, None
        else:
            q, query_centers, secondary_query, secondary_centers = raw_query, centers, semantic_query, values
        if shuffle_query:
            # Matched control: preserve the local raw-token multiset while
            # breaking the query-to-output-position correspondence.
            order = torch.roll(torch.arange(q.shape[1], device=q.device), q.shape[1]//2)
            q = q[:, order]
        baseline = _sample_global(global_grid, (ph//16, pw//16), (h, w), tile['top'], tile['left'])
        yy = (tile['top']+(torch.arange(32, device=q.device, dtype=torch.float32)+.5)*16)/h*2-1
        xx = (tile['left']+(torch.arange(32, device=q.device, dtype=torch.float32)+.5)*16)/w*2-1
        y, x = torch.meshgrid(yy.clamp(-1, 1), xx.clamp(-1, 1), indexing='ij')
        query_coords = torch.stack((x, y), -1).reshape(1, 32*32, 2)
        features = query_slot_features(q, centers, values, baseline,
                                       temperature=temperature, beta=beta, routing=routing,
                                       query_coords=query_coords, slot_coords=slot_coords,
                                       spatial_weight=spatial_weight, spatial_sigma=spatial_sigma,
                                       secondary_query=secondary_query, secondary_centers=secondary_centers,
                                       hybrid_weight=hybrid_weight, slot_mass=slot_mass,
                                       mass_weight=mass_weight, transport=transport,
                                       residual_anchor=anchor)
        # The query is only needed for this tile; it is never retained in the
        # source list, so the number of local views does not grow the temporary
        # memory footprint.
        ah, aw = min(512, h-tile['top']), min(512, w-tile['left'])
        wide.append(dict(top=tile['top'], left=tile['left'], ah=ah, aw=aw,
                         grid_hw=(32, 32), output_hw=(512, 512), features=features))
        source['local'].append(tile)
    return dict(source, wide=wide, wide_size=(h, w), query_slot_count=slot_count,
                query_wide_size=(wide_h, wide_w), query_transport=transport)
