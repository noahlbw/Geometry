"""Re-encode Geometry-defined visual units while retaining fine descriptor detail."""
from dataclasses import asdict, dataclass
import time

import numpy as np
import torch
import torch.nn.functional as F

from .matched_readout_controls import run_head
from .tcpr import _intervened_head


IMPLEMENTATION = "geometry-semantic-unit-detail-lift-v1-196-20261002"
PRIMARY = "Geometry_UnitDetail"
BASELINES = ("Geometry", "SCLIP_Two", "VIPProxy_Two")
METHODS = (*BASELINES, "UnitOnly", "MeanLogit_Unit", "Spatial_UnitDetail", PRIMARY)


@dataclass(frozen=True)
class UnitConfig:
    grid_side: int = 14

    @property
    def units(self):
        return self.grid_side ** 2

    def signature(self):
        return {"unit_readout": asdict(self), "implementation": IMPLEMENTATION,
                "grouping": "4-connected minimum spanning forest of symmetric original Geometry edges",
                "budget": "196 semantic units, fixed from author nominal224/patch16 configuration",
                "pool": "within-unit normalized original Geometry incoming support",
                "coarse_relation": "row-normalized S G U, no recomputation from pooled descriptors",
                "head": "original two frozen Geometry blocks on pooled backbone tokens and native starting prefixes",
                "lift": "Y_final = Y_Geometry + U (Y_units - S Y_Geometry), before descriptor L2 normalization",
                "detail": "within-unit projected descriptor differences retained before final L2 normalization",
                "identity": "all singleton/no valid units returns exact original Geometry",
                "padding": "invalid patches receive no correction",
                "text": "unchanged20 aliases/six RS templates/normalized LME .07",
                "view": "native512/128/Hann probability blending; no extra RGB crop or backbone",
                "limits": "same frozen semantic source; nominal token count is not proof of distribution alignment"}


def unit_labels(geometry, valid, shape, config=UnitConfig(), *, spatial=False):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components, minimum_spanning_tree

    h, w = shape
    n = h * w
    if (config.grid_side < 1 or geometry.shape != (n, n) or valid.shape != (n,)
            or valid.dtype != torch.bool or not bool(torch.isfinite(geometry).all())
            or bool((geometry < 0).any())):
        raise ValueError("Invalid Geometry, image validity or unit budget.")
    ids = np.flatnonzero(valid.detach().cpu().numpy())
    labels = torch.full((n,), -1, dtype=torch.long, device=geometry.device)
    if not len(ids):
        return labels
    if len(ids) <= config.units:
        labels[torch.as_tensor(ids, device=labels.device)] = torch.arange(len(ids), device=labels.device)
        return labels
    if spatial:
        regions = (ids // w * config.grid_side // h) * config.grid_side + (ids % w * config.grid_side // w)
        _, components = np.unique(regions, return_inverse=True)
    else:
        grid = np.arange(n).reshape(h, w)
        a = np.concatenate((grid[:-1].ravel(), grid[:, :-1].ravel()))
        b = np.concatenate((grid[1:].ravel(), grid[:, 1:].ravel()))
        keep = np.isin(a, ids) & np.isin(b, ids)
        a, b = a[keep], b[keep]
        index = np.full(n, -1, dtype=np.int64)
        index[ids] = np.arange(len(ids))
        # Use established sparse MST/connected-components algorithms; no
        # semantic prediction or target mask participates in the partition.
        source = torch.as_tensor(a, device=geometry.device)
        target = torch.as_tensor(b, device=geometry.device)
        costs = (-.5 * (geometry[source, target].float().clamp_min(1e-30).log()
                       + geometry[target, source].float().clamp_min(1e-30).log())).clamp_min(1e-12)
        edges = coo_matrix((costs.cpu().numpy(), (index[a], index[b])), shape=(len(ids), len(ids))).tocsr()
        tree = minimum_spanning_tree(edges).tocoo()
        order = np.argsort(tree.data, kind="stable")[:max(0, len(ids) - config.units)]
        forest = coo_matrix((np.ones(len(order)), (tree.row[order], tree.col[order])), shape=edges.shape)
        _, components = connected_components(forest, directed=False)
    labels[torch.as_tensor(ids, device=labels.device)] = torch.as_tensor(components, device=labels.device, dtype=torch.long)
    return labels


def unit_operators(geometry, labels):
    if geometry.shape != (len(labels), len(labels)) or labels.dtype != torch.long:
        raise ValueError("Geometry and unit labels differ.")
    groups = torch.unique(labels[labels >= 0], sorted=True)
    if not len(groups):
        return geometry.new_zeros((0, len(labels)), dtype=torch.float32), geometry.new_zeros((len(labels), 0), dtype=torch.float32)
    membership = (labels[:, None] == groups[None]).float()
    if not torch.equal(groups, torch.arange(len(groups), device=labels.device)):
        raise ValueError("Unit labels must be contiguous from zero.")
    with torch.autocast(geometry.device.type, enabled=False):
        pooled_edges = membership.T @ geometry.float()
        support = pooled_edges * membership.T
        normalizer = support.sum(-1, keepdim=True)
        if bool((normalizer <= 0).any()):
            raise ValueError("Unit has no internal Geometry evidence.")
        pool = support / normalizer
    return pool, membership


def detail_lift(fine, coarse, pool, membership):
    if (fine.ndim != 2 or coarse.shape != (pool.shape[0], fine.shape[-1])
            or pool.shape[1] != fine.shape[0] or membership.shape != pool.T.shape):
        raise ValueError("Incompatible fine/unit descriptors and support operators.")
    if not all(bool(torch.isfinite(x).all()) for x in (fine, coarse, pool, membership)):
        raise ValueError("Nonfinite unit reconstruction input.")
    with torch.autocast(fine.device.type, enabled=False):
        return fine.float() + membership.float() @ (coarse.float() - pool.float() @ fine.float())


def reencode_units(head, prepared, labels, fine):
    pool, membership = unit_operators(prepared.geometry_patch_conditional[0], labels)
    valid = labels >= 0
    if len(pool) == 0 or len(pool) == int(valid.sum()):
        return fine.clone(), fine.clone(), {"units": float(len(pool)), "singleton_fraction": 1.,
                                           "unit_mean_max_error": 0., "invalid_displacement_max_error": 0.}
    tokens = prepared.backbone_tokens
    prefix = prepared.prefix_tokens
    with torch.autocast(tokens.device.type, enabled=False):
        patches = pool @ tokens[0, prefix:].float()
        coarse_relation = pool @ prepared.geometry_patch_conditional[0].float() @ membership
        coarse_relation = coarse_relation / coarse_relation.sum(-1, keepdim=True).clamp_min(1e-30)
    compressed = torch.cat((tokens[:, :prefix], patches[None].to(tokens.dtype)), 1)
    coarse = _intervened_head(head, compressed, coarse_relation[None].to(tokens.dtype),
                             prefix, prepared.block_index, 2, "preserve")[0, prefix:].float()
    lifted = detail_lift(fine[0], coarse, pool, membership)
    # Invalid positions retain the original fine descriptor in both controls.
    with torch.autocast(tokens.device.type, enabled=False):
        only = torch.where(valid[:, None], membership @ coarse, fine[0].float())
        counts = membership.sum(0)
        error = float((pool @ lifted - coarse).abs().max())
        invalid_error = float((lifted - fine[0].float()).masked_fill(valid[:, None], 0.).abs().max())
    if error > 2e-4 or invalid_error != 0. or not bool(torch.isfinite(lifted).all()):
        raise RuntimeError("Unit mean reconstruction, padding or finite invariant failed.")
    return lifted[None], only[None], {"units": float(len(pool)),
        "singleton_fraction": float((counts == 1).float().mean()), "unit_mean_max_error": error,
        "invalid_displacement_max_error": invalid_error,
        "unit_maximum_patches": float(counts.max()), "mean_abs_descriptor_displacement": float((lifted-fine[0].float()).abs().mean())}


@torch.inference_mode()
def read_unit_geometry(geometry, prepared, valid, *, config=UnitConfig(), controls=True, singleton=False):
    if prepared.backbone_tokens.shape[0] != 1 or valid.shape != (1, prepared.raw_patch_tokens.shape[1]):
        raise ValueError("Unit readout uses one prepared window and image-only validity.")
    started = time.perf_counter()
    relation = prepared.geometry_patch_conditional[0]
    labels = (torch.arange(len(relation), device=relation.device).masked_fill(~valid[0], -1)
              if singleton else unit_labels(relation, valid[0], (prepared.grid_height, prepared.grid_width), config))
    if singleton:
        labels[valid[0]] = torch.arange(int(valid.sum()), device=labels.device)
    partition_time = time.perf_counter() - started
    head = geometry.backbone.model.visual_model.head
    with geometry.backbone._autocast():
        fine = _intervened_head(head, prepared.backbone_tokens, prepared.geometry_patch_conditional,
                               prepared.prefix_tokens, prepared.block_index, 2, "preserve")[:, prepared.prefix_tokens:].float()
        primary, only, diagnostics = reencode_units(head, prepared, labels, fine)
        features = {"Geometry": prepared.geometry_projected, PRIMARY: F.normalize(primary, dim=-1)}
        if controls:
            features["UnitOnly"] = F.normalize(only, dim=-1)
            spatial = unit_labels(relation, valid[0], (prepared.grid_height, prepared.grid_width), config, spatial=True)
            spatial_lift, _, _ = reencode_units(head, prepared, spatial, fine)
            features["Spatial_UnitDetail"] = F.normalize(spatial_lift, dim=-1)
            for method in BASELINES[1:]:
                features[method] = run_head(head, prepared.backbone_tokens,
                    prepared.backbone_tokens[:, prepared.prefix_tokens:],
                    prepared.geometry_patch_conditional, prepared.prefix_tokens, method, prepared.block_index)[0]
    replay = F.normalize(fine, dim=-1)
    error = float((replay - prepared.geometry_projected).abs().max())
    if error != 0. or any(not bool(torch.isfinite(value).all()) for value in features.values()):
        raise RuntimeError(f"Original Geometry replay failed: {error}")
    diagnostics.update(geometry_replay_max_error=error, partition_seconds=partition_time)
    return features, diagnostics
