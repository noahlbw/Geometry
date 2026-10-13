"""Fixed-capacity semantic memory with physical local-donor mass returned to self."""
import torch

from .compact_deployment import local_observations
from .context_memory_deployment import memory_rgb
from .context_memory_observer import CompactContextScorer, context_assignments
from .view_disjoint_memory import view_donor_weights, retained_query_factors, retained_context_head


@torch.inference_mode()
def view_disjoint_wide_observations(image, observer, source, *, wide_policy='long448', final_only=False):
    rgb, size = memory_rgb(image,wide_policy=wide_policy)
    h,w = size
    ph,pw = rgb.shape[-2:]
    raw = observer.raw_patch_features(rgb)
    assignment = context_assignments(raw,observer.config)
    factors = []
    for tile in source['local']:
        _,_,outside = view_donor_weights((ph//16,pw//16),size,source['size'],tile,device=raw.device)
        factors.append(retained_query_factors(assignment,outside))
    packed = tuple(torch.cat([row[index] for row in factors],dim=0) for index in range(4))
    with observer.backbone._autocast():
        features = retained_context_head(observer.backbone.model.visual_model.head,raw,assignment,packed,final_only=final_only)
    views = {}
    for index,tile in enumerate(source['local']):
        views[tile['top'],tile['left']] = dict(top=0,left=0,ah=h,aw=w,
            grid_hw=(ph//16,pw//16),output_hw=(ph,pw),features=features[index:index+1])
    return dict(size=source['size'],wide_size=size,wide_by_tile=views,wide=[next(iter(views.values()))])


@torch.inference_mode()
def view_disjoint_observations(image,geometry,observer,strengths,*,wide_policy='long448',reconstruction=None,final_only=False):
    source = local_observations(image,geometry,strengths,reconstruction=reconstruction)
    return dict(source,**view_disjoint_wide_observations(image,observer,source,wide_policy=wide_policy,final_only=final_only))


class ViewDisjointScorer(CompactContextScorer):
    @torch.inference_mode()
    def __call__(self,source,query,tau=1.,tem=None):
        from eval_geometry_vip_reliability import sample_broad
        output = {}
        for position,crop in source['wide_by_tile'].items():
            dense=super().__call__(dict(source,wide=[crop]),query,tau,tem)
            output[position]=sample_broad(dense,*position,*source['size']).reshape(-1,len(query.class_names))
            del dense
        return output
