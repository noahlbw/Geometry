"""Image-conditioned alias weights from two existing regional response fields.

Agreement is a fallible signal, not a semantic truth oracle. No canonical or
rival query supplies a pseudo-label. All aliases retain positive mass and the
declared class mass G is shared by the candidate and its controls.
"""
import math

import torch
import torch.nn.functional as F

from .family_mass_alias import prepare_plan


IMPLEMENTATION='geometry-regional-two-observer-alias-residual-v1-20261009'
CURRENT='Current20_Base'
RAW_BASE='Complete20_Base'
BASE='Region_BaseUniform'
PRIMARY='RegionResidual_Soft'
ALIAS_NULL='RegionResidual_AliasShuffle'
IMAGE_NULL='RegionResidual_ImageShuffle'
FAMILY='Complete20_FamilySum'
VIP='VIP_Complete20'
METHODS=(CURRENT,RAW_BASE,BASE,PRIMARY,ALIAS_NULL,IMAGE_NULL,FAMILY)
SEED=20261010
ANCHORS=tuple(y*32+x for y in (4,12,20,28) for x in (4,12,20,28))


class CaptureGeometry:
    """Capture the already computed affinity without another encoder/head call."""
    def __init__(self,geometry):
        self.geometry=geometry; self.relations=[]
    def __getattr__(self,name): return getattr(self.geometry,name)
    def prepare_image(self,rgb):
        prepared=self.geometry.prepare_image(rgb)
        self.relations.append(prepared.geometry_patch_conditional[0])
        return prepared


def observations(image,geometry,vip,policy):
    from eval_development_readout import observations as unchanged
    recorder=CaptureGeometry(geometry)
    source=unchanged(image,recorder,vip,(policy.strength,),wide_policy=policy.wide_policy)
    if len(source['local'])!=len(recorder.relations): raise RuntimeError('Captured tile inventory differs.')
    for tile,relation in zip(source['local'],recorder.relations): tile['region_relation']=relation
    return source


def region_support(relation,valid):
    """Sixteen fixed valid anchors, at most32 positive leave-self-out donors."""
    if relation.shape!=(1024,1024) or valid.shape!=(1024,) or valid.dtype!=torch.bool:
        raise ValueError('The frozen32x32 lattice and boolean validity are required.')
    anchors=torch.tensor(ANCHORS,device=relation.device)
    rows=relation[anchors].float().masked_fill(~valid[None],0.)
    rows=rows.scatter(1,anchors[:,None],0.)
    values,indices=rows.topk(32,-1)
    values=values.clamp_min(0.)
    sums=values.sum(-1,keepdim=True)
    use=valid[anchors]&(sums[:,0]>0)
    weights=torch.zeros_like(rows)
    weights.scatter_(1,indices,values/sums.clamp_min(1e-12))
    return weights[use]


def residual_correlation(local,wide):
    if local.shape!=wide.shape or local.ndim!=3 or local.shape[-1]!=20:
        raise ValueError('Matching[region,class,20] fields required.')
    if local.shape[0]<2: return local.new_zeros(local.shape[1:])
    # Linear pooling and subtraction commute. These are raw, unprofiled fields.
    local=local-local.mean(-1,keepdim=True)
    wide=wide-wide.mean(-1,keepdim=True)
    local=local-local.mean(0,keepdim=True)
    wide=wide-wide.mean(0,keepdim=True)
    lv=local.square().sum(0); wv=wide.square().sum(0)
    q=(local*wide).sum(0)/(lv*wv).sqrt().clamp_min(1e-12)
    return torch.where((lv>1e-12)&(wv>1e-12),q.clamp(-1.,1.),torch.zeros_like(q))


def alias_derangements(classes,device):
    rng=torch.Generator().manual_seed(SEED); rows=[]; identity=torch.arange(20)
    for _ in range(classes):
        while True:
            row=torch.randperm(20,generator=rng)
            if bool((row!=identity).all()): break
        rows.append(row)
    return torch.stack(rows).to(device)


def log_weights(q,family_counts):
    if q.ndim!=2 or q.shape[-1]!=20 or len(family_counts)!=q.shape[0]:
        raise ValueError('Matching class masses and[classes,20] correlations required.')
    mass=q.new_tensor(family_counts).log()[:,None]
    return mass+q-q.logsumexp(-1,keepdim=True)


@torch.inference_mode()
def raw_wide(source,query):
    crops=[]
    for crop in source['wide']:
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            raw=torch.einsum('bnd,mtd->bnmt',crop['features'],query.features.float()).mean(-1)[0]
            visual=F.normalize(crop['features'].mean(1),dim=-1)
            text=F.normalize(query.features.float().mean(1),dim=-1)
            salience=((visual@text.T)[0]).float()
            # Preserve inherited BF16 multiplication/rounding before conversion.
            # The diagnostic reads unscaled raw responses; the final reader uses
            # exactly the original scaled responses, not FP32 raw*40.
            scaled_raw=(raw*40.).T.reshape(-1,21,21)
        crops.append(dict(crop,raw=raw.T.reshape(-1,21,21).float(),
                          scaled_raw=scaled_raw,salience=salience))
    return crops


def aligned_raw(crops,source,tile):
    """Sample raw21x21 fields directly at the same physical patch centres.

    This proxy uses its declared direct bilinear sampling; it is not an exact
    replay of the final crop-upsample/stitch/sample score operator.
    """
    ih,iw=source['size']; wh,ww=source['wide_size']; device=crops[0]['raw'].device
    y=(tile['top']+(torch.arange(32,device=device).float()+.5)*16)/ih*wh
    x=(tile['left']+(torch.arange(32,device=device).float()+.5)*16)/iw*ww
    yy,xx=torch.meshgrid(y,x,indexing='ij')
    total=torch.zeros((1024,crops[0]['raw'].shape[0]),device=device)
    count=torch.zeros(1024,device=device)
    for crop in crops:
        cy,cx=yy-crop['top'],xx-crop['left']
        covered=((cy>=0)&(cy<crop['ah'])&(cx>=0)&(cx<crop['aw'])).flatten()
        grid=torch.stack((cx/336*2-1,cy/336*2-1),-1)[None]
        value=F.grid_sample(crop['raw'][None],grid,mode='bilinear',padding_mode='border',align_corners=False)[0]
        total+=value.reshape(-1,1024).T*covered[:,None]
        count+=covered
    # Invalid padded local centres have no support and do not enter regions.
    return total/count.clamp_min(1.)[:,None]


@torch.inference_mode()
def evidence(source,bank,query,plan,policy):
    from dinotool.geometry_readout_trace import alias_class_scores
    crops=raw_wide(source,query); local_regions=[]; wide_regions=[]; cache={}
    text=F.normalize(bank.features.float(),dim=-1); h,w=source['size']
    for number,tile in enumerate(source['local']):
        local=tile['features'][policy.strength].float()@text.T
        cache[(policy.profile().bank,policy.strength,policy.temperature,number)]=alias_class_scores(
            local,bank.parent_indices,bank.class_count)/policy.temperature
        y,x=torch.meshgrid(torch.arange(32,device=text.device)*16+8+tile['top'],
                           torch.arange(32,device=text.device)*16+8+tile['left'],indexing='ij')
        valid=((y<h)&(x<w)).flatten()
        support=region_support(tile['region_relation'],valid)
        if len(support):
            local_regions.append((support@local).reshape(-1,bank.class_count,20))
            wide_regions.append((support@aligned_raw(crops,source,tile)).reshape(-1,bank.class_count,20))
    if local_regions:
        l=torch.cat(local_regions); b=torch.cat(wide_regions); q=residual_correlation(l,b)
    else:
        l=text.new_empty((0,bank.class_count,20)); q=text.new_zeros((bank.class_count,20))
    if len(l)>64: raise RuntimeError('Fixed region budget exceeded.')
    return q,crops,cache,dict(region_count=len(l),mean_abs_q=float(q.abs().mean()),
        nonzero_q_fraction=float((q!=0).float().mean()),additional_visual_forwards=0,
        additional_semantic_heads=0,maximum_region_elements=len(l)*bank.class_count*20)


@torch.inference_mode()
def wide_fields(source,crops,plan,q,peer_q,wanted,policy):
    h,w=source['wide_size']; classes=len(plan.offsets); device=crops[0]['raw'].device
    dynamic=log_weights(q,plan.mean.family_counts)
    permutation=alias_derangements(classes,device) if ALIAS_NULL in wanted else None
    image_weights=log_weights(peer_q,plan.mean.family_counts) if IMAGE_NULL in wanted else None
    logs={PRIMARY:dynamic}
    if permutation is not None: logs[ALIAS_NULL]=dynamic.gather(1,permutation)
    if image_weights is not None: logs[IMAGE_NULL]=image_weights
    totals={name:torch.zeros(classes,h,w,device=device) for name in wanted}
    count=torch.zeros(h,w,device=device)
    for crop in crops:
        fields={name:[] for name in wanted}
        for c,ids in enumerate(plan.mean.members):
            salience=(crop['salience'][ids]/policy.tem).softmax(0)
            e=crop['scaled_raw'][ids]*(salience/salience.mean())[:,None,None]
            base=(policy.tau*e).logsumexp(0)/policy.tau
            uniform=base+plan.offsets[c]/policy.tau
            for name in wanted:
                if name==RAW_BASE: value=base
                elif name==BASE: value=uniform
                elif name==FAMILY:
                    from .family_mass_alias import reduce_routes
                    value=reduce_routes(e,plan,c,('sum',),policy.tau)['sum']
                else:
                    lw=logs[name][c]
                    value=(policy.tau*e+lw[:,None,None]).logsumexp(0)/policy.tau
                    # Exact neutral/class-pooled fallback, independent of log rounding.
                    reference=peer_q[c] if name==IMAGE_NULL else q[c]
                    value=torch.where((reference==reference[0]).all(),uniform,value)
                fields[name].append(value)
        for name,rows in fields.items():
            dense=F.interpolate(torch.stack(rows)[None],(336,336),mode='bilinear',align_corners=False)[0]
            top,left,ah,aw=(crop[k] for k in ('top','left','ah','aw'))
            totals[name][:,top:top+ah,left:left+aw]+=dense[:,:ah,:aw]
        count[crop['top']:crop['top']+crop['ah'],crop['left']:crop['left']+crop['aw']]+=1
    if not bool((count>0).all()): raise RuntimeError('Original wide coverage gap.')
    return {name:value/count[None] for name,value in totals.items()}


@torch.inference_mode()
def predict(image,geometry,vip,banks,queries,plans,dataset,methods=METHODS,peer_q=None,expected_q=None):
    from .fixed20_task_readout import POLICIES
    from .family_mass_alias import wide_fields as inherited_fields
    from eval_development_readout import probabilities
    if IMAGE_NULL in methods and peer_q is None: raise ValueError('Explicit mapped donor q required.')
    policy=POLICIES[dataset]; source=observations(image,geometry,vip,policy)
    outputs={}; complete=[name for name in methods if name!=CURRENT]
    diag=dict(geometry_encodings=len(source['local']),wide_encodings=len(source['wide']),fine_forwards=0,
              additional_visual_forwards=0,additional_semantic_heads=0,alias_slots_per_class=20)
    if complete:
        plan=plans['Complete20']; need_q=any(name in complete for name in (PRIMARY,ALIAS_NULL,IMAGE_NULL))
        if need_q:
            q,crops,cache,rd=evidence(source,banks['Complete20'],queries['Complete20'],plan,policy); diag.update(rd)
            if expected_q is not None:
                error=float((q-expected_q).abs().max())
                if error>2e-6: raise RuntimeError('Fresh current-image q differs from mask-free prepass.')
                diag['prepass_q_max_abs']=error
        else:
            q=queries['Complete20'].features.new_zeros((len(plan.offsets),20),dtype=torch.float32)
            crops=raw_wide(source,queries['Complete20']); cache={}
        fields=wide_fields(source,crops,plan,q,peer_q,complete,policy)
        for name in complete:
            probability=probabilities(source,policy.profile(),banks['Complete20'],fields[name],cache)
            outputs[name]=probability.argmax(0).cpu().numpy()
    if CURRENT in methods:
        base=inherited_fields(source,queries['Current20'],plans['Current20'],('base',),policy.tau,policy.tem)['base']
        probability=probabilities(source,policy.profile(),banks['Current20'],base,{})
        outputs[CURRENT]=probability.argmax(0).cpu().numpy()
    return outputs,diag
