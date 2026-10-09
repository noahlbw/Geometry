"""Bounded semantic attenuation of inherited wide alias evidence.

Retains inherited salience and query-count offsets. Explicit residual groups
are not synonym ensembles and are left unchanged. No visual observation added.
"""
import torch
import torch.nn.functional as F

IMPLEMENTATION='wide-alias-bounded-rival-loggate-v1-20261007'


class BoundedWideRival:
    @torch.inference_mode()
    def __init__(self,query,background=None):
        parents=query.parents.detach().cpu().tolist()
        self.groups=tuple(torch.tensor([i for i,p in enumerate(parents) if p==c],device=query.features.device)
                          for c in range(len(query.class_names)))
        if any(len(g)==0 for g in self.groups):raise ValueError('Nonempty classes required.')
        self.background=background
        text=F.normalize(query.features.float().mean(1),dim=-1)
        prototypes=F.normalize(torch.stack([text[g].mean(0) for g in self.groups]),dim=-1)
        self.semantic=text@prototypes.T

    @torch.inference_mode()
    def reduce(self,scaled_groups,tau):
        """Inputs are inherited K*q-weighted [alias,height,width] logits."""
        if len(scaled_groups)!=len(self.groups) or tau<=0:raise ValueError('Matching groups and positive tau required.')
        baseline=torch.stack([torch.logsumexp(tau*x,dim=0)/tau for x in scaled_groups])
        eligible=[c for c in range(len(self.groups)) if c!=self.background]
        if len(eligible)<2:return baseline
        candidates=baseline[eligible]
        ranking=candidates.topk(2,dim=0).indices
        ids=torch.tensor(eligible,device=baseline.device)
        first,second=ids[ranking[0]],ids[ranking[1]]
        output=[]
        for c,(indices,values) in enumerate(zip(self.groups,scaled_groups)):
            if c==self.background:
                output.append(baseline[c]);continue
            rival=torch.where(first==c,second,first)
            own=self.semantic[indices,c][:,None,None]
            other=self.semantic[indices[:,None,None],rival[None]]
            # Reuse .07 and the one-half floor; no redistributed mass or new fit.
            gate=.5+.5*torch.sigmoid((own-other)/.07)
            output.append(torch.logsumexp(tau*values+gate.log(),dim=0)/tau)
        return torch.stack(output)

    @torch.inference_mode()
    def crop(self,features,query,settings):
        with torch.autocast(device_type='cuda',dtype=torch.bfloat16):
            similarities=torch.einsum('bnd,mtd->bnmt',features,query.features.float()).mean(-1)[0]
            patch_mean=F.normalize(features.mean(1),dim=-1)
            text_mean=F.normalize(query.features.float().mean(1),dim=-1)
            salience=((patch_mean@text_mean.T)[0]/settings.tem).float()
            alias=(similarities*settings.logit_scale).T.reshape(-1,21,21)
            scaled=[]
            for members in self.groups:
                weights=salience[members].softmax(0)
                scaled.append(alias[members]*(weights/weights.mean())[:,None,None])
            logits=self.reduce(scaled,settings.tau)[None]
            return F.interpolate(logits,(336,336),mode='bilinear',align_corners=False)[0].float()

    @torch.inference_mode()
    def score(self,source,query,tau,tem):
        from .vip_official_adapter import VIPSettings
        h,w=source['wide_size']
        output=torch.zeros(len(self.groups),h,w,device=query.features.device)
        count=torch.zeros(h,w,device=query.features.device)
        settings=VIPSettings(tau=tau,tem=tem)
        for crop in source['wide']:
            top,left,ah,aw=(crop[k] for k in ('top','left','ah','aw'))
            logits=self.crop(crop['features'],query,settings)
            output[:,top:top+ah,left:left+aw]+=logits[:,:ah,:aw]
            count[top:top+ah,left:left+aw]+=1
        if not bool((count>0).all()):raise RuntimeError('Wide coverage gap.')
        return output/count[None]
