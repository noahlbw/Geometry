"""Precomputed variable-size alias groups; vectorized equivalent soft readout."""
import math
import torch
import torch.nn.functional as F


class PackedAliasReadout:
    """Build once after text encoding; neither masks nor fitted weights are used."""

    @torch.inference_mode()
    def __init__(self,bank,temperature=.07):
        if temperature<=0:
            raise ValueError('Positive alias temperature required.')
        self.temperature=temperature
        self.classes=bank.class_count
        self.aliases=len(bank.features)
        parents=bank.parent_indices.detach().cpu().tolist()
        groups=[[i for i,p in enumerate(parents) if p==c] for c in range(self.classes)]
        if any(not g for g in groups) or any(p<0 or p>=self.classes for p in parents):
            raise ValueError('Every alias requires a valid nonempty semantic group.')
        device=bank.features.device
        self.reference_groups=tuple((torch.tensor(g,device=device),math.log(len(g))) for g in groups)
        width=max(map(len,groups))
        self.indices=torch.tensor([g+[g[0]]*(width-len(g)) for g in groups],device=device)
        self.valid=torch.arange(width,device=device)[None]<torch.tensor(list(map(len,groups)),device=device)[:,None]
        self.counts=torch.tensor(list(map(len,groups)),dtype=torch.float32,device=device)
        self.class_indices=torch.arange(self.classes,device=device)
        text=F.normalize(bank.features.float(),dim=-1)
        # Preserve the scalar reference's exact prototype reduction order.
        prototypes=F.normalize(torch.stack([text[torch.tensor(g,device=device)].mean(0) for g in groups]),dim=-1)
        semantic=text@prototypes.T
        self.semantic=semantic
        self.own_semantic=semantic[self.indices,self.class_indices[:,None]]

    def grouped(self,alias):
        if alias.ndim!=2 or alias.shape[1]!=self.aliases:
            raise ValueError('Expected patch-by-alias scores for this frozen bank.')
        return alias[:,self.indices]

    @torch.inference_mode()
    def uniform(self,alias):
        grouped=self.grouped(alias)/self.temperature
        grouped=grouped.masked_fill(~self.valid[None],-float('inf'))
        return self.temperature*(torch.logsumexp(grouped,-1)-self.counts.log())

    @torch.inference_mode()
    def uniform_reference(self,alias):
        """Preserve scalar reduction/count-log order before discontinuous gates.

        Cache integer groups and Python logs to avoid per-class GPU count syncs.
        This is intentionally used only for the exact protected residual route.
        """
        if alias.ndim!=2 or alias.shape[1]!=self.aliases:
            raise ValueError('Expected patch-by-alias scores for this frozen bank.')
        return torch.stack([self.temperature*(torch.logsumexp(alias[...,indices]/self.temperature,-1)-log_count)
                            for indices,log_count in self.reference_groups],dim=-1)

    @torch.inference_mode()
    def coverage(self,alias):
        grouped=self.grouped(alias)
        baseline=self.temperature*(torch.logsumexp((grouped/self.temperature).masked_fill(~self.valid[None],-float('inf')),-1)
                                   -self.counts.log())
        if self.classes==1:
            return baseline
        top=baseline.topk(2,-1).indices
        rival=torch.where(top[:,0,None]==self.class_indices[None],top[:,1,None],top[:,0,None])
        specificity=self.own_semantic[None]-self.semantic[self.indices[None],rival[...,None]]
        weights=torch.sigmoid(specificity/self.temperature).clamp_min(1e-6)*self.valid[None]
        weights=weights/weights.sum(-1,keepdim=True)
        weights=.5/self.counts[None,:,None]+.5*weights
        weighted=(grouped/self.temperature+weights.log()).masked_fill(~self.valid[None],-float('inf'))
        return self.temperature*torch.logsumexp(weighted,-1)
