"""Bounded residual evidence suppression conditioned on the current rival."""
import math
import torch
import torch.nn.functional as F


class ResidualRivalWeight:
    def __init__(self,residual_text,foreground_anchors,temperature=.07):
        if temperature<=0:
            raise ValueError('Positive shared alias temperature required.')
        self.temperature=temperature
        text=F.normalize(residual_text.float(),dim=-1)
        centroid=F.normalize(text.mean(0),dim=-1)
        self.own=text@centroid
        self.cross=text@F.normalize(foreground_anchors.float(),dim=-1).T

    @torch.inference_mode()
    def score(self,cosine,rival):
        specificity=self.own[None]-self.cross[:,rival].T
        gate=.5+.5*torch.sigmoid(specificity/self.temperature)
        return (cosine+self.temperature*gate.log()).amax(-1)

    @property
    def maximum_raw_score_loss(self):
        return self.temperature*math.log(2)
