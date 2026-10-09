"""Remove inherited wide LSE query-count offsets without a fitted strength."""
import torch

IMPLEMENTATION='geometry-wide-count-normalized-lse-v1-20261007'


class CountNormalizedWide:
    def __init__(self,parents,class_count,*,keep_equal_count_gauge=False):
        counts=torch.bincount(parents.long(),minlength=class_count)
        if len(counts)!=class_count or not bool((counts>0).all()):
            raise ValueError('Every scored class requires a nonempty query group.')
        self.log_counts=counts.double().log()
        self.counts=counts.detach().cpu().tolist()
        self.skip=keep_equal_count_gauge and len(set(self.counts))==1

    def apply(self,logits,tau):
        if logits.ndim<1 or logits.shape[0]!=len(self.counts) or tau<=0:
            raise ValueError('Class-first wide logits and positive frozen tau required.')
        # A common class offset has no effect on geometric coupled probabilities.
        # Do not skip it when an external residual score replaces one class.
        if self.skip:return logits
        offset=(self.log_counts/tau).to(logits.dtype)
        return logits-offset.reshape(-1,*([1]*(logits.ndim-1)))
