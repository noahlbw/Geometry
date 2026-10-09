"""Permutation-invariant source supervision for the shared region partition."""
import torch
from torch import Tensor
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment


def proposal_partition_loss(logits: Tensor, target: Tensor, height: int, width: int) -> Tensor:
    """Match source semantic masks to slots, then supervise actual memberships.

    Targets are semantic masks (possibly disconnected), not invented instances.
    Invalid pixels remain unsupervised. Matching uses detached costs, while CE
    and Dice propagate gradients through the original membership logits.
    """
    if logits.ndim != 3 or logits.shape[1] != height * width or len(logits) != len(target):
        raise ValueError("Expected B x (height*width) x slots proposal logits")
    losses = []
    for proposal, labels in zip(logits.float(), target):
        valid = labels != 255
        classes = labels[valid].unique()
        if not len(classes):
            losses.append(proposal.sum() * 0)
            continue
        if len(classes) > proposal.shape[-1]:
            raise ValueError("More semantic regions than available slots")
        masks = (labels[None] == classes[:, None, None]).float()
        occupancy = F.adaptive_avg_pool2d(masks, (height, width)).flatten(1).T
        fraction = F.adaptive_avg_pool2d(valid[None].float(), (height, width)).flatten()
        desired = occupancy / fraction[:, None].clamp_min(1e-6)
        log_prob = proposal.log_softmax(-1)
        prob = log_prob.exp()
        mass = occupancy.sum(0).clamp_min(1e-6)
        # Columns are target masks, rows are candidate slots.
        cost_ce = -(log_prob.T @ occupancy) / mass[None]
        intersection = prob.T @ occupancy
        union = (prob * fraction[:, None]).sum(0)[:, None] + mass[None]
        cost_dice = 1 - (2 * intersection + 1e-6) / (union + 1e-6)
        slots, targets = linear_sum_assignment((cost_ce + cost_dice).detach().cpu().numpy())
        selected = torch.as_tensor(slots, device=proposal.device)
        order = torch.as_tensor(targets, device=proposal.device)
        ce = -(log_prob[:, selected] * occupancy[:, order]).sum() / fraction.sum().clamp_min(1)
        pred = prob[:, selected]
        truth = desired[:, order]
        numerator = 2 * (pred * truth * fraction[:, None]).sum(0)
        denominator = ((pred + truth) * fraction[:, None]).sum(0)
        dice = (1 - (numerator + 1e-6) / (denominator + 1e-6)).mean()
        losses.append(ce + dice)
    return torch.stack(losses).mean()
