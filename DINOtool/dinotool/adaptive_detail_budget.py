"""One image-only detail acquisition, guided by semantic disagreement and H."""
import torch
import torch.nn.functional as F

from .fine_reference_admission import fine_crop_positions


IMPLEMENTATION = 'geometry-disagreement-detail-budget1-bounded4-v1-20261005'
PRIMARY = 'SparseAdaptiveDetail1'
METHODS = ('Geometry', 'NoAdmission_Exact', 'SparseNativeSoft', PRIMARY)


def detail_rgb(rgb, position):
    if rgb.ndim == 4 and rgb.shape[0] == 1:
        rgb = rgb[0]
    if rgb.ndim != 3 or rgb.shape[0] != 3:
        raise ValueError('One RGB window, unbatched or batch-one, required.')
    top, left, height, width = position
    if not 0 < height <= 256 or not 0 < width <= 256:
        raise ValueError('Actual positive budgeted crop dimensions at most256 required.')
    cropped = rgb[:, top:top + height, left:left + width]
    if cropped.shape[-2:] != (height, width):
        raise ValueError('Budgeted crop lies outside its actual RGB window.')
    padded = F.pad(cropped, (0, 256 - width, 0, 256 - height))
    return F.interpolate(padded[None], (512, 512), mode='bilinear', align_corners=False)[0]


def select_detail(local_fields, broad_fields, operator, coordinates, valid, height, width):
    if (not local_fields or len(local_fields) != len(broad_fields)
            or operator.shape != (len(valid), len(valid)) or coordinates.shape != (len(valid), 2)):
        raise ValueError('Matched image-only branch fields, H and valid local coordinates required.')
    priority = torch.zeros(len(valid), device=operator.device, dtype=torch.float64)
    disagreement = torch.zeros_like(valid)
    for local, broad in zip(local_fields, broad_fields):
        if local.shape != broad.shape or local.shape[0] != len(valid):
            raise ValueError('Matching local and broad class scores required.')
        p, q = local.double().softmax(-1), broad.double().softmax(-1)
        m = (p + q) * .5
        entropy_p = (p * (p.clamp_min(1e-300).log() - m.clamp_min(1e-300).log())).sum(-1)
        entropy_q = (q * (q.clamp_min(1e-300).log() - m.clamp_min(1e-300).log())).sum(-1)
        differs = (local.argmax(-1) != broad.argmax(-1)) & valid
        priority += ((entropy_p + entropy_q) * .5).clamp_min(0) * differs / len(local_fields)
        disagreement |= differs
    priority *= operator.double().square().sum(0)
    priority.masked_fill_(~valid, 0.)
    positions = fine_crop_positions(height, width)
    values = []
    for top, left, h, w in positions:
        inside = ((coordinates[:, 0] >= top) & (coordinates[:, 0] < top + h)
                  & (coordinates[:, 1] >= left) & (coordinates[:, 1] < left + w))
        values.append(priority[inside].sum())
    values = torch.stack(values)
    if not bool((values > 0).any()):
        return None, dict(detail_priority_sum=float(priority.sum()), detail_disagreement_fraction=float(disagreement[valid].double().mean())
                         if bool(valid.any()) else 0., detail_selected_priority_fraction=0.)
    choice = int(values.argmax())
    return positions[choice], dict(detail_priority_sum=float(priority.sum()),
        detail_disagreement_fraction=float(disagreement[valid].double().mean()) if bool(valid.any()) else 0.,
        detail_selected_priority_fraction=float(values[choice] / values.sum()), detail_selected_quadrant=float(choice))


def replace_witness(native, observation, layout, valid):
    if observation.evidence.shape[1:] != native.shape[1:] or native.shape[0] != len(valid):
        raise ValueError('Matching native/detail alias fields required.')
    mass = observation.coefficients.sum(-1)
    covered = (mass > 0) & valid
    evidence = observation.evidence.masked_fill(~layout.valid[None], 0.)
    sampled = (evidence[observation.indices] * observation.coefficients[..., None, None]).sum(1)
    sampled /= mass.clamp_min(1e-12)[:, None, None]
    sampled.masked_fill_(~layout.valid[None], -torch.inf)
    witness = torch.where(covered[:, None, None], sampled, native)
    return witness, covered
