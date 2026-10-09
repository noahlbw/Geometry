"""Experimental grouped wide readout; no changes to deployed inference.

Groups must be declared semantic equivalences, not inferred from target masks.
The frozen reference count preserves the old class offset independently of
the number of paraphrases subsequently supplied. Residual unions require a
separate ontology and must not be represented as one synonym group.
"""
import math
import torch

# Conservative lexical equivalences only; modifiers and subtypes stay distinct.
# Never merge superclass/subclass or RS wall/roof/building by text similarity.
EQUIVALENCES = (
    ('aeroplane', 'airplane', 'plane'), ('car', 'cars', 'automobile', 'motorcar'),
    ('vehicle', 'vehicles'), ('wall', 'walls'), ('facade', 'facades'),
    ('roof', 'roofs'), ('rooftop', 'rooftops'), ('tree', 'trees'),
    ('building', 'buildings', 'edifice'), ('house', 'houses'),
    ('road', 'roads', 'roadway'), ('street', 'streets'),
    ('motorbike', 'motorcycle'), ('sofa', 'couch'), ('tv', 'television'),
    ('shelf', 'shelves'), ('person', 'people'), ('bicycle', 'bicycles'),
)


def lexical_groups(aliases, *, residual=False):
    """Frozen conservative language-only grouping, with residual union exempt."""
    if residual:
        return list(range(len(aliases)))
    lookup = {word: group[0] for group in EQUIVALENCES for word in group}
    keys = [lookup.get(word.strip().lower(), word.strip().lower()) for word in aliases]
    labels = {key: i for i, key in enumerate(dict.fromkeys(keys))}
    return [labels[key] for key in keys]


class SemanticMassWide:
    def __init__(self, queries, background=None, *, singleton=False):
        self.query = queries
        parents = queries.parents.detach().cpu().tolist()
        self.members = [torch.tensor([i for i, p in enumerate(parents) if p == c],
                                    device=queries.features.device)
                        for c in range(len(queries.class_names))]
        self.groups = []
        for c, members in enumerate(self.members):
            words = [queries.aliases[i] for i in members.detach().cpu().tolist()]
            self.groups.append(list(range(len(words))) if singleton else
                               lexical_groups(words, residual=c == background))
        self.changed = any(len(set(g)) < len(g) for g in self.groups)

    @torch.inference_mode()
    def score(self, source, tau, tem):
        import torch.nn.functional as F
        from eval_development_readout import wide_scores
        if not self.changed:
            return wide_scores(source, self.query, tau, tem)
        h, w = source['wide_size']
        output = torch.zeros(len(self.members), h, w, device=self.query.features.device)
        count = torch.zeros(h, w, device=output.device)
        for crop in source['wide']:
            features = crop['features']
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                similarity = torch.einsum('bnd,mtd->bnmt', features, self.query.features.float()).mean(-1)[0]
                visual = F.normalize(features.mean(1), dim=-1)
                text = F.normalize(self.query.features.float().mean(1), dim=-1)
                salience = ((visual @ text.T)[0] / tem).float()
                # Same frozen VIP logit scale as the inherited score.
                from dinotool.vip_official_adapter import VIPSettings
                raw = similarity * VIPSettings(tau=tau, tem=tem).logit_scale
                rows = []
                for idx, groups in zip(self.members, self.groups):
                    a, s = raw[:, idx], salience[idx]
                    if len(set(groups)) == len(groups):
                        weights = s.softmax(0)
                        score = (tau*a*(weights/weights.mean())).logsumexp(-1)/tau
                    else:
                        score = semantic_mass_score(a, s, groups, len(groups), tau, 1.)
                    rows.append(score)
                dense = F.interpolate(torch.stack(rows).reshape(1, len(rows), 21, 21),
                    (336, 336), mode='bilinear', align_corners=False)[0].float()
            top, left, ah, aw = (crop[k] for k in ('top', 'left', 'ah', 'aw'))
            output[:, top:top+ah, left:left+aw] += dense[:, :ah, :aw]
            count[top:top+ah, left:left+aw] += 1
        if not bool((count > 0).all()):
            raise RuntimeError('Wide coverage gap.')
        return output/count[None]


def semantic_mass_score(raw, salience, groups, reference_count, tau=1., tem=1.):
    """Return patch scores for one class, from [..., aliases] raw evidence.

    Singleton groups and reference_count == aliases reproduce inherited VIP
    K*softmax(salience/tem) then LSE/tau. Exact duplicate observations within
    an equivalence group do not add semantic mass. This is an algebraic
    property, not a claim that arbitrary paraphrases are interchangeable.
    """
    if tau <= 0 or tem <= 0 or reference_count <= 0:
        raise ValueError('Positive frozen scales and reference count required.')
    if salience.ndim != 1 or raw.shape[-1] != len(salience):
        raise ValueError('One salience per alias required.')
    if len(groups) != len(salience) or not groups:
        raise ValueError('One declared semantic group per alias required.')
    labels = tuple(dict.fromkeys(groups))
    indices = [torch.tensor([i for i, g in enumerate(groups) if g == label],
                            device=raw.device) for label in labels]
    group_salience = torch.stack([salience.index_select(0, idx).mean() for idx in indices])
    mass = (group_salience / tem).softmax(0)
    terms = []
    for j, idx in enumerate(indices):
        evidence = raw.index_select(-1, idx) * (len(labels) * mass[j])
        terms.append((tau * evidence).logsumexp(-1) - math.log(len(idx)))
    return (torch.stack(terms, -1).logsumexp(-1)
            + math.log(reference_count / len(labels))) / tau
