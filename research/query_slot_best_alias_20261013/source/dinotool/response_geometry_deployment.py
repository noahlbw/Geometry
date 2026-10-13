"""Local Geometry wrapper for the frozen DonorBefore response ordering."""
from .compact_deployment import CompactGeometry
from .geometry_semantic_response import read_head
from .efficient_deployment import EfficientGeometry


class ResponseGeometry(CompactGeometry):
    """Keep the original Geometry relation/H/g, change only local head order."""
    @staticmethod
    def _project_response(head, prepared):
        features, _ = read_head(head, prepared, 'Geometry_DonorBefore', trace=False)
        return features

    def project(self, prepared, strength):
        # DonorBefore is defined for the normalized original Geometry relation.
        # Numeric strength routes intentionally retain the incumbent operator:
        # scaling G changes its row mass, so transporting a response with that
        # unnormalized relation would no longer implement the declared
        # self-conditional donor rule.  This makes the deployment conservative
        # for calibrated strength profiles such as Potsdam and VOC.
        if strength == 'original':
            return self._project_response(self.backbone.model.visual_model.head, prepared)
        return super().project(prepared, strength)


class EfficientResponseGeometry(ResponseGeometry, EfficientGeometry):
    """Keep DonorBefore projection and reuse efficient preparation/cached geometry."""
