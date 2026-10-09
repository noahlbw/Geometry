"""Device-resident float32 stitching with shared max/argmax finalization."""
import torch
import torch.nn.functional as F


class DeviceProbabilityAccumulator:
    def __init__(self, classes, height, width, device):
        self.classes, self.height, self.width = classes, height, width
        self.probabilities = torch.zeros(classes, height, width, dtype=torch.float32, device=device)
        self.normalizer = torch.zeros(height, width, dtype=torch.float32, device=device)

    def add(self, probabilities, weights, left, top):
        height, width = weights.shape
        if probabilities.shape != (self.classes, height, width):
            raise ValueError('Matching class probability tile and blend weights required.')
        # Separate multiply/add matches the original NumPy stitching, not fused addcmul.
        weighted = probabilities * weights[None]
        self.probabilities[:, top:top + height, left:left + width].add_(weighted)
        self.normalizer[top:top + height, left:left + width].add_(weights)

    def finalize_outputs(self, threshold, thresholds, background, background_index=0):
        if not bool((self.normalizer > 0).all()):
            raise ValueError('Uncovered device probability pixels.')
        limits = torch.as_tensor(thresholds, dtype=torch.float64, device=self.probabilities.device)
        if (limits.shape != (self.classes,) or not bool(torch.isfinite(limits).all())
                or bool(((limits < 0) | (limits > 1)).any()) or not background and bool(limits.any())):
            raise ValueError('Valid frozen class thresholds and background protocol required.')
        default = torch.empty(self.height, self.width, dtype=torch.uint8, device=self.probabilities.device)
        calibrated = torch.empty_like(default)
        rows = max(1, min(1024, (64 * 1024 * 1024) // max(self.classes * self.width * 4, 1)))
        for top in range(0, self.height, rows):
            bottom = min(top + rows, self.height)
            normalized = self.probabilities[:, top:bottom] / self.normalizer[None, top:bottom]
            confidence, labels = normalized.max(0)
            standard = labels.to(torch.uint8)
            if threshold is not None:
                standard = standard.masked_fill(confidence < threshold, background_index)
            default[top:bottom] = standard
            if background:
                labels = labels.masked_fill(confidence.double() < limits[labels], 0)
            calibrated[top:bottom] = labels.to(torch.uint8)
        return default.cpu().numpy(), calibrated.cpu().numpy()

    def finalize_resized(self, output_size):
        if (len(output_size) != 2 or min(output_size) < 1
                or not bool((self.normalizer > 0).all())):
            raise ValueError('Nonempty output size and complete probability coverage required.')
        normalized = self.probabilities / self.normalizer[None]
        restored = F.interpolate(normalized[None], tuple(output_size), mode='bilinear',
                                 align_corners=False)[0]
        return restored.argmax(0).to(torch.uint8).cpu().numpy()

    def close(self):
        self.probabilities = self.normalizer = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
