"""Opt-in execution changes for the unchanged single-crop fine observer."""
import time

import torch

from .fine_reference_admission import CONFIG, fine_patch_features


IMPLEMENTATION = 'frozen-fine-observer-execution-v1-20261005'


def validate_features(features):
    if not bool(torch.isfinite(features).all()):
        raise RuntimeError('Nonfinite physically fine reference features.')
    return features


@torch.inference_mode()
def sync_free_patch_features(observer, rgb):
    # Empty-row counts are unused here; keep the same per-crop finite-value check.
    return validate_features(fine_patch_features(observer, rgb, capture_safe=True))


class FineObserverGraph:
    """One batch-one512 graph; returned clones survive subsequent replays."""
    def __init__(self, observer):
        self.observer = observer
        self.graph = None
        self.static_rgb = None
        self.static_features = None
        self.setup_seconds = None
        self.replays = 0

    @torch.inference_mode()
    def __call__(self, observer, rgb):
        if observer is not self.observer:
            raise ValueError('A fine graph belongs to one fixed observer.')
        if rgb.shape != (3, CONFIG.encoder_side, CONFIG.encoder_side) or not rgb.is_floating_point():
            raise ValueError('Require one floating-point CHW512 RGB crop.')
        device = torch.device(observer.device)
        if device.type != 'cuda':
            raise ValueError('Fine observer graph requires CUDA.')
        with torch.cuda.device(device):
            if self.graph is None:
                started = time.perf_counter()
                self.static_rgb = rgb.to(device).clone()
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        fine_patch_features(observer, self.static_rgb, capture_safe=True)
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                self.graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(self.graph):
                    self.static_features = fine_patch_features(observer, self.static_rgb, capture_safe=True)
                torch.cuda.synchronize(device)
                self.setup_seconds = time.perf_counter()-started
            if rgb.dtype != self.static_rgb.dtype:
                raise ValueError('Fine graph RGB dtype changed.')
            self.static_rgb.copy_(rgb)
            self.graph.replay()
            self.replays += 1
            return validate_features(self.static_features.clone())
