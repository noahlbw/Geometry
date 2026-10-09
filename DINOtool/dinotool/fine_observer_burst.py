"""One replay for several unchanged batch-one fine-observer forwards."""
import time

import torch

from .fine_observer_execution import validate_features
from .fine_reference_admission import CONFIG, fine_patch_features


IMPLEMENTATION = 'frozen-fine-observer-burst-execution-v1-20261005'


class FineObserverBurstGraph:
    def __init__(self, observer):
        self.observer = observer
        self.graphs = {}
        self.setup_seconds = {}
        self.replays = 0

    @torch.inference_mode()
    def __call__(self, observer, crops):
        if observer is not self.observer:
            raise ValueError('A fine burst belongs to one fixed observer.')
        if (not 1 <= len(crops) <= 4 or any(rgb.shape != (3, CONFIG.encoder_side, CONFIG.encoder_side)
                or not rgb.is_floating_point() for rgb in crops)):
            raise ValueError('Require one to four floating-point CHW512 crops.')
        if any(rgb.dtype != crops[0].dtype for rgb in crops):
            raise ValueError('Fine burst crop dtypes must match.')
        device = torch.device(observer.device)
        if device.type != 'cuda':
            raise ValueError('Fine burst graph requires CUDA.')
        count = len(crops)
        with torch.cuda.device(device):
            packed = torch.stack([rgb.to(device) for rgb in crops])
            if count not in self.graphs:
                started = time.perf_counter()
                static_rgb = packed.clone()
                stream = torch.cuda.Stream(device=device)
                stream.wait_stream(torch.cuda.current_stream(device))
                with torch.cuda.stream(stream):
                    for _ in range(3):
                        for index in range(count):
                            fine_patch_features(observer, static_rgb[index], capture_safe=True)
                torch.cuda.current_stream(device).wait_stream(stream)
                torch.cuda.synchronize(device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    outputs = [fine_patch_features(observer, static_rgb[index], capture_safe=True)
                               for index in range(count)]
                    static_features = torch.cat(outputs, 0)
                torch.cuda.synchronize(device)
                self.graphs[count] = (graph, static_rgb, static_features)
                self.setup_seconds[count] = time.perf_counter()-started
            graph, static_rgb, static_features = self.graphs[count]
            if packed.dtype != static_rgb.dtype:
                raise ValueError('Fine burst graph RGB dtype changed.')
            static_rgb.copy_(packed)
            graph.replay()
            self.replays += 1
            # Copy once; slices keep this replay's storage alive independently.
            features = validate_features(static_features.clone())
            return tuple(features[index:index+1] for index in range(count))
