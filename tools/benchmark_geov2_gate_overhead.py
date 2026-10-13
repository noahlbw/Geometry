"""Measure the extra arithmetic of the image-only Geov2 gate on one GPU."""
import json
import sys
import time

import torch

sys.path.insert(0, '/data/test/code/ovss_cafe_ped_v2_20260919/DINOtool_hero_external_20260926')
from dinotool.geov2 import adaptive_writeback_gain


@torch.inference_mode()
def main():
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this benchmark.')
    device = torch.device('cuda')
    n, classes = 1024, 21
    local = torch.randn(n, classes, device=device, dtype=torch.float32)
    wide = torch.randn_like(local)
    target = local + .65 * (wide - local)
    for _ in range(20):
        gain = adaptive_writeback_gain(local, wide, target)
        _ = local + gain[:, None] * (target - local)
    torch.cuda.synchronize()
    samples = []
    peaks = []
    for _ in range(100):
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        gain = adaptive_writeback_gain(local, wide, target)
        _ = local + gain[:, None] * (target - local)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000.)
        peaks.append(torch.cuda.max_memory_allocated(device) / 1048576.)
    result = dict(device=torch.cuda.get_device_name(device), patches=n, classes=classes,
                  repeats=len(samples), median_ms=float(torch.tensor(samples).median()),
                  p95_ms=float(torch.tensor(samples).quantile(.95)),
                  peak_allocated_mib=max(peaks),
                  gain_min=float(gain.min()), gain_max=float(gain.max()),
                  operation='adaptive gate plus final elementwise writeback; excludes encoder and Z* solve')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
