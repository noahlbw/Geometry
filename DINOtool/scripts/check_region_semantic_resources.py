"""Read-only check of pinned source, original references and fixed diagnostics."""
import json

from safetensors import safe_open

from run_region_semantic_suite_a800 import SETTINGS, SOURCE, TOOL, idle, reference


for gpu in range(8):
    print("GPU", gpu, "idle", idle(gpu))
for dataset, _, _, _, total in SETTINGS:
    path = reference(dataset)
    if not path.is_file():
        print("MISSING", path)
        continue
    row = json.loads(path.read_text())
    print(dataset, row["status"], row["processed_images"], total, path)
    if dataset in ("vdd", "potsdam"):
        path = TOOL/f"results/geometry_readout_diagnostic_8_20261001_r2/{dataset}/results.json"
        diagnostic = json.loads(path.read_text())
        print("diagnostic", diagnostic["processed_images"], list(diagnostic["stage_probe_metrics"]))
with safe_open(SOURCE/"model.safetensors", framework="pt", device="cpu") as source:
    print("SigLIP2 frozen logit_scale", source.get_tensor("logit_scale").item())
