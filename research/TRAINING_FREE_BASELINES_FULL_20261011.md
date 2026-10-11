# Pinned native training-free baseline comparison

Authors native official config where provided; fixed untuned transfer elsewhere. Matched existing images/targets. Different pretrained backbones and external resources remain part of each full method; this is not an operator ablation. Geometry references used prior labeled development optimization.

Status: running. Only verified complete protocols receive scores.

|Protocol|Images|Geometry best measured|Strong verified local VIP|PEARL|GLA-CLIP|PTC|CorrCLIP|SegEarth-OV|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|vdd|80|60.4352|52.5765|37.9121|37.8462|41.1191|37.1270|44.8106|
|potsdam|504|56.1934|44.8995|31.5545|34.8473|33.6962|32.2386|48.1943|
|udd5|40|55.8266|44.9713|43.3762|47.7428|39.8159|40.0991|50.5568|
|vaihingen|113|58.8562|49.1039|26.1222|28.5356|25.8137|25.0724|29.2379|
|oem|384|45.1052|35.0388|27.7070|30.7688|29.4806|24.4936|35.6405|
|loveda P|1669|69.6117|56.3110|47.5212|50.6544|50.6271|43.7985|53.4885|
|loveda D|1669|45.3171|36.1457|29.6031|31.0830|31.1501|26.6162|36.8928|
|landcoverai|1602|65.5400|54.9420|30.9757|31.8306|33.0437|30.8354|32.7043|
|flair1|15700|43.6579|36.6074|19.3884|19.4995|20.7257|20.3994|pending|
|voc20|1449|92.7936|92.5051|86.9143|84.2087|82.7334|87.5405|pending|
|voc21|1449|75.8586|73.2591|64.0821|66.2608|67.1752|73.9381|pending|
|ade150|2000|31.3398|29.4630|19.3537|20.0191|20.9090|26.4301|pending|
|context59|5105|46.3075|45.9359|38.7213|40.0670|40.5982|47.7036|pending|
|context60|5105|44.7613|42.5987|36.5263|37.6621|37.7346|45.1164|pending|
|coco_object81|5000|50.3999|48.9955|38.6592|39.0724|39.6947|44.6265|pending|
|coco_stuff171|5000|33.7661|33.4940|26.3142|26.9368|27.4517|31.8365|pending|
|Eight remote mean|20092|53.8665|44.2856|30.8299|32.7692|31.8556|29.6102|pending|

## Standalone whole-image cost

|Method|Protocol|Median ms|Peak allocated MiB|Prior same-text VIP ms|Ratio|Native config|
|---|---|---:|---:|---:|---:|---|
|PEARL|vdd|167.28|1152.58|190.85|0.877|False|
|PEARL|potsdam|68.77|410.15|104.11|0.661|False|
|PEARL|voc20|19.32|713.49|58.16|0.332|True|
|PEARL|voc21|20.60|1015.36|59.50|0.346|True|
|PEARL|ade150|47.41|851.92|113.98|0.416|True|
|PEARL|context60|55.92|458.84|112.06|0.499|True|
|PEARL|udd5|158.26|947.27|191.15|0.828|False|
|PEARL|oem|67.51|432.07|105.38|0.641|False|
|PEARL|vaihingen|67.64|404.27|108.55|0.623|False|
|PEARL|landcoverai|58.50|370.72|102.71|0.570|False|
|PEARL|loveda|66.97|422.07|105.86|0.633|False|
|PEARL|context59|56.64|457.05|111.84|0.506|True|
|PEARL|coco_object81|78.38|6465.10|131.46|0.596|True|
|PEARL|coco_stuff171|59.75|852.69|185.87|0.321|True|
|PEARL|flair1|62.18|382.07|111.14|0.559|False|
|GLA-CLIP|vdd|163.62|1515.09|190.85|0.857|False|
|GLA-CLIP|potsdam|57.92|1026.15|104.11|0.556|False|
|GLA-CLIP|voc20|38.46|948.71|58.16|0.661|True|
|GLA-CLIP|voc21|38.29|963.84|59.50|0.644|True|
|GLA-CLIP|ade150|41.81|1142.77|113.98|0.367|True|
|GLA-CLIP|context60|47.02|969.43|112.06|0.420|True|
|GLA-CLIP|udd5|159.56|1263.31|191.15|0.835|False|
|GLA-CLIP|oem|58.48|1028.39|105.38|0.555|False|
|GLA-CLIP|vaihingen|57.37|1024.89|108.55|0.528|False|
|GLA-CLIP|landcoverai|52.49|1024.89|102.71|0.511|False|
|GLA-CLIP|loveda|56.76|1026.92|105.86|0.536|False|
|GLA-CLIP|context59|47.24|968.04|111.84|0.422|True|
|GLA-CLIP|coco_object81|50.30|1005.31|131.46|0.383|True|
|GLA-CLIP|coco_stuff171|53.02|1148.07|185.87|0.285|True|
|GLA-CLIP|flair1|51.83|1030.62|111.14|0.466|False|
|PTC|vdd|257.33|1494.40|190.85|1.348|False|
|PTC|potsdam|243.54|706.64|104.11|2.339|False|
|PTC|voc20|136.05|716.49|58.16|2.339|True|
|PTC|voc21|150.90|1591.43|59.50|2.536|True|
|PTC|ade150|534.66|1123.71|113.98|4.691|True|
|PTC|context60|210.23|751.25|112.06|1.876|True|
|PTC|udd5|258.93|1243.40|191.15|1.355|False|
|PTC|oem|248.91|709.80|105.38|2.362|False|
|PTC|vaihingen|246.12|706.53|108.55|2.267|False|
|PTC|landcoverai|230.25|706.53|102.71|2.242|False|
|PTC|loveda|246.39|1218.69|105.86|2.328|False|
|PTC|context59|208.03|750.39|111.84|1.860|True|
|PTC|coco_object81|293.69|10622.90|131.46|2.234|True|
|PTC|coco_stuff171|253.39|1123.31|185.87|1.363|True|
|PTC|flair1|238.56|713.16|111.14|2.146|False|
|CorrCLIP|vdd|1313.54|6838.17|190.85|6.882|False|
|CorrCLIP|potsdam|341.75|2608.44|104.11|3.282|False|
|CorrCLIP|voc20|194.92|2604.48|58.16|3.351|True|
|CorrCLIP|voc21|196.22|2604.70|59.50|3.298|True|
|CorrCLIP|ade150|394.60|2606.23|113.98|3.462|True|
|CorrCLIP|context60|217.07|2604.52|112.06|1.937|True|
|CorrCLIP|udd5|1184.18|6837.51|191.15|6.195|False|
|CorrCLIP|oem|359.23|2609.09|105.38|3.409|False|
|CorrCLIP|vaihingen|358.41|2608.90|108.55|3.302|False|
|CorrCLIP|landcoverai|259.63|2605.62|102.71|2.528|False|
|CorrCLIP|loveda|335.25|3557.50|105.86|3.167|False|
|CorrCLIP|context59|217.71|2604.89|111.84|1.947|True|
|CorrCLIP|coco_object81|242.16|2605.48|131.46|1.842|True|
|CorrCLIP|coco_stuff171|244.81|2605.46|185.87|1.317|True|
|CorrCLIP|flair1|278.40|2605.63|111.14|2.505|False|

Previous Geometry/VIP costs are retained below as historical full-method measurements. VIP timing uses its previous same-text bank, which can differ from the strongest accuracy reference. New baselines use the same seven VIP image keys; no matched-backbone/operator claim follows.

|Protocol|Previous Geometry ms / MiB|Previous VIP ms / MiB|
|---|---:|---:|
|vdd|494.15 / 5993.52|190.85 / 2332.93|
|potsdam|259.20 / 5515.77|104.11 / 1739.21|
|udd5|500.18 / 5825.38|191.15 / 2147.02|
|vaihingen|238.85 / 5519.76|108.55 / 1747.82|
|oem|249.53 / 5540.32|105.38 / 1810.02|
|loveda|243.35 / 5532.14|105.86 / 1811.12|
|landcoverai|233.39 / 5529.22|102.71 / 1762.63|
|flair1|247.00 / 5548.45|111.14 / 1868.40|
|voc20|173.80 / 2349.33|58.16 / 1743.14|
|voc21|169.76 / 2303.51|59.50 / 1745.57|
|ade150|293.66 / 6433.20|113.98 / 2174.80|
|context59|247.86 / 5567.80|111.84 / 1804.30|
|context60|222.04 / 2626.03|112.06 / 1811.42|
|coco_object81|283.66 / 5799.36|131.46 / 1944.13|
|coco_stuff171|366.10 / 6537.00|185.87 / 2187.17|

Reproduce: `python tools/training_free_baseline_experiment.py prepare`, `deploy`, `start`, `status`, `collect`.
Pinned commits, exact query files, official settings and fixed transfer policies are in `protocol.json` and `words/`.
CorrCLIP uses native online SAM2 masks, including their cost; SegEarth-OV retains pretrained SimFeatUp.
No evaluation labels select a baseline vocabulary, threshold or transfer setting. LoveDA P is read independently.
VOC20/21 and PC59/60 reuse photographs. Eight remote domains count LoveDA D once; LandCover.ai substitutes for unavailable labeled iSAID.
Runtime uses a dedicated Torch2.4/MMCV2.2 environment and an explicitly recorded MMSeg version-guard compatibility change.

## Recorded execution repairs

PTC originally failed on rectangular UDD5 crops (width grid used height) and on LoveDA all-masked proxy-attention rows (softmax of all negative infinity). An isolated source copy corrects the width index and uses self-Value only on empty attention rows. Vocabulary, calibration, thresholds and finite attention rows are unchanged. Originally finite image predictions replay exactly in the image-only validation. These are local execution-repaired results, not unmodified published baseline scores. Completed original shards are preserved; merged execution_variants record each source variant. Failed attempts and logs remain in place; partial failed prefixes without a per-image checkpoint are recomputed only in new attempt directories. Exact source diffs, hashes and image-only evidence are in [source_repairs.json](training_free_baselines_full_20261011/source_repairs.json).

CorrCLIP native online SAM2 cannot open FLAIR multi-band TIFF with PIL. The isolated decoder compatibility patch falls back to rasterio uint8 RGB bands 1,2,3 only when PIL raises UnidentifiedImageError, matching the frozen input RGB loader. SAM2 and segmentation operators/settings are unchanged. Image-only validation confirms finite FLAIR output and exact native-image prediction replay. Source diffs and evidence are also in source_repairs.json.
