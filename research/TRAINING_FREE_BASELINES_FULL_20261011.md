# Pinned native training-free baseline comparison

Authors native official config where provided; fixed untuned transfer elsewhere. Matched existing images/targets. Different pretrained backbones and external resources remain part of each full method; this is not an operator ablation. Geometry references used prior labeled development optimization.

Status: running. Only verified complete protocols receive scores.

|Protocol|Images|Geometry best measured|Strong verified local VIP|PEARL|GLA-CLIP|PTC|CorrCLIP|SegEarth-OV|
|---|---:|---:|---:|---:|---:|---:|---:|---:|
|vdd|80|60.4352|52.5765|37.9121|pending|pending|pending|pending|
|potsdam|504|56.1934|44.8995|31.5545|pending|pending|pending|pending|
|udd5|40|55.8266|44.9713|43.3762|47.7428|pending|pending|pending|
|vaihingen|113|58.8562|49.1039|26.1222|28.5356|pending|pending|pending|
|oem|384|45.1052|35.0388|27.7070|pending|pending|pending|pending|
|loveda P|1669|69.6117|56.3110|47.5212|pending|pending|pending|pending|
|loveda D|1669|45.3171|36.1457|29.6031|pending|pending|pending|pending|
|landcoverai|1602|65.5400|54.9420|30.9757|pending|pending|pending|pending|
|flair1|15700|43.6579|36.6074|19.3884|pending|pending|pending|pending|
|voc20|1449|92.7936|92.5051|86.9143|pending|pending|pending|pending|
|voc21|1449|75.8586|73.2591|64.0821|pending|pending|pending|pending|
|ade150|2000|31.3398|29.4630|19.3537|pending|pending|pending|pending|
|context59|5105|46.3075|45.9359|38.7213|pending|pending|pending|pending|
|context60|5105|44.7613|42.5987|36.5263|pending|pending|pending|pending|
|coco_object81|5000|50.3999|48.9955|38.6592|pending|pending|pending|pending|
|coco_stuff171|5000|33.7661|33.4940|26.3142|pending|pending|pending|pending|
|Eight remote mean|20092|53.8665|44.2856|30.8299|pending|pending|pending|pending|

## Standalone whole-image cost

|Method|Protocol|Median ms|Peak allocated MiB|Native config|
|---|---|---:|---:|---|
|PEARL|vdd|167.28|1152.58|False|
|PEARL|potsdam|68.77|410.15|False|
|PEARL|voc20|19.32|713.49|True|
|PEARL|voc21|20.60|1015.36|True|
|PEARL|ade150|47.41|851.92|True|
|PEARL|context60|55.92|458.84|True|
|PEARL|udd5|158.26|947.27|False|
|PEARL|oem|67.51|432.07|False|
|PEARL|vaihingen|67.64|404.27|False|
|PEARL|landcoverai|58.50|370.72|False|
|PEARL|loveda|66.97|422.07|False|
|PEARL|context59|56.64|457.05|True|
|PEARL|coco_object81|78.38|6465.10|True|
|PEARL|coco_stuff171|59.75|852.69|True|
|PEARL|flair1|62.18|382.07|False|

Reproduce: `python tools/training_free_baseline_experiment.py prepare`, `deploy`, `start`, `status`, `collect`.
Pinned commits, exact query files, official settings and fixed transfer policies are in `protocol.json` and `words/`.
CorrCLIP uses native online SAM2 masks, including their cost; SegEarth-OV retains pretrained SimFeatUp.
No evaluation labels select a baseline vocabulary, threshold or transfer setting. LoveDA P is read independently.
VOC20/21 and PC59/60 reuse photographs. Eight remote domains count LoveDA D once; LandCover.ai substitutes for unavailable labeled iSAID.
Runtime uses a dedicated Torch2.4/MMCV2.2 environment and an explicitly recorded MMSeg version-guard compatibility change.
