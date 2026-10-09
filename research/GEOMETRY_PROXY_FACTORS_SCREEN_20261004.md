# Geometry / VIPProxy: verified factor transplantation

UDD5 full40; seven domains8 complete images each. Same20 aliases, six RS templates, native512/128/Hann.
Developed validation only;96 unique keys. All four historical controls exactly replay per image.
Primary changes only the supported Geometry donors using the pinned VIP global-mean mask.
All borrowed VIP mechanisms are attributed; this factorial is not a new CVPR module claim.

| Dataset/protocol | Geometry | Geometry_BlockPrefix | SCLIP_Two | VIPProxy_Two | Geometry_VIPSupport | Geometry_VIPWeights | Geometry_VIPSupportWeights | VIPRelation_GeometryPath | Geometry_VIPPath |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| vdd/vdd | 31.9462 | 31.1897 | 30.6800 | 31.1056 | 31.7690 | 31.7838 | 31.9136 | 31.9046 | 31.1897 |
| potsdam/potsdam | 40.3528 | 42.1920 | 44.0525 | 42.7377 | 40.5152 | 34.2228 | 40.0871 | 40.3688 | 42.1920 |
| udd5/udd5 | 50.5553 | 49.6843 | 50.2837 | 49.9071 | 50.4809 | 48.1950 | 50.6500 | 50.7345 | 49.6843 |
| oem/oem | 39.3543 | 38.8317 | 37.3120 | 38.9899 | 39.2264 | 38.0636 | 39.1846 | 39.3507 | 38.8317 |
| loveda/P | 62.8254 | 67.7492 | 68.2854 | 68.5692 | 62.0852 | 59.1463 | 63.2283 | 63.3489 | 67.7492 |
| loveda/D | 38.4558 | 35.6249 | 36.5043 | 35.6670 | 38.0826 | 37.9235 | 38.4308 | 38.3442 | 35.6249 |
| landcoverai/landcoverai | 60.9049 | 60.0368 | 61.7170 | 59.3073 | 60.4927 | 61.0735 | 60.8252 | 60.8115 | 60.0368 |
| vaihingen/vaihingen | 50.2325 | 49.8551 | 52.0872 | 50.5393 | 50.4171 | 44.5987 | 50.0722 | 50.3948 | 49.8551 |
| flair1/flair1 | 38.8428 | 39.3487 | 37.6225 | 39.3436 | 38.6991 | 35.3383 | 38.6333 | 38.5862 | 39.3487 |
| Eight-domain mean, LoveDA D once | 43.8306 | 43.3454 | 43.7824 | 43.4497 | 43.7104 | 41.3999 | 43.7246 | 43.8119 | 43.3454 |

Frozen advancement gate passed: `False`. No automatic full rollout.

## Decision

Retain original Geometry. The frozen support-only primary averages43.710389
versus43.830569 (-0.120180pp), improving only Potsdam/Vaihingen among eight
primary domain entries. LoveDA P also falls0.740153pp. All primary domain losses
are modest, but the proposed transplant does not meet its advancement gate.
The best new control, VIPRelation_GeometryPath, averages43.811921 (-0.018648pp).
No control is retroactively promoted. These are complete fixed96 development
images, not full eight-dataset results or untouched independent validation.

## Mechanism Interpretation

The mask is active: it retains9.44-17.08% of all patch edges and removes5.57-26.04%
of original Geometry probability mass across domain tile means. UDD5 has1.0829%
empty VIP rows; other panel domains have none. Small metric changes do not mean
an inactive intervention. The removed mass is not selected for class correctness.

Changing cosine scale from10 to.5 while retaining the original spatial term
lowers the mean2.430663pp. Adding support to that weak-weight control recovers
2.324704pp. This is a feature-versus-space balance change, not evidence that
all smoothing is harmful or that support contributes a fixed additive gain.
Under the preserved native Geometry pathway, even the exact full VIP relation
leaves the mean almost unchanged. The relation replacement alone is not the
large missing semantic correction on this panel.

Geometry_VIPPath exactly equals Geometry_BlockPrefix in EVERY per-image
confusion matrix across all protocols. With patch-only reading in both blocks,
patch Q/K/V, tokenwise normalization and MLP never read prefix tokens. Changing
prefix evolution therefore cannot affect patch descriptors in this architecture.
The VIP normalized-prefix increment is not a separate useful patch module here.

Holding the VIP relation fixed, moving from native Geometry allocation to the
full VIP pathway changes Potsdam40.368846 ->42.737745 (+2.368899pp), while VDD
31.904585 ->31.105609 and UDD550.734522 ->49.907116 fall. The Potsdam gain coexists
with worse car precision11.5919% ->10.8078% and greater car area20.0103% ->21.6117%.
Low-vegetation recall rises37.2065% ->55.8431%. VDD water recall rises34.6753%
->41.4740%, but vehicle area rises4.1761% ->4.8515% and precision falls5.6564%
->4.8658%. These are observed nonlinear pathway contrasts, not a linear
decomposition or proof that unit-mass reading has the same effect elsewhere.

The support-only primary does not remove the main small-target false response:
VDD vehicle precision5.6966% ->5.6531%, area4.1453% ->4.1769%; Potsdam car
precision11.6716% ->11.7953%, area19.8762% ->19.6622%. Recall remains near99-100%.
On these images the principal car/vehicle deficit is precision, not missing TP.
Visual similarity can preserve a coherent but semantically incorrect donor.

## Next Research Requirement

Do not add another fixed mask, layer switch or universal patch amplification
and claim a new Geometry contribution. The unresolved target is retaining
useful water/low-vegetation evidence while preventing wrong vehicle/car evidence
from receiving the same enhancement. Geometry support is useful structure, not
semantic certification. Borrowed VIP mechanisms must remain attributed.

One distinct, still UNTESTED hypothesis is to form frozen nonlinear donor
semantic increments before geometric aggregation, instead of sending a mixed
Value descriptor into the nonlinear semantic block. It requires query-versus-
donor native-mass controls and identity-relation replay; it is not equivalent to
post-head descriptor smoothing or previously failed attention-increment paths.
Novelty and cross-domain benefit must be established before treating it as an
optimized model. This experiment does not validate that hypothesis.

## Verification And Recovery

Nine synthetic tests passed. Actual-checkpoint fp32/bf16 smoke loaded no masks,
replayed four historical operators and singleton Geometry/primary exactly,
and verified unchanged frozen weights. Collection independently checked96
unique image keys, all per-image sums, four historical per-image references,
checkpoint/vocabulary identities and unchanged Geometry settings. Every worker
and controller is terminal; original and retained coupled references untouched.

The initial mask-free smoke caught singleton/shared-cache BF16 disagreement:
the singleton cast G to raw-feature dtype instead of native Value dtype before
support normalization. It was corrected before any new label evaluation, and
the failed log is preserved. The scheduler then exited after an idle-GPU
recheck race; it was repaired to defer an occupied GPU and resumed from live
workers/complete shards, without relaunching evaluations or overwriting outputs.
The original controller log and resume log are both preserved. suite_results
wall_seconds measures the resumed controller interval, NOT total suite time;
the independent deployed costs below are the inference timing comparison.

## Independent Cost

|Method|512-window median ms|Peak allocated MiB|
|---|---:|---:|
|Geometry|22.109|3589.010|
|Geometry_VIPSupport|22.596|3593.010|

Two warmups/five synchronized repeats; one backbone/one selected head, no comparison heads.

## Class Outcomes

|Dataset/protocol|Class|Geometry IoU|Support IoU|Delta|Geometry precision/recall|Support precision/recall|Geometry/Support area|
|---|---|---:|---:|---:|---|---|---|
|vdd/vdd|other|25.3949|25.3340|-0.0609|38.9278/42.2128|38.6961/42.3187|29.1864/29.4348|
|vdd/vdd|wall|11.8982|11.8817|-0.0165|11.9838/94.3324|11.9677/94.2975|15.3444/15.3594|
|vdd/vdd|road|24.1970|24.0608|-0.1362|24.7412/91.6676|24.5972/91.6900|5.1104/5.1415|
|vdd/vdd|vegetation|62.0257|61.9845|-0.0412|84.6841/69.8628|84.7881/69.7401|17.8191/17.7660|
|vdd/vdd|vehicle|5.6965|5.6529|-0.0436|5.6966/99.9563|5.6531/99.9484|4.1453/4.1769|
|vdd/vdd|roof|60.7577|60.0712|-0.6865|88.0153/66.2377|87.9042/65.4840|20.9399/20.7278|
|vdd/vdd|water|33.6534|33.3981|-0.2553|93.0590/34.5198|93.0865/34.2476|7.4546/7.3936|
|potsdam/potsdam|impervious surface|51.6875|51.8644|0.1769|75.3909/62.1781|75.5283/62.3404|29.8644/29.8879|
|potsdam/potsdam|building|78.2026|78.2347|0.0321|80.6769/96.2262|80.7567/96.1615|14.5546/14.5304|
|potsdam/potsdam|low vegetation|33.4221|33.7953|0.3732|72.4703/38.2826|72.8968/38.6520|10.1329/10.1708|
|potsdam/potsdam|tree|62.5813|62.8325|0.2512|90.6538/66.8975|90.5205/67.2580|17.6734/17.7948|
|potsdam/potsdam|car|11.6582|11.7812|0.1230|11.6716/99.0262|11.7953/98.9979|19.8762/19.6622|
|potsdam/potsdam|clutter|4.5652|4.5831|0.0179|7.7447/10.0073|7.7501/10.0844|7.8986/7.9539|
|udd5/udd5|vegetation|82.4937|82.4272|-0.0665|96.4862/85.0488|96.4470/85.0085|26.1079/26.1062|
|udd5/udd5|building|83.9035|83.7832|-0.1203|88.3604/94.3293|88.4395/94.0875|41.9106/41.7657|
|udd5/udd5|road|45.7856|45.6237|-0.1619|70.6780/56.5218|70.4418/56.4260|10.7242/10.7420|
|udd5/udd5|vehicle|10.0092|10.0541|0.0449|10.0323/97.7477|10.0778/97.7175|7.7994/7.7618|
|udd5/udd5|other|30.5846|30.5164|-0.0682|52.8537/42.0591|52.4043/42.2175|13.4578/13.6244|
|oem/oem|bareland|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|10.4255/10.6287|
|oem/oem|rangeland|47.7780|47.6578|-0.1202|65.2945/64.0414|65.2046/63.9117|18.7786/18.7664|
|oem/oem|developed space|28.5294|28.5166|-0.0128|61.5617/34.7130|62.6579/34.3553|12.6303/12.2815|
|oem/oem|road|40.8291|40.8754|0.0463|45.7530/79.1399|45.4888/80.1208|7.8925/8.0367|
|oem/oem|tree|56.0657|56.7154|0.6497|87.4475/60.9727|87.2586/61.8363|17.7308/18.0209|
|oem/oem|water|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|0.0647/0.0721|
|oem/oem|agriculture land|79.1353|77.3780|-1.7573|90.6413/86.1766|90.3382/84.3593|15.9338/15.6501|
|oem/oem|building|62.4971|62.6683|0.1712|65.5844/92.9954|65.6950/93.1516|16.5437/16.5436|
|loveda/P|building|64.4987|62.8548|-1.6439|65.3671/97.9819|63.6510/98.0487|0.9799/1.0071|
|loveda/P|road|66.9348|66.8020|-0.1328|67.6522/98.4404|67.5645/98.3385|10.3611/10.3638|
|loveda/P|water|81.1851|81.3076|0.1225|86.7638/92.6613|86.9480/92.6111|22.3129/22.2535|
|loveda/P|barren|24.3403|24.1792|-0.1611|63.8735/28.2261|60.4287/28.7279|3.2301/3.4749|
|loveda/P|tree|53.7307|52.2677|-1.4630|64.5184/76.2666|62.1058/76.7419|13.0831/13.6761|
|loveda/P|farm|86.2626|85.1000|-1.1626|95.3302/90.0685|95.4351/88.7110|50.0330/49.2247|
|loveda/D|background|34.1983|33.3377|-0.8606|65.7504/41.6109|64.5718/40.8006|28.2483/28.2038|
|loveda/D|building|31.1862|30.7092|-0.4770|31.8582/93.6650|31.3015/94.1963|1.0641/1.0892|
|loveda/D|road|45.4818|45.0827|-0.3991|45.8348/98.3352|45.4502/98.2379|8.4577/8.5209|
|loveda/D|water|61.9193|61.8016|-0.1177|65.9551/91.0065|65.8851/90.8852|15.9605/15.9562|
|loveda/D|barren|14.5296|14.9710|0.4414|59.0631/16.1567|57.1150/16.8670|1.1070/1.1951|
|loveda/D|tree|26.8904|26.5129|-0.3775|30.0530/71.8726|29.4550/72.6350|14.6543/15.1104|
|loveda/D|farm|54.9851|54.1629|-0.8222|69.5721/72.3948|69.5557/70.9932|30.5080/29.9244|
|vaihingen/vaihingen|impervious surface|49.2002|49.3755|0.1753|82.3971/54.9789|82.3578/55.2156|20.4283/20.5260|
|vaihingen/vaihingen|building|74.6610|74.9561|0.2951|75.5001/98.5334|75.7976/98.5405|28.1288/28.0204|
|vaihingen/vaihingen|low vegetation|46.7115|47.1613|0.4498|92.0732/48.6687|91.9787/49.1842|13.4530/13.6095|
|vaihingen/vaihingen|tree|71.8329|71.7096|-0.1233|82.5509/84.6923|82.3181/84.7664|21.4401/21.5195|
|vaihingen/vaihingen|car|8.7566|8.8831|0.1265|8.7725/97.9748|8.8990/98.0342|16.5499/16.3246|
|landcoverai/landcoverai|background|82.6090|82.0344|-0.5746|96.6505/85.0437|96.3797/84.6426|59.6226/59.5082|
|landcoverai/landcoverai|building|34.9562|35.0115|0.0553|35.0436/99.2919|35.1031/99.2604|4.2927/4.2841|
|landcoverai/landcoverai|woodland|78.3638|77.2252|-1.1386|86.4992/89.2841|85.6679/88.6827|22.0960/22.1601|
|landcoverai/landcoverai|water|93.6635|93.3379|-0.3256|93.6635/100.0000|93.3379/100.0000|8.8309/8.8617|
|landcoverai/landcoverai|road|14.9318|14.8545|-0.0773|15.6288/77.0019|15.5441/77.0019|5.1578/5.1858|
|flair1/flair1|building|49.6979|49.8461|0.1482|50.7258/96.0825|50.8445/96.2101|13.5991/13.5854|
|flair1/flair1|pervious surface|57.0621|56.1059|-0.9562|92.1299/59.9861|91.9458/59.0059|11.1728/11.0122|
|flair1/flair1|impervious surface|52.6509|52.7728|0.1219|62.6245/76.7765|62.8435/76.7069|20.0846/19.9964|
|flair1/flair1|bare soil|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|3.4411/3.5642|
|flair1/flair1|water|73.2328|72.7309|-0.5019|77.1432/93.5263|76.1676/94.1586|5.3781/5.4838|
|flair1/flair1|coniferous|43.0233|42.7165|-0.3068|61.7114/58.6897|60.8539/58.9019|0.5346/0.5441|
|flair1/flair1|deciduous|54.0229|54.2383|0.2154|78.9723/63.0995|78.5653/63.6583|13.6004/13.7919|
|flair1/flair1|brushwood|18.6136|18.3195|-0.2941|23.4212/47.5559|23.4339/45.6345|9.9866/9.5779|
|flair1/flair1|vineyard|NA|NA|NA|0.0000/0.0000|0.0000/0.0000|0.0000/0.0000|
|flair1/flair1|herbaceous vegetation|60.2329|60.7358|0.5029|94.3169/62.5013|94.2458/63.0748|21.2289/21.4398|
|flair1/flair1|agricultural land|18.7339|18.2246|-0.5093|21.6468/58.1977|20.9696/58.1977|0.8198/0.8463|
|flair1/flair1|plowed land|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|0.1540/0.1578|

## Support And Transitions

|Dataset/protocol|VIP supported edges|Empty rows|Removed Geometry mass|Beneficial|Harmful|
|---|---:|---:|---:|---:|---:|
|vdd/vdd|0.130533|0.000000|0.158659|245300|497595|
|potsdam/potsdam|0.146247|0.000000|0.073799|43057|26086|
|udd5/udd5|0.155635|0.010829|0.133831|757332|1167298|
|oem/oem|0.110239|0.000000|0.174610|48036|57780|
|loveda/P|0.094419|0.000000|0.213094|10745|40361|
|loveda/D|0.094419|0.000000|0.213094|44260|103221|
|vaihingen/vaihingen|0.170801|0.000000|0.055744|31270|13664|
|landcoverai/landcoverai|0.099285|0.000000|0.260369|4988|13397|
|flair1/flair1|0.112905|0.000000|0.138225|11345|10441|

Every arm retains full per-class metrics, transition tensors and effective donor diagnostics in merged.json.
A fixed96 rank is not a full-domain rank; broader locked validation is required for positive findings.
Original/best models, old results and paused automations remain unchanged.
