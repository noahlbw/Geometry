# Geometry staged readout: verified fixed96 screen

UDD5 full40; seven domains8 complete images each. Same20 aliases, six RS templates, native512/128/Hann.
Developed validation only. All96 keys covered; four historical controls exactly replay per image.
Primary is fixed context-preserve block0 / patch-only unit-mass block1; no VIP sparse relation or prefix increment.
A performance improvement alone does not establish a distinct CVPR contribution.

| Dataset/protocol | Geometry | SCLIP_Two | VIPProxy_Two | Geometry_BlockPrefix | Geometry_Staged | Geometry_ReverseStage | Geometry_PrefixUnit | Geometry_PatchMass |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| potsdam/potsdam | 40.3528 | 44.0525 | 42.7377 | 42.1920 | 38.9475 | 43.1612 | 42.7687 | 40.3529 |
| oem/oem | 39.3543 | 37.3120 | 38.9899 | 38.8317 | 38.0262 | 39.0669 | 39.5852 | 37.9853 |
| loveda/P | 62.8254 | 68.2854 | 68.5692 | 67.7492 | 67.4142 | 64.0382 | 66.4841 | 64.6918 |
| loveda/D | 38.4558 | 36.5043 | 35.6670 | 35.6249 | 36.8004 | 37.0399 | 37.6150 | 36.1602 |
| vaihingen/vaihingen | 50.2325 | 52.0872 | 50.5393 | 49.8551 | 48.2955 | 50.8242 | 50.4668 | 49.2213 |
| landcoverai/landcoverai | 60.9049 | 61.7170 | 59.3073 | 60.0368 | 61.6955 | 60.5615 | 61.1440 | 58.8228 |
| flair1/flair1 | 38.8428 | 37.6225 | 39.3436 | 39.3487 | 36.8755 | 41.3898 | 37.9859 | 38.6502 |
| vdd/vdd | 31.9462 | 30.6800 | 31.1056 | 31.1897 | 31.4876 | 31.8332 | 30.6724 | 32.3123 |
| udd5/udd5 | 50.5553 | 50.2837 | 49.9071 | 49.6843 | 50.9063 | 49.3356 | 51.1131 | 49.9024 |
| Eight-domain mean, LoveDA D once | 43.8306 | 43.7824 | 43.4497 | 43.3454 | 42.8793 | 44.1515 | 43.9189 | 42.9259 |

Frozen promotion gate passed: `False`. No automatic full rollout.

## Decision

Retain original Geometry. The fixed primary loses0.951241 pp in the equal-domain mean and improves
only UDD5 and LandCover.ai among the eight domains. LoveDA P improves4.588842 pp, but D falls1.655445 pp.
This rejects the proposed context-first/final-patch-only ordering on this development panel.

Reverse staging averages44.151545 (+0.320976 pp), improving Potsdam/Vaihingen/FLAIR-1 but lowering the other five domains.
PrefixUnit averages43.918891 (+0.088322 pp), improving five domains but lowering VDD/LoveDA D/FLAIR-1.
Neither control is relabeled as the primary or promoted as a finished model.
Runtime is effectively unchanged, so the failure is in readout accuracy rather than solver overhead.

## Mechanism And Next Target

Removing special-token reading lowers the mean by0.904636 pp at native patch mass and0.573489 pp at unit patch mass.
Thus the special-token term is not uniformly harmful. Increasing patch mass has different consequences when
the special term is retained or removed; attention row mass alone cannot explain semantic correctness.
The original block1 native patch mass averages0.3587 on VDD,0.3716 on Potsdam and0.3697 on UDD5.
The primary simultaneously removes the special increment and increases patch reading to unit mass.
Factorial controls distinguish these two interventions; they do not make downstream predictions linear.

In the primary, VDD vehicle area rises4.1453% ->6.0043%, precision falls5.6966% ->3.9345%,
and recall remains effectively100%. Potsdam car area rises19.8762% ->27.9434%, precision falls11.6716% ->8.3829%,
and recall rises99.0262% ->99.9909%. The car/vehicle problem on this panel is false activation, not insufficient recall.
Potsdam low-vegetation recall rises38.2826% ->50.8606%, demonstrating the competing benefit of greater local coverage.
UDD5 road IoU rises45.7856 ->49.3989 while vehicle IoU falls10.0092 ->9.3457.
Any next readout must handle this class-dependent tradeoff; a fixed layer switch or scalar boost is insufficient here.

For the CVPR objective, the unresolved contribution is how geometric support controls class-discriminative evidence
without discarding useful context. These results do not validate a new semantic gate, and no such gate was fitted.
Borrowing patch-only reading from existing dense-readout work and changing layer order alone is not a strong novelty claim.
The original Geometry relation remains fixed and useful, but its stable superiority over matched nearest operators is still unproven.

The fixed96 panel is not a complete-domain ranking estimate: historical BlockPrefix improves the full-domain mean
46.1508 ->46.8188, yet lowers this panel43.8306 ->43.3454. Broader locked validation is necessary before claiming
general improvement or deterioration. Current fixed-primary promotion fails; do not relax its gate against these labels.

## Verification

Seven synthetic tests passed. Actual-checkpoint fp32/bf16 Geometry and BlockPrefix replay errors are exactly0;
weights and numerical caches remain unchanged and smoke loaded no masks. Four historical operators exactly replay
all96 per-image confusion matrices. All new workers/controllers finished and GPUs0-7 are idle;
old outputs and paused automations remain unchanged.

## Independent Cost

| Method |512-window median ms |Peak allocated MiB |
|---|---:|---:|
|Geometry|22.478|3590.010|
|Geometry_Staged|22.371|3590.010|

Two warmups/five synchronized repeats; one backbone/one selected head, no comparison heads.

## Allocation Factorial

Differences below are mIoU responses after nonlinear head evolution, not linear decompositions of predictions.

| Effect |Equal-domain mean response pp |
|---|---:|
|Remove special reading at native patch mass|-0.904636|
|Remove special reading at unit patch mass|-0.573489|
|Increase patch mass with special reading retained|+0.088322|
|Increase patch mass without special reading|+0.419469|
|Final-block-only change vs original|-0.951241|
|Context-first vs reversed staging|-1.272216|

## Class Outcomes

|Dataset/protocol|Class|Geometry IoU|Staged IoU|Delta|Geometry precision/recall|Staged precision/recall|Geometry/Staged area|
|---|---|---:|---:|---:|---|---|---|
|vdd/vdd|other|25.3949|20.5107|-4.8842|38.9278/42.2128|37.6623/31.0528|29.1864/22.1917|
|vdd/vdd|wall|11.8982|12.9455|1.0473|11.9838/94.3324|13.0491/94.2197|15.3444/14.0749|
|vdd/vdd|road|24.1970|21.6928|-2.5042|24.7412/91.6676|22.1939/90.5728|5.1104/5.6289|
|vdd/vdd|vegetation|62.0257|63.3121|1.2864|84.6841/69.8628|79.6155/75.5606|17.8191/20.4993|
|vdd/vdd|vehicle|5.6965|3.9345|-1.7620|5.6966/99.9563|3.9345/99.9996|4.1453/6.0043|
|vdd/vdd|roof|60.7577|61.7750|1.0173|88.0153/66.2377|83.0761/70.6683|20.9399/23.6688|
|vdd/vdd|water|33.6534|36.2425|2.5891|93.0590/34.5198|93.9969/37.1013|7.4546/7.9321|
|potsdam/potsdam|impervious surface|51.6875|48.6822|-3.0053|75.3909/62.1781|81.8449/54.5758|29.8644/24.1459|
|potsdam/potsdam|building|78.2026|75.7671|-2.4355|80.6769/96.2262|78.0103/96.3435|14.5546/15.0704|
|potsdam/potsdam|low vegetation|33.4221|41.2694|7.8473|72.4703/38.2826|68.6368/50.8606|10.1329/14.2140|
|potsdam/potsdam|tree|62.5813|56.8715|-5.7098|90.6538/66.8975|89.8988/60.7538|17.6734/16.1851|
|potsdam/potsdam|car|11.6582|8.3829|-3.2753|11.6716/99.0262|8.3829/99.9909|19.8762/27.9434|
|potsdam/potsdam|clutter|4.5652|2.7122|-1.8530|7.7447/10.0073|9.2527/3.6951|7.8986/2.4412|
|udd5/udd5|vegetation|82.4937|83.3816|0.8879|96.4862/85.0488|95.9165/86.4504|26.1079/26.6958|
|udd5/udd5|building|83.9035|84.4531|0.5496|88.3604/94.3293|88.3415/95.0464|41.9106/42.2382|
|udd5/udd5|road|45.7856|49.3989|3.6133|70.6780/56.5218|69.9413/62.7130|10.7242/12.0243|
|udd5/udd5|vehicle|10.0092|9.3457|-0.6635|10.0323/97.7477|9.3632/98.0339|7.7994/8.3812|
|udd5/udd5|other|30.5846|27.9522|-2.6324|52.8537/42.0591|56.5019/35.6166|13.4578/10.6605|
|oem/oem|bareland|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|10.4255/12.9095|
|oem/oem|rangeland|47.7780|50.1766|2.3986|65.2945/64.0414|66.7952/66.8516|18.7786/19.1622|
|oem/oem|developed space|28.5294|16.1937|-12.3357|61.5617/34.7130|61.2619/18.0410|12.6303/6.5963|
|oem/oem|road|40.8291|40.1723|-0.6568|45.7530/79.1399|43.4475/84.1998|7.8925/8.8427|
|oem/oem|tree|56.0657|55.8154|-0.2503|87.4475/60.9727|88.5008/60.1798|17.7308/17.2919|
|oem/oem|water|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|0.0647/0.0383|
|oem/oem|agriculture land|79.1353|80.5064|1.3711|90.6413/86.1766|87.0367/91.4748|15.9338/17.6139|
|oem/oem|building|62.4971|61.3454|-1.1517|65.5844/92.9954|63.3049/95.1966|16.5437/17.5451|
|loveda/P|building|64.4987|73.7390|9.2403|65.3671/97.9819|75.4541/97.0096|0.9799/0.8405|
|loveda/P|road|66.9348|71.2333|4.2985|67.6522/98.4404|72.1640/98.2217|10.3611/9.6917|
|loveda/P|water|81.1851|82.1096|0.9245|86.7638/92.6613|87.3205/93.2246|22.3129/22.3054|
|loveda/P|barren|24.3403|26.2434|1.9031|63.8735/28.2261|74.4803/28.8363|3.2301/2.8299|
|loveda/P|tree|53.7307|60.9011|7.1704|64.5184/76.2666|75.7772/75.6230|13.0831/11.0453|
|loveda/P|farm|86.2626|90.2589|3.9963|95.3302/90.0685|94.5850/95.1769|50.0330/53.2872|
|loveda/D|background|34.1983|16.7198|-17.4785|65.7504/41.6109|69.6074/18.0365|28.2483/11.5659|
|loveda/D|building|31.1862|32.1348|0.9486|31.8582/93.6650|32.5632/96.0674|1.0641/1.0678|
|loveda/D|road|45.4818|47.5291|2.0473|45.8348/98.3352|47.9438/98.2125|8.4577/8.0756|
|loveda/D|water|61.9193|62.4274|0.5081|65.9551/91.0065|65.4383/93.1357|15.9605/16.4629|
|loveda/D|barren|14.5296|19.1379|4.6083|59.0631/16.1567|47.3484/24.3118|1.1070/2.0779|
|loveda/D|tree|26.8904|22.7364|-4.1540|30.0530/71.8726|24.6435/74.6058|14.6543/18.5506|
|loveda/D|farm|54.9851|56.9173|1.9322|69.5721/72.3948|61.4727/88.4800|30.5080/42.1992|
|vaihingen/vaihingen|impervious surface|49.2002|40.1023|-9.0979|82.3971/54.9789|86.3696/42.8118|20.4283/15.1757|
|vaihingen/vaihingen|building|74.6610|76.5731|1.9121|75.5001/98.5334|77.1946/98.9595|28.1288/27.6303|
|vaihingen/vaihingen|low vegetation|46.7115|46.6026|-0.1089|92.0732/48.6687|92.6573/48.3897|13.4530/13.2916|
|vaihingen/vaihingen|tree|71.8329|71.8004|-0.0325|82.5509/84.6923|83.5270/83.6448|21.4401/20.9274|
|vaihingen/vaihingen|car|8.7566|6.3992|-2.3574|8.7725/97.9748|6.4022/99.2612|16.5499/22.9750|
|landcoverai/landcoverai|background|82.6090|82.5835|-0.0255|96.6505/85.0437|95.9315/85.5809|59.6226/60.4490|
|landcoverai/landcoverai|building|34.9562|37.6305|2.6743|35.0436/99.2919|37.7687/99.0369|4.2927/3.9728|
|landcoverai/landcoverai|woodland|78.3638|75.7306|-2.6332|86.4992/89.2841|84.0670/88.4219|22.0960/22.5157|
|landcoverai/landcoverai|water|93.6635|95.1066|1.4431|93.6635/100.0000|95.1066/100.0000|8.8309/8.6969|
|landcoverai/landcoverai|road|14.9318|17.4265|2.4947|15.6288/77.0019|18.3990/76.7286|5.1578/4.3656|
|flair1/flair1|building|49.6979|54.7753|5.0774|50.7258/96.0825|54.8795/99.6545|13.5991/13.0371|
|flair1/flair1|pervious surface|57.0621|58.8624|1.8003|92.1299/59.9861|92.8464/61.6588|11.1728/11.3957|
|flair1/flair1|impervious surface|52.6509|53.7144|1.0635|62.6245/76.7765|63.8900/77.1303|20.0846/19.7775|
|flair1/flair1|bare soil|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|3.4411/3.1560|
|flair1/flair1|water|73.2328|74.4388|1.2060|77.1432/93.5263|80.6008/90.6863|5.3781/4.9911|
|flair1/flair1|coniferous|43.0233|43.4141|0.3908|61.7114/58.6897|63.0771/58.2060|0.5346/0.5187|
|flair1/flair1|deciduous|54.0229|55.3984|1.3755|78.9723/63.0995|79.1234/64.8821|13.6004/13.9580|
|flair1/flair1|brushwood|18.6136|22.2741|3.6605|23.4212/47.5559|28.2044/51.4413|9.9866/8.9705|
|flair1/flair1|vineyard|NA|0.0000|NA|0.0000/0.0000|0.0000/0.0000|0.0000/0.0003|
|flair1/flair1|herbaceous vegetation|60.2329|63.3428|3.1099|94.3169/62.5013|92.5648/66.7384|21.2289/23.0971|
|flair1/flair1|agricultural land|18.7339|16.2858|-2.4481|21.6468/58.1977|18.4432/58.1977|0.8198/0.9622|
|flair1/flair1|plowed land|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|0.1540/0.1358|

## Diagnostics

|Dataset/protocol|Beneficial changes|Harmful changes|Foreground/non-residual Geometry|Foreground/non-residual Staged|
|---|---:|---:|---:|---:|
|vdd/vdd|3330498|3367632|33.0381|33.3171|
|potsdam/potsdam|275089|447926|47.5104|46.1946|
|udd5/udd5|8956248|7021941|55.5480|56.6448|
|oem/oem|197452|352704|39.3543|38.0262|
|loveda/P|142284|15276|62.8254|67.4142|
|loveda/D|464866|883193|39.1654|40.1472|
|vaihingen/vaihingen|99546|409813|50.2325|48.2955|
|landcoverai/landcoverai|32352|28730|55.4788|56.4736|
|flair1/flair1|75007|26276|38.8428|36.8755|

Stage norms, margins, complete per-class confusion and transition tensors are preserved in each downloaded merged.json.
Original Geometry, old full evaluations and paused automations remain unchanged.
