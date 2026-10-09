# Geometry / CSA allocation: verified264 mechanism study

UDD5 full40; other domains32 each. Exact same developed264 keys as prior row-composition study.
Fixed20/six RS templates/native512/overlap128(step384)/Hann. No target fitting, model updates or domain routing.
CSA is borrowed SCLIP machinery; these are matched operator adaptations, not official complete systems or novelty/SOTA claims.

|Dataset/protocol|Geometry|Geometry_BlockPrefix|SCLIP_Two|VIPProxy_Two|Geometry_DoubleRow|CSA_NativePrefixSum|CSA_NativePrefixMean|Geometry_CSAAllocationMean|Geometry_CSAAllocationSum|Geometry_CSAMassMean|Geometry_CSAMassSum|Geometry_CSAConditional|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|vdd/vdd|40.6839|41.7909|42.2973|41.8286|42.1730|42.3010|41.5835|41.7909|41.6345|41.7910|41.6344|40.9567|
|potsdam/potsdam|40.5893|42.7069|45.7020|42.9724|43.8195|45.7030|41.6034|42.7068|46.4644|42.7070|46.4646|38.3756|
|udd5/udd5|50.5553|49.6843|50.2837|49.9071|51.5339|50.2770|49.3463|49.6835|49.8784|49.6854|49.8813|49.7289|
|oem/oem|42.9602|43.6819|42.9439|43.8081|43.4952|42.9451|43.4110|43.6826|43.3305|43.6824|43.3306|42.9059|
|loveda/P|65.0902|68.4444|65.6949|68.6542|63.9782|65.7229|67.9474|68.4442|66.9002|68.4433|66.9014|65.1394|
|loveda/D|39.8728|37.0786|37.3274|36.9445|38.0109|37.3453|37.3559|37.0773|36.9028|37.0775|36.9023|40.2439|
|vaihingen/vaihingen|49.0088|49.4759|50.3555|50.2703|50.3006|50.3568|48.3328|49.4755|51.5819|49.4758|51.5825|47.5723|
|landcoverai/landcoverai|59.7998|62.1314|62.9003|61.5546|61.7107|62.9012|62.4645|62.1311|61.8862|62.1347|61.8878|60.3159|
|flair1/flair1|46.2616|46.7937|45.4426|46.9027|46.7144|45.4435|47.0003|46.7954|45.3210|46.7941|45.3216|45.4157|
|Eight-domain mean, LoveDA D once|46.2165|46.6679|47.1566|46.7735|47.2198|47.1591|46.3872|46.6679|47.1250|46.6685|47.1256|45.6894|

Frozen gate: `False`. No automatic full rollout or control-to-primary promotion.

## Decision

Retain original Geometry and the retained coupled reference. The primary improves
the developed eight-domain mean46.216453 ->47.124962 (+0.908509pp), but is below
SCLIP_Two47.156590, CSA_NativePrefixSum47.159113 and the simple
Geometry_DoubleRow47.219782. It improves six/eight primary domain entries,
losing0.676929pp on UDD5 and2.969962pp on LoveDA D. The frozen gate fails;
neither the candidate nor the favorable doubling control is promoted.

Conditional paired source-group bootstrap2000 gives primary-minus-Geometry
+0.908509pp [0.050595,1.615687], primary-minus-SCLIP -0.031628pp
[-0.281951,0.287259], and primary-minus-DoubleRow -0.094820pp
[-0.929175,0.650581]. These are reused-development, filename-group conditional
intervals, not selection-adjusted significance or independent validation.
See geometry_csa_allocation_screen_20261005/PAIRED_UNCERTAINTY.md.

## Mechanism And Next Hypothesis

The proposed CSA allocation is nearly saturated: at primary input states its
normalized patch mass averages0.999643-0.999816 across domain/block means.
The mean-strength Geometry_CSAAllocationMean almost reproduces BlockPrefix:
their eight-domain means differ only -0.000052pp. Changing the conditional
special-token content changes the primary mean by only -0.000664pp. Thus the
new allocation does not provide substantial adaptive special-versus-patch
selection here; much of its behavior resembles unit-mass patch-only Geometry
with changed read strength. This is an observed mechanism limit, not semantic
reliability inferred from attention mass.

At the matched CSA allocation/strength/prefix construction, substituting G for
the CSA conditional relation improves Potsdam0.761373pp and Vaihingen1.225125pp,
but loses VDD0.666522pp and LandCover.ai1.014999pp. Geometry contributes different
class tradeoffs rather than uniformly better conditional semantics. Doubling
the original read improves seven/eight primary domain entries but loses
LoveDA D1.861905pp and P1.111944pp; it is not a safe universal fix either.

The primary improves VDD vehicle IoU10.1560 ->18.6353, reducing vehicle area
4.6223% ->2.5137% while retaining98.9631% recall. Potsdam car improves
9.9333 ->15.4413 and low vegetation37.5641 ->51.0997. However the same change
reduces LoveDA D background IoU33.7840 ->3.6081 and recall to3.7080%, expanding
farm area27.9532% ->52.0648%. LoveDA D foreground mean improves40.8876 ->42.4519,
so its overall loss must not be misdescribed as all foreground evidence failing.
VDD wall/other, OEM developed space and FLAIR brushwood also deteriorate.

A distinct next mechanism study will keep the original Geometry/special-token
read and amplify only surviving centered patch differences when Geometry
contracts their spatial variance. Its added Value-read correction will have
zero window mean, keeping the original aggregate read before the frozen
projection at a fixed input. This is a contrast-compensation hypothesis, not
a semantic correctness certificate, a new-priority claim or an accepted model.
Same-strength whole-read, uncentered and fixed-gain controls must distinguish
it from generic amplification. Nonlinear head evolution can still change
class areas and background; preservation of an intermediate mean is not a
guarantee of retained mIoU. No thresholds or per-domain rules will be fitted.

The verified current primary costs24.541ms versus21.746ms (+12.85%) and peaks
at3716.254 versus3589.010MiB. All14 workers and controller are terminal; the
eight GPUs have no compute processes at collection. Historical results and
paused automations remain unchanged.

## Frozen gate

```json
{
  "mean_vs_Geometry": true,
  "mean_vs_SCLIP_Two": false,
  "mean_vs_VIPProxy_Two": true,
  "mean_vs_CSA_NativePrefixSum": false,
  "mean_vs_Geometry_DoubleRow": false,
  "retain_udd5_udd5": true,
  "retain_vdd_vdd": true,
  "focus_vdd": true,
  "retain_landcoverai_landcoverai": true,
  "retain_flair1_flair1": true,
  "retain_potsdam_potsdam": true,
  "focus_potsdam": true,
  "retain_oem_oem": true,
  "retain_vaihingen_vaihingen": true,
  "retain_loveda_P": true,
  "retain_loveda_D": false,
  "latency_ratio_at_most_1_15": true
}
```

## Mechanism contrasts

Whole-system mIoU contrasts include evolving states; not an additive causal decomposition.

|Dataset/protocol|GeoDouble-Geo|CSASum-CSAMean|GeoAllocSum-GeoAllocMean|GeoMassSum-GeoMassMean|Primary-CSASum|Primary-GeoMassSum|CSASum-SCLIP|
|---|---:|---:|---:|---:|---:|---:|---:|
|vdd/vdd|+1.4891|+0.7175|-0.1564|-0.1566|-0.6665|+0.0002|+0.0037|
|potsdam/potsdam|+3.2302|+4.0996|+3.7576|+3.7576|+0.7614|-0.0002|+0.0010|
|udd5/udd5|+0.9786|+0.9306|+0.1949|+0.1959|-0.3986|-0.0029|-0.0067|
|oem/oem|+0.5351|-0.4659|-0.3521|-0.3517|+0.3853|-0.0002|+0.0013|
|loveda/P|-1.1119|-2.2244|-1.5440|-1.5418|+1.1773|-0.0012|+0.0280|
|loveda/D|-1.8619|-0.0106|-0.1745|-0.1753|-0.4425|+0.0005|+0.0179|
|vaihingen/vaihingen|+1.2918|+2.0240|+2.1064|+2.1067|+1.2251|-0.0006|+0.0013|
|landcoverai/landcoverai|+1.9109|+0.4368|-0.2449|-0.2470|-1.0150|-0.0015|+0.0009|
|flair1/flair1|+0.4528|-1.5568|-1.4744|-1.4726|-0.1225|-0.0006|+0.0008|

## Foreground and independent cost

|Dataset/protocol|Metric|Geometry|Primary|CSA_NativePrefixSum|
|---|---|---:|---:|---:|
|udd5/udd5|non_residual_mean_iou_percent|55.5480|56.1636|56.8467|
|loveda/D|foreground_mean_iou_percent|40.8876|42.4519|43.2719|
|landcoverai/landcoverai|foreground_mean_iou_percent|53.7911|56.3611|57.4976|
|landcoverai/landcoverai|non_residual_mean_iou_percent|53.7911|56.3611|57.4975|

|Method|512-window median ms|Peak allocated MiB|
|---|---:|---:|
|Geometry|21.746|3589.010|
|Geometry_DoubleRow|21.736|3589.010|
|Geometry_CSAAllocationSum|24.541|3716.254|
|SCLIP_Two|23.073|3617.304|

Five warmups/15 synchronized512-window repeats; one backbone/one selected head; traces and comparison heads disabled.

## Per-class outcomes

|Dataset/protocol|Class|Geometry IoU|Primary IoU|CSA source IoU|Primary-Geometry|Primary precision/recall|Geometry/Primary area|
|---|---|---:|---:|---:|---:|---|---|
|vdd/vdd|other|11.0738|3.6275|2.9612|-7.4463|9.9758/5.3929|17.3211/7.9006|
|vdd/vdd|wall|22.3327|13.5902|13.5825|-8.7425|13.8985/85.9705|7.9351/13.2140|
|vdd/vdd|road|46.1955|45.6061|44.5721|-0.5894|47.0545/93.6773|9.5603/9.9855|
|vdd/vdd|vegetation|72.6499|77.7114|79.8336|5.0615|93.6571/82.0285|38.3752/42.7367|
|vdd/vdd|vehicle|10.1560|18.6353|21.2847|8.4793|18.6717/98.9631|4.6223/2.5137|
|vdd/vdd|roof|65.9570|70.0753|71.5011|4.1183|85.2054/79.7829|13.3154/14.2182|
|vdd/vdd|water|56.4221|62.1958|62.3721|5.7737|94.3712/64.5919|8.8706/9.4313|
|potsdam/potsdam|impervious surface|53.8649|59.5611|62.3515|5.6962|80.2930/69.7588|29.0511/30.0859|
|potsdam/potsdam|building|82.2362|81.7805|77.8802|-0.4557|82.4011/99.0875|24.5007/25.5475|
|potsdam/potsdam|low vegetation|37.5641|51.0997|45.7436|13.5356|89.9920/54.1785|7.5714/11.0960|
|potsdam/potsdam|tree|53.6529|63.7366|63.5986|10.0837|79.6367/76.1466|13.3723/17.9166|
|potsdam/potsdam|car|9.9333|15.4413|18.8969|5.5080|15.4477/99.7335|20.3213/13.0959|
|potsdam/potsdam|clutter|6.2844|7.1673|5.7475|0.8829|21.2855/9.7521|5.1832/2.2581|
|udd5/udd5|vegetation|82.4937|81.8473|82.5547|-0.6464|94.8087/85.6875|26.1079/26.7694|
|udd5/udd5|building|83.9035|80.6120|79.4018|-3.2915|81.9791/97.9732|41.9106/46.9179|
|udd5/udd5|road|45.7856|47.8935|48.0830|2.1079|69.5744/60.5819|10.7242/11.6769|
|udd5/udd5|vehicle|10.0092|14.3017|17.3471|4.2925|14.3648/97.0164|7.7994/5.4063|
|udd5/udd5|other|30.5846|24.7374|23.9984|-5.8472|56.1705/30.6545|13.4578/9.2295|
|oem/oem|bareland|9.5228|8.5134|8.6553|-1.0094|8.7250/77.8274|9.0985/10.2993|
|oem/oem|rangeland|25.2283|28.8134|27.7280|3.5851|56.1417/37.1831|13.2907/12.5641|
|oem/oem|developed space|29.0591|21.3122|21.5957|-7.7469|47.0730/28.0286|19.6194/12.3324|
|oem/oem|road|37.3322|39.2286|38.7675|1.8964|49.4982/65.4071|9.3608/9.5899|
|oem/oem|tree|66.5893|65.2283|64.6790|-1.3610|83.6646/74.7480|16.9292/17.2186|
|oem/oem|water|63.0704|69.3277|69.6249|6.2573|76.1974/88.4922|1.7993/1.5877|
|oem/oem|agriculture land|52.7895|57.8959|55.9185|5.1064|81.1999/66.8580|10.0235/13.5471|
|oem/oem|building|60.0899|56.3241|56.5920|-3.7658|59.3768/91.6356|19.8788/22.8608|
|loveda/P|building|70.5532|60.6277|60.6453|-9.9255|61.3616/98.0655|15.3451/17.8034|
|loveda/P|road|49.0969|65.8848|62.9430|16.7879|85.0613/74.5058|9.2374/5.7394|
|loveda/P|water|80.6240|75.9367|74.1642|-4.6873|99.3097/76.3396|17.6431/16.4137|
|loveda/P|barren|48.1737|46.2449|50.0887|-1.9288|57.2533/70.6327|5.9565/6.2592|
|loveda/P|tree|63.8268|76.5515|72.4065|12.7247|87.3209/86.1246|12.9850/10.0222|
|loveda/P|farm|78.2664|76.1557|74.0898|-2.1107|88.3983/84.6126|38.8329/43.7620|
|loveda/D|background|33.7840|3.6081|1.7858|-30.1759|57.2623/3.7080|33.6870/3.0805|
|loveda/D|building|47.1770|38.0724|37.6948|-9.1046|38.3797/97.9408|11.8833/14.9042|
|loveda/D|road|38.2798|42.2258|43.1433|3.9460|49.4303/74.3402|6.2907/5.1666|
|loveda/D|water|46.1774|66.1127|65.2161|19.9353|84.0942/75.5614|8.3355/10.0588|
|loveda/D|barren|26.3988|27.3898|31.3496|0.9910|32.2872/64.3584|2.9231/5.3021|
|loveda/D|tree|42.3356|44.6766|48.1168|2.3410|48.3389/85.5007|8.9271/9.4230|
|loveda/D|farm|44.9570|36.2344|34.1106|-8.7226|38.8421/84.3681|27.9532/52.0648|
|vaihingen/vaihingen|impervious surface|49.2086|55.4077|56.7046|6.1991|82.8259/62.5998|19.6449/21.9053|
|vaihingen/vaihingen|building|71.9253|76.8205|72.6512|4.8952|78.3675/97.4947|33.4953/33.6847|
|vaihingen/vaihingen|low vegetation|40.8351|40.1220|33.7234|-0.7131|80.1349/44.5534|10.4596/10.6054|
|vaihingen/vaihingen|tree|71.9771|69.9354|70.1885|-2.0417|80.0130/84.7390|23.7523/24.7910|
|vaihingen/vaihingen|car|11.0980|15.6238|18.5162|4.5258|15.6973/97.0917|12.6479/9.0136|
|landcoverai/landcoverai|background|83.8346|83.9869|84.5157|0.1523|94.5204/88.2854|55.6075/54.5334|
|landcoverai/landcoverai|building|26.4985|31.6239|32.4337|5.1254|32.0672/95.8109|2.0215/1.6656|
|landcoverai/landcoverai|woodland|83.0001|83.4550|82.5986|0.4549|90.6452/91.3201|31.0406/33.3540|
|landcoverai/landcoverai|water|84.9499|87.0961|88.2517|2.1462|93.9579/92.2636|6.9217/6.7586|
|landcoverai/landcoverai|road|20.7158|23.2694|26.7062|2.5536|24.3408/84.0930|4.4087/3.6884|
|flair1/flair1|building|65.0673|68.3451|67.4362|3.2778|74.0000/89.9434|19.2463/18.0037|
|flair1/flair1|pervious surface|43.4861|47.8649|53.6314|4.3788|64.8005/64.6824|6.1225/6.2061|
|flair1/flair1|impervious surface|53.1460|57.1038|55.2977|3.9578|69.5721/76.1128|15.0046/14.0538|
|flair1/flair1|bare soil|50.4564|51.6009|59.3644|1.1445|53.9572/92.1973|4.5550/4.2019|
|flair1/flair1|water|90.2846|87.2388|86.6070|-3.0458|97.1715/89.5119|7.2545/6.6193|
|flair1/flair1|coniferous|2.3843|0.0175|0.0000|-2.3668|0.0368/0.0332|0.4686/0.7178|
|flair1/flair1|deciduous|66.4061|59.9012|57.3891|-6.5049|67.7777/83.7517|12.7660/16.5506|
|flair1/flair1|brushwood|28.5708|9.5766|6.8705|-18.9942|28.2483/12.6550|9.9854/3.2284|
|flair1/flair1|vineyard|84.7180|86.4945|83.6663|1.7765|86.9867/99.3501|10.4483/11.2402|
|flair1/flair1|herbaceous vegetation|45.4019|56.4463|56.1725|11.0444|87.8176/61.2418|12.7094/17.1013|
|flair1/flair1|agricultural land|0.2526|0.0014|0.0000|-0.2512|0.0023/0.0034|0.3876/0.5242|
|flair1/flair1|plowed land|24.9653|19.2610|18.8864|-5.7043|19.9265/85.2229|1.0518/1.5527|

## Allocation and increment diagnostics

|Dataset|Block0 native/CSA mean patch mass|Block1 native/CSA mean patch mass|Block0 read/increment norm|Block1 read/increment norm|
|---|---|---|---|---|
|vdd|0.494985/0.999816|0.556437/0.999723|65.412759/16.445801|60.465176/24.171781|
|potsdam|0.576233/0.999706|0.530343/0.999740|61.671503/15.129457|60.890347/23.550383|
|udd5|0.502194/0.999779|0.549652/0.999723|64.785071/16.138193|60.684045/24.031874|
|oem|0.558279/0.999814|0.462537/0.999800|62.117373/14.192225|64.035093/23.297437|
|loveda|0.518547/0.999787|0.476779/0.999712|62.610216/15.276293|60.472446/23.245516|
|vaihingen|0.583371/0.999684|0.593238/0.999643|62.222578/14.932154|60.547396/23.215281|
|landcoverai|0.532349/0.999804|0.482277/0.999759|62.639735/15.202966|60.791881/24.096029|
|flair1|0.534282/0.999800|0.500755/0.999766|61.933632/14.850756|61.555130/23.227051|

Eleven invariant tests plus actual-checkpoint fp32/bf16 mask-free smoke. Frozen weights, unique keys, target counts and four exact historical per-image controls verified.
All arms retain full class metrics, transition tensors and diagnostics. The reused panel is development, not independent validation.
