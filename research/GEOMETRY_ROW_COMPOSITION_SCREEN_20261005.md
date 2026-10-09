# Native-row Geometry composition: verified broader development results

UDD5 full40 developed images; seven other domains32 each excluding prior exact8 keys;264 total.
Same fixed20 aliases/six RS templates/native512/overlap128(step384)/Hann. No model or label-fitted changes.
Initial protocol.md mislabeled overlap as stride; evaluator/settings never changed. See the view clarification.
Parent geography may overlap; full references were inspected previously. NOT untouched validation.

## Decision and mechanism

Retain original/best production models; do not promote this candidate or any
control to a full rollout. All264 unique images and four historical per-image
controls were independently verified on A800 and again after collection.
All14 evaluation workers and controller are terminal, and GPUs0-7 are idle.

RowCompose improves all eight primary domain entries versus Geometry, averaging
46.216453 ->46.771300 (+0.554846pp). Conditional paired source-group bootstrap
95% interval for that mean delta is [+0.348976,+0.733358]pp. This supports a
development-panel improvement, not untouched generalization. Numerical FPRead
changes the original mean only+0.000199pp; the main gain is not that precision
change. Prior DonorBefore and explicit RowCompose differ by just+0.000098pp in
mean, consistent with their related algebra rather than a new semantic-response
mechanism. Do not treat them as independent contributions.

The frozen gate fails. SCLIP_Two mean47.156590 and GlobalBudget46.954070 exceed
the primary, while VIPProxy_Two46.773533 is effectively tied (primary-Proxy
-0.002233pp; conditional95% interval [-0.763422,+0.831002]pp). LoveDA P falls
1.002347pp, narrowly beyond the frozen1pp loss limit; do not relax that limit
after results. Primary-minus-GlobalBudget is-0.182770pp with interval
[-0.471278,+0.064151]pp, so point ranking is not proof of a population advantage
for the simpler control either. Neither establishes nearest-operator superiority.

Mean contrasts versus FPRead: BudgetOnly+0.445413pp, SpecialOnly+0.186188pp,
GroupCompose+0.566014pp, ConditionalOnly-0.084642pp. Full-minus-GroupCompose
is-0.011367pp. Allocation movement explains more of the observed gain than
special-content movement, but nonlinear interactions prevent an additive causal
decomposition. Reweighting G by native donor mass alone does not help overall;
the crop/head constant-budget control is stronger in point mean and improves
LoveDA P+0.492632pp. These observations do NOT justify treating native mass as
semantic donor reliability or whole-row composition as necessary sophistication.

Class tradeoffs remain decisive:

- UDD5 vehicle IoU improves10.0092 ->11.0478 with precision10.0323 ->11.0761,
  lower predicted area7.7994 ->7.0640 and almost unchanged recall. Road loses
  0.1573pp and building loses0.3539pp.
- Potsdam car IoU improves0.7476pp and low-vegetation0.8244pp, but is still far
  below the matched SCLIP domain result. Car recall already approaches100%:
  this is mainly reducing overactivation, not demonstrating small-object recovery.
- VDD vehicle gains only0.0761pp; water loses1.7205pp and roof1.9645pp.
- LoveDA D gains0.7018pp largely through background IoU+5.6562pp; D foreground
  mean falls40.8876 ->40.7637. Water loses3.7425pp. LoveDA P tree loses3.3265pp
  and farm1.6999pp. Do not frame the D gain as uniform foreground enhancement.
- FLAIR-1 mIoU rises0.1686pp despite more harmful than beneficial pixel changes;
  vineyard loses3.5772pp. Macro-class and pixel-accuracy objectives differ.

Next requirement: stop attributing gains to nonlinear-response ordering or
native-mass reliability. A new candidate must outperform constant-budget and
nearest-operator controls under a separately frozen protocol, preserve useful
special/background competition, and address semantic errors rather than merely
smooth allocation. The present evidence identifies a limitation but does not
validate another design, so no untested final module or further run is declared.
The publication goal remains active; no SOTA or priority claim is established.

Companion files: `GEOMETRY_ROW_COMPOSITION_METHOD_NOTE_20261005.md`,
`geometry_row_composition_screen_20261005/PAIRED_UNCERTAINTY.md` and
`geometry_row_composition_screen_20261005/paired_statistics.json`.

## Complete results

|Dataset/protocol|Geometry|Geometry_BlockPrefix|SCLIP_Two|VIPProxy_Two|Geometry_FPRead|Geometry_RowCompose|Geometry_BudgetOnly|Geometry_SpecialOnly|Geometry_GroupCompose|Geometry_ConditionalOnly|Geometry_GlobalBudget|Geometry_DonorBefore|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|vdd/vdd|40.6839|41.7909|42.2973|41.8286|40.6853|40.7786|40.8350|40.8405|40.8524|40.5094|40.9161|40.7782|
|potsdam/potsdam|40.5893|42.7069|45.7020|42.9724|40.5914|41.4396|41.3673|40.6797|41.3603|40.5773|41.3905|41.4371|
|udd5/udd5|50.5553|49.6843|50.2837|49.9071|50.5567|50.9303|51.0197|50.6407|51.0739|50.2880|50.7753|50.9302|
|oem/oem|42.9602|43.6819|42.9439|43.8081|42.9601|43.2382|43.2438|42.9480|43.1800|43.0503|43.3898|43.2366|
|loveda/P|65.0902|68.4444|65.6949|68.6542|65.0904|64.0878|64.9784|65.0050|64.4265|64.6114|65.5828|64.0840|
|loveda/D|39.8728|37.0786|37.3274|36.9445|39.8755|40.5746|40.0770|40.0343|40.4091|39.9534|40.2932|40.5712|
|vaihingen/vaihingen|49.0088|49.4759|50.3555|50.2703|49.0096|49.9651|49.5442|49.4220|49.9729|48.9218|50.0436|49.9656|
|landcoverai/landcoverai|59.7998|62.1314|62.9003|61.5546|59.7974|60.8139|60.6321|60.0090|60.6947|59.8487|61.6376|60.8183|
|flair1/flair1|46.2616|46.7937|45.4426|46.9027|46.2572|46.4302|46.5774|46.6485|46.7180|45.9074|47.1865|46.4324|
|Eight-domain mean, LoveDA D once|46.2165|46.6679|47.1566|46.7735|46.2167|46.7713|46.6621|46.4028|46.7827|46.1320|46.9541|46.7712|

Frozen advancement gate: `False`. No automatic full rollout or control-to-primary promotion.

## Frozen gate

```json
{
  "mean_vs_Geometry": true,
  "mean_vs_Geometry_BlockPrefix": true,
  "mean_vs_SCLIP_Two": false,
  "mean_vs_VIPProxy_Two": false,
  "mean_vs_Geometry_FPRead": true,
  "mean_vs_Geometry_GlobalBudget": false,
  "retain_udd5_udd5": true,
  "retain_vdd_vdd": true,
  "focus_vdd": true,
  "retain_landcoverai_landcoverai": true,
  "retain_flair1_flair1": true,
  "retain_potsdam_potsdam": true,
  "focus_potsdam": true,
  "retain_oem_oem": true,
  "retain_vaihingen_vaihingen": true,
  "retain_loveda_P": false,
  "retain_loveda_D": true,
  "latency_ratio_at_most_1_10": true
}
```

## Allocation/content contrasts

These are nonlinear whole-system mIoU contrasts, NOT additive pixel causal effects or a complete three-factor factorial.

|Dataset/protocol|Primary-Geometry|Primary-FPRead|Budget-FPRead|Special-FPRead|Group-Budget-Special+FPRead|Full-Group|Conditional-FPRead|
|---|---:|---:|---:|---:|---:|---:|---:|
|vdd/vdd|+0.0947|+0.0933|+0.1497|+0.1552|-0.1378|-0.0738|-0.1759|
|potsdam/potsdam|+0.8503|+0.8482|+0.7759|+0.0883|-0.0953|+0.0793|-0.0141|
|udd5/udd5|+0.3750|+0.3736|+0.4630|+0.0841|-0.0299|-0.1436|-0.2687|
|oem/oem|+0.2780|+0.2781|+0.2837|-0.0120|-0.0518|+0.0582|+0.0902|
|loveda/P|-1.0023|-1.0026|-0.1120|-0.0854|-0.4665|-0.3387|-0.4790|
|loveda/D|+0.7018|+0.6990|+0.2015|+0.1588|+0.1733|+0.1655|+0.0778|
|vaihingen/vaihingen|+0.9563|+0.9555|+0.5346|+0.4124|+0.0163|-0.0078|-0.0879|
|landcoverai/landcoverai|+1.0141|+1.0164|+0.8347|+0.2116|-0.1490|+0.1191|+0.0512|
|flair1/flair1|+0.1686|+0.1730|+0.3202|+0.3912|-0.2507|-0.2878|-0.3498|

## Fixed target and foreground metrics

Target-present mIoU scores only classes with positive GT support (identical target counts verified for all arms). Standard union-scored mIoU remains the primary table above.

|Dataset/protocol|Metric|Geometry|FPRead|RowCompose|
|---|---|---:|---:|---:|
|vdd/vdd|Target-present mIoU|40.6839|40.6853|40.7786|
|potsdam/potsdam|Target-present mIoU|40.5893|40.5914|41.4396|
|udd5/udd5|Target-present mIoU|50.5553|50.5567|50.9303|
|udd5/udd5|non_residual_mean_iou_percent|55.5480|55.5492|55.8473|
|oem/oem|Target-present mIoU|42.9602|42.9601|43.2382|
|loveda/P|Target-present mIoU|65.0902|65.0904|64.0878|
|loveda/D|Target-present mIoU|39.8728|39.8755|40.5746|
|loveda/D|foreground_mean_iou_percent|40.8876|40.8904|40.7637|
|vaihingen/vaihingen|Target-present mIoU|49.0088|49.0096|49.9651|
|landcoverai/landcoverai|Target-present mIoU|59.7998|59.7974|60.8139|
|landcoverai/landcoverai|foreground_mean_iou_percent|53.7911|53.7891|54.9605|
|landcoverai/landcoverai|non_residual_mean_iou_percent|53.7911|53.7891|54.9605|
|flair1/flair1|Target-present mIoU|46.2616|46.2572|46.4302|

## Independent cost

|Method|512-window median ms|Peak allocated MiB|
|---|---:|---:|
|Geometry|22.061|3589.010|
|Geometry_FPRead|23.391|3591.119|
|Geometry_RowCompose|23.376|3591.119|

Five warmups/15 synchronized512-window repeats; one backbone/one selected head, diagnostics/control branches disabled.

## Per-class outcomes

|Dataset/protocol|Class|Geometry IoU|RowCompose IoU|Delta|Geometry precision/recall|RowCompose precision/recall|Geometry/RowCompose predicted area|
|---|---|---:|---:|---:|---|---|---|
|vdd/vdd|other|11.0738|12.3555|1.2817|18.3817/21.7860|19.7643/24.7898|17.3211/18.3306|
|vdd/vdd|wall|22.3327|23.1046|0.7719|23.1704/86.0666|23.9989/86.1123|7.9351/7.6652|
|vdd/vdd|road|46.1955|47.7110|1.5155|48.1764/91.8266|49.7413/92.1190|9.5603/9.2890|
|vdd/vdd|vegetation|72.6499|73.3527|0.7028|95.5844/75.1728|95.5350/75.9567|38.3752/38.7955|
|vdd/vdd|vehicle|10.1560|10.2321|0.0761|10.1657/99.0756|10.2405/99.2038|4.6223/4.5944|
|vdd/vdd|roof|65.9570|63.9925|-1.9645|85.0657/74.5948|85.3514/71.8878|13.3154/12.7893|
|vdd/vdd|water|56.4221|54.7016|-1.7205|92.1013/59.2911|92.4385/57.2639|8.8706/8.5361|
|potsdam/potsdam|impervious surface|53.8649|55.8538|1.9889|76.7376/64.3767|76.3783/67.5167|29.0511/30.6114|
|potsdam/potsdam|building|82.2362|83.0247|0.7885|84.2564/97.1669|85.4873/96.6468|24.5007/24.0187|
|potsdam/potsdam|low vegetation|37.5641|38.3885|0.8244|93.7777/38.5243|93.3628/39.4655|7.5714/7.7908|
|potsdam/potsdam|tree|53.6529|54.4589|0.8060|83.8470/59.8379|84.5029/60.5013|13.3723/13.4157|
|potsdam/potsdam|car|9.9333|10.6809|0.7476|9.9377/99.5585|10.6856/99.5942|20.3213/18.9058|
|potsdam/potsdam|clutter|6.2844|6.2309|-0.0535|11.5353/12.1310|11.3638/12.1224|5.1832/5.2576|
|udd5/udd5|vegetation|82.4937|83.1635|0.6698|96.4862/85.0488|96.5397/85.7187|26.1079/26.2990|
|udd5/udd5|building|83.9035|83.5496|-0.3539|88.3604/94.3293|89.0643/93.1004|41.9106/41.0376|
|udd5/udd5|road|45.7856|45.6283|-0.1573|70.6780/56.5218|70.5959/56.3345|10.7242/10.7011|
|udd5/udd5|vehicle|10.0092|11.0478|1.0386|10.0323/97.7477|11.0761/97.7425|7.7994/7.0640|
|udd5/udd5|other|30.5846|31.2620|0.6774|52.8537/42.0591|50.8519/44.7974|13.4578/14.8983|
|oem/oem|bareland|9.5228|9.9565|0.4337|9.7982/77.2103|10.2541/77.4292|9.0985/8.7185|
|oem/oem|rangeland|25.2283|25.4261|0.1978|48.9007/34.2601|48.1896/34.9916|13.2907/13.7747|
|oem/oem|developed space|29.0591|30.4899|1.4308|46.2861/43.8445|45.5681/47.9556|19.6194/21.7971|
|oem/oem|road|37.3322|37.6396|0.3074|48.2593/62.2468|49.3043/61.4040|9.3608/9.0384|
|oem/oem|tree|66.5893|66.5573|-0.0320|85.4774/75.0839|85.3322/75.1555|16.9292/16.9741|
|oem/oem|water|63.0704|63.2922|0.2218|68.0649/89.5781|68.3504/89.5316|1.7993/1.7908|
|oem/oem|agriculture land|52.7895|51.1889|-1.6006|91.2638/55.5990|91.7709/53.6516|10.0235/9.6189|
|oem/oem|building|60.0899|61.3551|1.2652|65.5050/87.9064|68.8256/84.9685|19.8788/18.2874|
|loveda/P|building|70.5532|70.9969|0.4437|71.3984/98.3498|71.8869/98.2861|15.3451/15.2309|
|loveda/P|road|49.0969|48.4518|-0.6451|56.2880/79.3518|55.5711/79.0882|9.2374/9.3255|
|loveda/P|water|80.6240|80.3735|-0.2505|98.6576/81.5183|98.7049/81.2301|17.6431/17.5723|
|loveda/P|barren|48.1737|47.6379|-0.5358|60.2037/70.6816|58.7514/71.5778|5.9565/6.1812|
|loveda/P|tree|63.8268|60.5003|-3.3265|69.4480/88.7458|65.0537/89.6305|12.9850/14.0003|
|loveda/P|farm|78.2664|76.5665|-1.6999|95.5949/81.1948|95.9674/79.1118|38.8329/37.6898|
|loveda/D|background|33.7840|39.4402|5.6562|60.9139/43.1347|60.6980/52.9666|33.6870/41.5126|
|loveda/D|building|47.1770|49.0588|1.8818|47.8089/97.2747|50.0086/96.2729|11.8833/11.2436|
|loveda/D|road|38.2798|39.5365|1.2567|42.8004/78.3750|44.5387/77.8775|6.2907/6.0068|
|loveda/D|water|46.1774|42.4349|-3.7425|74.0157/55.1117|74.5692/49.6152|8.3355/7.4485|
|loveda/D|barren|26.3988|26.1440|-0.2548|39.8903/43.8371|42.4649/40.4845|2.9231/2.5359|
|loveda/D|tree|42.3356|43.9543|1.6187|47.4935/79.5844|50.2664/77.7793|8.9271/8.2433|
|loveda/D|farm|44.9570|43.4533|-1.5037|57.6087/67.1817|61.8466/59.3677|27.9532/23.0093|
|vaihingen/vaihingen|impervious surface|49.2086|51.0415|1.8329|81.6361/55.3337|79.1983/58.9436|19.6449/21.5706|
|vaihingen/vaihingen|building|71.9253|72.7675|0.8422|75.6531/93.5885|77.1955/92.6932|33.4953/32.5120|
|vaihingen/vaihingen|low vegetation|40.8351|41.6598|0.8247|81.8734/44.8938|80.9252/46.1960|10.4596/10.8891|
|vaihingen/vaihingen|tree|71.9771|72.0287|0.0516|83.0994/84.3204|83.3310/84.1537|23.7523/23.6395|
|vaihingen/vaihingen|car|11.0980|12.3282|1.2302|11.1403/96.6894|12.3795/96.7484|12.6479/11.3889|
|landcoverai/landcoverai|background|83.8346|84.2273|0.3927|93.4841/89.0373|92.8821/90.0390|55.6075/56.5976|
|landcoverai/landcoverai|building|26.4985|28.9157|2.4172|26.7245/96.9079|29.2523/96.1723|2.0215/1.8328|
|landcoverai/landcoverai|woodland|83.0001|82.7794|-0.2207|93.7305/87.8789|94.0682/87.3384|31.0406/30.7389|
|landcoverai/landcoverai|water|84.9499|85.4553|0.5054|91.6037/92.1230|91.9171/92.3987|6.9217/6.9188|
|landcoverai/landcoverai|road|20.7158|22.6917|1.9759|21.3165/88.0256|23.5423/86.2637|4.4087/3.9119|
|flair1/flair1|building|65.0673|66.0794|1.0121|69.7561/90.6369|71.3814/89.8952|19.2463/18.6541|
|flair1/flair1|pervious surface|43.4861|43.0846|-0.4015|61.0831/60.1514|62.4239/58.1713|6.1225/5.7938|
|flair1/flair1|impervious surface|53.1460|53.1182|-0.0278|64.4136/75.2366|63.5724/76.3600|15.0046/15.4301|
|flair1/flair1|bare soil|50.4564|52.0252|1.5688|51.6401/95.6543|53.1129/96.2129|4.5550/4.4546|
|flair1/flair1|water|90.2846|91.0033|0.7187|94.4443/95.3485|94.1264/96.4822|7.2545/7.3656|
|flair1/flair1|coniferous|2.3843|2.8644|0.4801|6.2860/3.6993|7.4024/4.4639|0.4686/0.4801|
|flair1/flair1|deciduous|66.4061|66.2594|-0.1467|81.7746/77.9415|82.9851/76.6763|12.7660/12.3756|
|flair1/flair1|brushwood|28.5708|28.4231|-0.1477|38.2594/53.0128|36.1700/57.0273|9.9854/11.3620|
|flair1/flair1|vineyard|84.7180|81.1408|-3.5772|89.0633/94.5546|90.4887/88.7063|10.4483/9.6476|
|flair1/flair1|herbaceous vegetation|45.4019|46.1462|0.7443|91.4731/47.4084|91.4158/48.2364|12.7094/12.9395|
|flair1/flair1|agricultural land|0.2526|0.0802|-0.1724|0.4833/0.5264|0.1285/0.2126|0.3876/0.5885|
|flair1/flair1|plowed land|24.9653|26.9374|1.9721|26.8733/77.8575|29.7032/74.3121|1.0518/0.9083|

## Allocation diagnostics and transitions

|Dataset/protocol|Block0 mass change|Block1 mass change|Block0 row-mass error|Block1 row-mass error|Beneficial|Harmful|
|---|---:|---:|---:|---:|---:|---:|
|vdd/vdd|0.09338804|0.06952682|0.00000035|0.00000035|6042465|5476411|
|potsdam/potsdam|0.09175776|0.08588753|0.00000036|0.00000036|751228|343250|
|udd5/udd5|0.09448812|0.07576680|0.00000035|0.00000035|5627728|4950799|
|oem/oem|0.09528895|0.08726280|0.00000036|0.00000035|555333|495603|
|loveda/P|0.08934408|0.07417377|0.00000034|0.00000034|96248|249647|
|loveda/D|0.08934408|0.07417377|0.00000034|0.00000034|1626196|1001405|
|vaihingen/vaihingen|0.09718712|0.09145611|0.00000036|0.00000036|613117|291448|
|landcoverai/landcoverai|0.09185206|0.06918313|0.00000034|0.00000034|79785|46067|
|flair1/flair1|0.09297210|0.07723990|0.00000035|0.00000035|109607|130761|

All arms' class metrics, target counts, transition tensors and diagnostics remain in merged.json.
Fifteen unit tests before labels; an additional all-Value/patch-only shift lemma test after launch. Mask-free fp32/bf16 smoke; four exact historical per-image controls; unique coverage verified.
Nineteen focused local tests pass after adding three paired-report tests; sixteen
model/panel tests also pass on A800. Reporting tests did not alter inference.
The combined local regression run, including prior semantic-response and
VIP-factor tests, passes all39 tests.
The initial A800 test deployment lacked the existing grouping helper (ModuleNotFoundError: report_matched_readout_publication). It was deployed before smoke/labels; no model rule changed or evaluation was relaunched.
Composition is related to the prior developed DonorBefore control and has broad precedents. Novelty/SOTA/CVPR acceptance are not established.
