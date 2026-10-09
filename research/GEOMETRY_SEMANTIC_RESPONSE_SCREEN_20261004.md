# Query-anchored semantic response: verified fixed96 results

UDD5 full40; seven other domains8 complete images. Same20/RS/native512/128/Hann.
Developed validation only; all96 unique keys and four exact historical per-image references verified.
Frozen primary transports attention-induced nonlinear responses while retaining query baselines.
No exhaustive external novelty claim; Parallel literature lookup was unauthenticated.

|Dataset/protocol|Geometry|Geometry_BlockPrefix|SCLIP_Two|VIPProxy_Two|Geometry_ResponseTransport|Geometry_DonorBefore|Geometry_FullDonor|Geometry_QueryResponse|Geometry_ResponseUniform|Geometry_ResponseSpatial|Geometry_SelfConditional|
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
|potsdam/potsdam|40.3528|42.1920|44.0525|42.7377|40.6843|40.9661|40.4474|40.1048|33.6696|35.8394|39.6506|
|oem/oem|39.3543|38.8317|37.3120|38.9899|39.7249|39.9834|39.9277|39.2082|39.2525|39.0613|39.1402|
|loveda/P|62.8254|67.7492|68.2854|68.5692|61.7776|62.6893|63.6750|61.6366|59.2478|59.9252|59.0053|
|loveda/D|38.4558|35.6249|36.5043|35.6670|38.8589|39.7318|40.4104|38.1129|37.6287|38.6797|36.9964|
|landcoverai/landcoverai|60.9049|60.0368|61.7170|59.3073|61.1442|61.9362|62.1978|60.5754|62.8305|62.2942|59.9867|
|vaihingen/vaihingen|50.2325|49.8551|52.0872|50.5393|50.5907|51.2780|51.4105|49.8165|46.2294|46.4374|49.7447|
|flair1/flair1|38.8428|39.3487|37.6225|39.3436|38.8259|39.4129|39.2582|38.8956|34.6941|36.1819|37.1555|
|vdd/vdd|31.9462|31.1897|30.6800|31.1056|32.0320|32.0909|32.1514|31.8462|31.9905|32.2560|31.1266|
|udd5/udd5|50.5553|49.6843|50.2837|49.9071|50.4936|50.9302|50.5617|50.0594|47.2058|48.4550|49.7221|
|Eight-domain mean, LoveDA D once|43.8306|43.3454|43.7824|43.4497|44.0443|44.5412|44.5456|43.5774|41.6876|42.4006|42.9403|

Frozen advancement gate passed: `False`. No automatic full rollout.

## Decision

The primary mean improves43.830569 ->44.044322 (+0.213753pp), with six/eight
domain wins and small losses in UDD5/FLAIR-1. VDD improves0.085865pp and Potsdam
0.331441pp. However LoveDA P loses1.047802pp, and primary is worse than the
same-attention DonorBefore control in EVERY domain. Its preregistered advancement
gate fails. Do not promote the primary or claim nonlinear-response transport
improves the proposed architecture. Original/best models remain retained.

DonorBefore averages44.541189 (+0.710620pp), higher than all four historical
controls on this panel and above original Geometry in all eight primary domains.
It is a mechanism finding, not a retroactively renamed primary, full-set win,
untouched validation result or established CVPR contribution. FullDonor is
44.545635, just0.004446pp above DonorBefore in mean, with mixed domain effects.
Their near tie is not evidence that donor-baseline smoothing is necessary.

|Domain|Primary minus Geometry|DonorBefore minus Geometry|Primary minus DonorBefore|
|---|---:|---:|---:|
|VDD|+0.0859|+0.1447|-0.0588|
|Potsdam|+0.3314|+0.6132|-0.2818|
|UDD5|-0.0617|+0.3749|-0.4367|
|OEM|+0.3706|+0.6290|-0.2584|
|LoveDA D|+0.4030|+1.2760|-0.8729|
|Vaihingen|+0.3583|+1.0456|-0.6873|
|LandCover.ai|+0.2394|+1.0313|-0.7919|
|FLAIR-1|-0.0169|+0.5702|-0.5870|

LoveDA P: primary -1.0478pp; DonorBefore -0.1361pp. This foreground protocol
must not be hidden by reporting only the D-once mean.

## Mechanism Evidence

The nonlinear response is active, not a numerical no-op. Primary block1 induced
response norms average149.75-166.88 across domains, and its order-gap norms are
46.62-62.00. Units are frozen hidden-state Euclidean norms, not probabilities,
semantic correctness or direct additive explanations of mIoU. At the same
primary input state, the gap measures transport-after versus transform-after
responses; whole two-block primary/control comparisons include state evolution.

Geometry remains useful structure in the primary: uniform relation mean41.687638
and Gaussian-only42.400617 are below44.044322. This is not uniform superiority:
spatial wins on VDD and uniform wins on LandCover.ai. These relation controls do
not rescue the claim that the newly transported nonlinear response is helpful.
QueryResponse retains original query-owned Geometry attention yet averages
43.577372 (-0.253197pp). The favorable primary-vs-original mean cannot be
attributed solely to nonlinear response ordering.

DonorBefore has a more specific interpretable allocation change. For patch
query i, original Geometry reads special_i + m_i*sum_j G_ij V_j. DonorBefore
transports the projected self-conditional donor read, equivalent in exact
arithmetic with normalized G and affine projection to:

`sum_j G_ij [special_j + m_j V_j]`.

Before projection, its effective attention is a composition `G * D_self`,
where D_self retains native special-key coefficients and puts native patch
mass on the donor's own patch. Explicit coefficients are:

`A_new[i,special] = sum_j G_ij A_native[j,special]`,
`A_new[i,j_patch] = G_ij m_j`, and `m_new[i] = sum_j G_ij m_j`.

This preserves normalized total row mass in exact arithmetic, while changing
both query patch budget and special-content source. It is not the earlier
failed KL donor-column-marginal projection, which held original query budgets
and special contributions fixed. Finite-precision G sums/projection-bias
commutation are approximate; do not claim this implementation is bit-identical
to every possible explicit composed-attention implementation.

No inference follows that m_j is class reliability. Budget transport and
special-content transport are bundled here and need a prospectively frozen
factorial to establish which supplies the benefit. Kernel composition itself
has broad precedents; public novelty must be verified rather than inferred
from the new method name or this algebra.

DonorBefore improves vehicle/car precision on all three inspected domains:
VDD5.6966% ->5.9273%, Potsdam11.6716% ->12.3653%, UDD510.0323% ->11.0756%.
Predicted area falls4.1453% ->3.9855%,19.8762% ->18.7637%,7.7994% ->7.0644%
respectively, while recall is effectively retained. But VDD water recall falls
34.5198% ->32.9246%, Potsdam low-vegetation38.2826% ->37.8122%, and UDD5 road IoU
45.7856 ->45.6276. It is not a solution to all class deficits.

## Next Research Target

Reject the nonlinear-response primary. The positive control motivates a NEW,
prospectively specified whole-row Geometry composition study, separating donor
patch-budget movement from donor special-content movement, with valid row-mass
controls and broader locked coverage. Do not simply rename this control as the
finished module or infer full-set superiority from fixed96. Keep the best
coupled reference unchanged and test independent cost of any future candidate.

## Verification

Eleven synthetic tests passed locally and on A800 before labels. Mask-free
checkpoint smoke exactly replayed four historical operators and singleton
Geometry/primary, verified finite features and unchanged frozen weights.
Identity-relation versus self-reading feature max errors: fp32 2.2724e-7,
bf16 8.6741e-4; algebraic identity is not a floating-point bit-equality claim.
The first smoke exposed assignment to a frozen cache dataclass; only the smoke
was corrected to use an independent dataclass replacement, before label
evaluation, and the failed log is preserved. No model rule was changed.
Collection independently verified96 unique keys, all per-image confusion sums,
four exact historical per-image references, checkpoint/vocabulary identities
and unchanged original Geometry configuration. All workers/controllers finished.

## Independent Cost

|Method|512-window median ms|Peak allocated MiB|
|---|---:|---:|
|Geometry|22.312|3589.010|
|Geometry_ResponseTransport|23.232|3552.677|

Two warmups/five synchronized repeats, one backbone and one selected head; trace/control branches disabled.

## Class Outcomes

|Dataset/protocol|Class|Geometry IoU|Response IoU|Delta|Geometry precision/recall|Response precision/recall|Geometry/Response area|
|---|---|---:|---:|---:|---|---|---|
|vdd/vdd|other|25.3949|26.2971|0.9022|38.9278/42.2128|39.1895/44.4249|29.1864/30.5108|
|vdd/vdd|wall|11.8982|11.8656|-0.0326|11.9838/94.3324|11.9552/94.0635|15.3444/15.3373|
|vdd/vdd|road|24.1970|26.5464|2.3494|24.7412/91.6676|27.3003/90.5774|5.1104/4.5763|
|vdd/vdd|vegetation|62.0257|61.8328|-0.1929|84.6841/69.8628|84.3681/69.8333|17.8191/17.8783|
|vdd/vdd|vehicle|5.6965|5.4662|-0.2303|5.6966/99.9563|5.4663/99.9956|4.1453/4.3217|
|vdd/vdd|roof|60.7577|59.1703|-1.5874|88.0153/66.2377|88.7164/63.9857|20.9399/20.0681|
|vdd/vdd|water|33.6534|33.0457|-0.6077|93.0590/34.5198|93.1433/33.8697|7.4546/7.3075|
|potsdam/potsdam|impervious surface|51.6875|52.6350|0.9475|75.3909/62.1781|74.4844/64.2131|29.8644/31.2171|
|potsdam/potsdam|building|78.2026|79.0209|0.8183|80.6769/96.2262|81.7589/95.9344|14.5546/14.3184|
|potsdam/potsdam|low vegetation|33.4221|33.9357|0.5136|72.4703/38.2826|72.9519/38.8201|10.1329/10.2073|
|potsdam/potsdam|tree|62.5813|61.7293|-0.8520|90.6538/66.8975|91.4706/65.4995|17.6734/17.1496|
|potsdam/potsdam|car|11.6582|12.2420|0.5838|11.6716/99.0262|12.2559/99.0801|19.8762/18.9389|
|potsdam/potsdam|clutter|4.5652|4.5428|-0.0224|7.7447/10.0073|7.5971/10.1523|7.8986/8.1687|
|udd5/udd5|vegetation|82.4937|82.3842|-0.1095|96.4862/85.0488|96.6248/84.8252|26.1079/26.0019|
|udd5/udd5|building|83.9035|83.6183|-0.2852|88.3604/94.3293|88.7784/93.5007|41.9106/41.3468|
|udd5/udd5|road|45.7856|45.0677|-0.7179|70.6780/56.5218|70.9243/55.2812|10.7242/10.4524|
|udd5/udd5|vehicle|10.0092|10.4057|0.3965|10.0323/97.7477|10.4304/97.7716|7.7994/7.5036|
|udd5/udd5|other|30.5846|30.9921|0.4075|52.8537/42.0591|50.8877/44.2181|13.4578/14.6953|
|oem/oem|bareland|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|10.4255/9.9320|
|oem/oem|rangeland|47.7780|47.8748|0.0968|65.2945/64.0414|64.6904/64.8106|18.7786/19.1816|
|oem/oem|developed space|28.5294|31.0051|2.4757|61.5617/34.7130|61.3533/38.5301|12.6303/14.0668|
|oem/oem|road|40.8291|41.4085|0.5794|45.7530/79.1399|46.7218/78.4539|7.8925/7.6618|
|oem/oem|tree|56.0657|56.5340|0.4683|87.4475/60.9727|87.9087/61.3007|17.7308/17.7327|
|oem/oem|water|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|0.0647/0.0629|
|oem/oem|agriculture land|79.1353|77.4883|-1.6470|90.6413/86.1766|90.9477/83.9641|15.9338/15.4724|
|oem/oem|building|62.4971|63.4888|0.9917|65.5844/92.9954|67.3481/91.7213|16.5437/15.8897|
|loveda/P|building|64.4987|64.4027|-0.0960|65.3671/97.9819|65.3190/97.8683|0.9799/0.9795|
|loveda/P|road|66.9348|66.4521|-0.4827|67.6522/98.4404|67.1879/98.3787|10.3611/10.4261|
|loveda/P|water|81.1851|81.0046|-0.1805|86.7638/92.6613|87.0073/92.1514|22.3129/22.1280|
|loveda/P|barren|24.3403|24.1067|-0.2336|63.8735/28.2261|60.2305/28.6702|3.2301/3.4793|
|loveda/P|tree|53.7307|50.5828|-3.1479|64.5184/76.2666|59.3708/77.3618|13.0831/14.4216|
|loveda/P|farm|86.2626|84.1166|-2.1460|95.3302/90.0685|95.5034/87.5855|50.0330/48.5654|
|loveda/D|background|34.1983|36.9045|2.7062|65.7504/41.6109|64.1120/46.5132|28.2483/32.3833|
|loveda/D|building|31.1862|32.7025|1.5163|31.8582/93.6650|33.6699/91.9242|1.0641/0.9882|
|loveda/D|road|45.4818|46.2968|0.8150|45.8348/98.3352|46.6773/98.2695|8.4577/8.2995|
|loveda/D|water|61.9193|62.4069|0.4876|65.9551/91.0065|66.9948/90.1117|15.9605/15.5583|
|loveda/D|barren|14.5296|13.3928|-1.1368|59.0631/16.1567|61.4919/14.6188|1.1070/0.9621|
|loveda/D|tree|26.8904|27.1974|0.3070|30.0530/71.8726|30.3920/72.1248|14.6543/14.5416|
|loveda/D|farm|54.9851|53.1112|-1.8739|69.5721/72.3948|71.9859/66.9487|30.5080/27.2670|
|vaihingen/vaihingen|impervious surface|49.2002|51.0343|1.8341|82.3971/54.9789|79.8850/58.5594|20.4283/22.4429|
|vaihingen/vaihingen|building|74.6610|75.1768|0.5158|75.5001/98.5334|76.2793/98.1136|28.1288/27.7228|
|vaihingen/vaihingen|low vegetation|46.7115|46.0110|-0.7005|92.0732/48.6687|91.0096/48.2019|13.4530/13.4797|
|vaihingen/vaihingen|tree|71.8329|71.4496|-0.3833|82.5509/84.6923|83.7006/82.9977|21.4401/20.7225|
|vaihingen/vaihingen|car|8.7566|9.2820|0.5254|8.7725/97.9748|9.2987/98.0928|16.5499/15.6321|
|landcoverai/landcoverai|background|82.6090|82.9131|0.3041|96.6505/85.0437|95.9400/85.9280|59.6226/60.6888|
|landcoverai/landcoverai|building|34.9562|35.1416|0.1854|35.0436/99.2919|35.2299/99.2919|4.2927/4.2700|
|landcoverai/landcoverai|woodland|78.3638|77.5285|-0.8353|86.4992/89.2841|87.0243/87.6621|22.0960/21.5637|
|landcoverai/landcoverai|water|93.6635|93.6443|-0.0192|93.6635/100.0000|93.6443/100.0000|8.8309/8.8327|
|landcoverai/landcoverai|road|14.9318|16.4938|1.5620|15.6288/77.0019|17.3495/76.9791|5.1578/4.6448|
|flair1/flair1|building|49.6979|49.9849|0.2870|50.7258/96.0825|51.0963/95.8300|13.5991/13.4650|
|flair1/flair1|pervious surface|57.0621|56.8569|-0.2052|92.1299/59.9861|94.5633/58.7782|11.1728/10.6661|
|flair1/flair1|impervious surface|52.6509|52.0981|-0.5528|62.6245/76.7765|61.4019/77.4690|20.0846/20.6692|
|flair1/flair1|bare soil|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|3.4411/3.5073|
|flair1/flair1|water|73.2328|73.7709|0.5381|77.1432/93.5263|77.2496/94.2468|5.3781/5.4121|
|flair1/flair1|coniferous|43.0233|43.4668|0.4435|61.7114/58.6897|64.3908/57.2217|0.5346/0.4995|
|flair1/flair1|deciduous|54.0229|49.4396|-4.5833|78.9723/63.0995|80.7977/56.0220|13.6004/11.8022|
|flair1/flair1|brushwood|18.6136|19.5905|0.9769|23.4212/47.5559|23.4314/54.4441|9.9866/11.4281|
|flair1/flair1|vineyard|NA|NA|NA|0.0000/0.0000|0.0000/0.0000|0.0000/0.0000|
|flair1/flair1|herbaceous vegetation|60.2329|61.2577|1.0248|94.3169/62.5013|93.9314/63.7820|21.2289/21.7527|
|flair1/flair1|agricultural land|18.7339|20.6197|1.8858|21.6468/58.1977|24.2046/58.1977|0.8198/0.7332|
|flair1/flair1|plowed land|0.0000|0.0000|0.0000|0.0000/0.0000|0.0000/0.0000|0.1540/0.0645|

## Nonlinear Activity And Changes

|Dataset/protocol|Block0 response norm|Block1 response norm|Block0 order gap|Block1 order gap|Beneficial|Harmful|
|---|---:|---:|---:|---:|---:|---:|
|vdd/vdd|8.221810|156.426859|4.900108|46.621458|1644326|1825223|
|potsdam/potsdam|6.146531|149.966479|2.982697|61.265462|181467|143091|
|udd5/udd5|8.249473|155.447291|5.037919|51.129390|4881799|5729074|
|oem/oem|7.227797|166.883900|4.069609|49.518185|137627|96561|
|loveda/P|7.553519|149.751862|5.124082|58.240128|22127|80401|
|loveda/D|7.553519|149.751862|5.124082|58.240128|253438|217689|
|vaihingen/vaihingen|6.782971|166.648635|3.441002|62.001203|146624|104144|
|landcoverai/landcoverai|7.542819|164.156267|4.527850|47.700674|22278|16998|
|flair1/flair1|6.945907|155.164352|3.938464|56.078665|34561|45963|

All arms retain full per-class metrics, transitions and nonlinear response diagnostics in merged.json.
Fixed96 ranks can reverse on full data. Original/best models, old results and paused automations unchanged.
