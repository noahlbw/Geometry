# PC59/PC60 alias and scalar tuning versus CorrCLIP

Exploratory labeled development on previously viewed validation images. Freeze a development-selected choice before one full run; preserve failures and negative results.

|Protocol|Images|Incumbent|Tuned|CorrCLIP|Tuned minus CorrCLIP|Median ms|Peak MiB|
|---|---:|---:|---:|---:|---:|---:|---:|
|context59|5105|46.3075|49.6965|47.7036|+1.9929|181.90|2482.71|
|context60|5105|44.7613|45.9656|45.1164|+0.8492|187.69|2552.63|

Only words/templates and declared scalars change. Geometry/local/wide/coupled equations, residual ontology, image views and pretrained weights are frozen. Variable alias counts also change inherited wide aggregation mass. CorrCLIP uses MetaCLIP plus online SAM2; these are local full-method comparisons on the same scored inputs, not published-paper numbers.

Timing uses the same seven fixed inputs, two warmups and seven synchronized repeats; whole RGB to CPU IDs, excluding decode/init/offline search. Original incumbents remain preserved.

Development selection averages classes with ground-truth support in the development subset. The split summaries below use the standard union-valid mean, which also scores false-positive-only classes. Their mIoU values can differ even though the development confusion matrix replays exactly.

## context59

Frozen choice: `{"bank": "mix_012c5eabe1c2bb3d147f", "strength": 2.5, "coupling": 0.75, "temperature": 0.08, "tau": 1, "tem": 0.3, "wide_policy": "natural_short336_cap672", "background_rule": "retained"}`.

Development 256; complement 4849. Bias 0.0; threshold 0.0.

Development mIoU 47.9979; complement mIoU 49.7294.

|Parameter|Incumbent|Frozen trial|
|---|---|---|
|strength|3.0|2.5|
|coupling|0.75|0.75|
|temperature|0.07|0.08|
|tau|1.0|1|
|tem|10.0|0.3|
|wide_policy|natural_short336_cap672|natural_short336_cap672|
|background_rule|retained|retained|
|background bias|0.0|0.0|
|background threshold|0.0|0.0|

|Class|Incumbent IoU|Tuned IoU|Delta pp|Selected aliases|
|---|---:|---:|---:|---|
|aeroplane|67.3383|71.3092|+3.9709|airplane|
|bag|44.8271|46.0504|+1.2233|bag, backpack, handbag, carry bag, carrying bag|
|bed|6.8257|9.9227|+3.0970|bed, sleeping bed, bed furniture, room bed, double bed, single bed|
|bedclothes|26.9829|22.9970|-3.9859|bedclothes, bedding, comforter|
|bench|18.2440|28.6663|+10.4223|bench, wooden bench|
|bicycle|69.3990|65.6548|-3.7442|bicycle, bike|
|bird|70.3675|74.4027|+4.0352|bird|
|boat|68.6008|49.8377|-18.7631|boat, watercraft, small boat, sailing boat, motorboat|
|book|6.5114|40.7058|+34.1944|book, books|
|bottle|71.2428|71.6751|+0.4323|bottle, drink bottle, glass bottle, plastic bottle|
|building|36.9821|38.8597|+1.8776|building, buildings, building facade, building frontage, building structure, constructed building, building exterior, edifice|
|bus|80.2299|77.1749|-3.0550|bus, red bus, double-decker bus, passenger bus, public transit bus|
|cabinet|36.8496|42.3094|+5.4598|cabinet, cupboard, storage cabinet, cabinet furniture|
|car|64.8208|72.4439|+7.6231|car, automobile, passenger car|
|cat|87.9852|88.8356|+0.8504|cat, feline|
|ceiling|48.0722|49.6743|+1.6021|ceiling, indoor ceiling, room ceiling|
|chair|43.2269|43.9229|+0.6960|chair, dining chair, wooden chair|
|cloth|16.9566|21.9480|+4.9914|cloth, fabric, piece of fabric, rag, towel, blanket, sheet|
|computer|26.6571|38.3188|+11.6617|computer|
|cow|83.9745|85.9299|+1.9554|cow, cattle, dairy cow|
|cup|31.5851|27.0434|-4.5417|cup, mug|
|curtain|57.4631|48.9093|-8.5538|curtain|
|dog|82.2504|86.5884|+4.3380|dog|
|door|27.4833|27.2575|-0.2258|door, wooden door, entry door|
|fence|38.4962|35.6927|-2.8035|fence, wire fence, fencing, fence structure, boundary fence|
|floor|50.5707|50.3247|-0.2460|floor, floor surface, indoor floor, room floor, tiled floor|
|flower|24.5721|29.4498|+4.8777|flower, flowers|
|food|50.2279|52.8325|+2.6046|food|
|grass|71.6279|67.0247|-4.6032|grass, grassy ground, grass cover, grass surface, a grass, a grassy ground, a grass cover, a grass surface, the grass, the grassy ground, the grass cover, the grass surface, visible grass, visible grassy ground, visible grass cover, visible grass surface, grass in a scene, grassy ground in a scene, grass cover in a scene, grass surface in a scene, meadow|
|ground|12.1732|17.1737|+5.0005|ground, soil, bare ground, dirt|
|horse|83.9426|84.2893|+0.3467|horse|
|keyboard|52.1329|69.4878|+17.3549|computer keyboard, keyboard, typing keyboard|
|light|24.1319|32.8224|+8.6905|light, lamp|
|motorbike|73.6571|76.5631|+2.9060|motorcycle|
|mountain|35.5850|32.0729|-3.5121|mountain, mountain terrain, mountain peak, mountain slope, a mountain, a mountain terrain, a mountain peak, a mountain slope, the mountain, the mountain terrain, the mountain peak, the mountain slope, visible mountain, visible mountain terrain, visible mountain peak, visible mountain slope, mountain in a scene, mountain terrain in a scene, mountain peak in a scene, mountain slope in a scene|
|mouse|62.1596|65.7354|+3.5758|computer mouse|
|person|66.1820|76.1765|+9.9945|person, people, people in shirt, people in jeans, people in dress, people in sweater, people in skirt, people in jacket, human, crowd, child, boy, girl, man, women, baby|
|plate|31.8412|49.4050|+17.5638|plate, ceramic plate, serving plate, dish|
|platform|9.2000|11.0322|+1.8322|railway platform, station platform, train platform, raised platform|
|pottedplant|49.8043|48.5541|-1.2502|potted plant, plant in a pot|
|road|37.2111|43.9521|+6.7410|road, roadway, street road, paved road|
|rock|21.0625|41.1865|+20.1240|rock, natural rock, rock surface, boulder|
|sheep|85.6449|85.1257|-0.5192|sheep|
|shelves|18.7586|26.8945|+8.1359|shelf, shelves|
|sidewalk|9.4003|12.8649|+3.4646|sidewalk, footpath, pedestrian walkway, street sidewalk|
|sign|42.5840|41.4560|-1.1280|sign, signboard, traffic sign|
|sky|81.5902|81.5909|+0.0007|sky, cloudy sky, blue sky, the sky above, clear sky, skyline, atmosphere|
|snow|64.2701|66.1980|+1.9279|snow|
|sofa|65.0447|61.5432|-3.5015|sofa, couch, settee, upholstered sofa, fabric sofa|
|table|46.5805|54.8306|+8.2501|table, table furniture, tabletop furniture, desk|
|track|5.0192|2.2409|-2.7783|track|
|train|49.3242|63.2197|+13.8955|train, locomotive|
|tree|64.1148|60.4947|-3.6201|tree, trees, forest tree, woody tree, tree canopy, branching tree|
|truck|23.4607|24.4457|+0.9850|truck, lorry, cargo truck|
|tvmonitor|50.2867|55.6045|+5.3178|television monitor, monitor, computer monitor, television|
|wall|30.6628|43.4204|+12.7576|wall, brick wall, stone wall, wall surface, vertical wall, interior wall|
|water|73.9754|77.3593|+3.3839|water, body of water, lake, pond, sea, green water, blue water, water surface, open water|
|window|36.7030|39.0162|+2.3132|window, building window, glass window|
|wood|15.2672|21.5723|+6.3051|wood, wooden, lumber, timber, wooden plank|

## context60

Frozen choice: `{"bank": "mix_1ec4f4d7abb136201636", "strength": 2, "coupling": 0.875, "temperature": 0.06, "tau": 1, "tem": 0.3, "wide_policy": "natural_short336_cap672", "background_rule": "residual_max"}`.

Development 256; complement 4849. Bias -0.5; threshold 0.05.

Development mIoU 44.9407; complement mIoU 46.0351.

|Parameter|Incumbent|Frozen trial|
|---|---|---|
|strength|2.0|2|
|coupling|0.75|0.875|
|temperature|0.07|0.06|
|tau|1.0|1|
|tem|1.0|0.3|
|wide_policy|natural_short336_cap672|natural_short336_cap672|
|background_rule|residual_max|residual_max|
|background bias|-0.25|-0.5|
|background threshold|0.05|0.05|

|Class|Incumbent IoU|Tuned IoU|Delta pp|Selected aliases|
|---|---:|---:|---:|---|
|background|20.7389|22.4156|+1.6767|background|
|aeroplane|68.7764|69.6298|+0.8534|airplane|
|bag|29.4065|30.6421|+1.2356|bag, backpack, handbag, carry bag, carrying bag|
|bed|12.1528|11.0802|-1.0726|bed, sleeping bed, bed furniture, room bed, double bed, single bed|
|bedclothes|26.4340|22.4328|-4.0012|bedclothes, bedding, comforter|
|bench|19.8780|19.7326|-0.1454|bench, wooden bench, long bench|
|bicycle|67.0011|68.9189|+1.9178|bicycle|
|bird|67.0847|63.6680|-3.4167|bird|
|boat|51.2771|49.3166|-1.9605|boat, watercraft, small boat, sailing boat, motorboat|
|book|37.4128|9.9489|-27.4639|book, printed book, bound book|
|bottle|63.1870|64.6752|+1.4882|bottle, drink bottle, glass bottle, plastic bottle|
|building|33.3275|37.6203|+4.2928|building, buildings, building facade, building frontage, building structure, constructed building, building exterior, edifice|
|bus|76.4648|76.9258|+0.4610|bus, red bus, double-decker bus, passenger bus, public transit bus|
|cabinet|34.4024|38.4635|+4.0611|cabinet, cupboard, storage cabinet, cabinet furniture|
|car|68.4574|65.5128|-2.9446|car, automobile, motorcar|
|cat|86.6568|87.1846|+0.5278|cat, feline|
|ceiling|47.7303|48.4235|+0.6932|ceiling, indoor ceiling, room ceiling|
|chair|41.2580|42.1565|+0.8985|chair, dining chair, wooden chair|
|cloth|16.6705|18.8309|+2.1604|cloth, fabric, piece of fabric, textile, rag, towel, blanket|
|computer|22.5117|22.5155|+0.0038|computer, computing device|
|cow|84.5857|82.1143|-2.4714|cow, cattle, dairy cow, cattle animal, bovine, domestic cow|
|cup|24.2944|21.0092|-3.2852|cup, mug|
|curtain|55.2001|55.1163|-0.0838|curtain, curtains, window curtain|
|dog|86.1203|85.2368|-0.8835|dog|
|door|23.4358|26.8088|+3.3730|door, door panel, doorway door, entry door|
|fence|34.7057|36.9908|+2.2851|fence, wire fence|
|floor|48.8606|46.8006|-2.0600|floor, floor surface, indoor floor, room floor, flooring, tiled floor|
|flower|27.5004|25.8910|-1.6094|flower, flowers|
|food|34.6374|50.6570|+16.0196|food|
|grass|67.8552|65.5222|-2.3330|grass, grassy ground, grass cover, grass surface, a grass, a grassy ground, a grass cover, a grass surface, the grass, the grassy ground, the grass cover, the grass surface, visible grass, visible grassy ground, visible grass cover, visible grass surface, grass in a scene, grassy ground in a scene, grass cover in a scene, grass surface in a scene, meadow|
|ground|10.2002|14.0304|+3.8302|ground, soil, bare ground, dirt|
|horse|77.8086|80.1608|+2.3522|horse|
|keyboard|54.8674|54.2179|-0.6495|computer keyboard, keyboard, typing keyboard|
|light|32.2097|31.2770|-0.9327|light, lamp|
|motorbike|64.2913|65.2011|+0.9098|motorcycle|
|mountain|38.9106|33.9915|-4.9191|mountain, mountain terrain, mountain peak, mountain slope, a mountain, a mountain terrain, a mountain peak, a mountain slope, the mountain, the mountain terrain, the mountain peak, the mountain slope, visible mountain, visible mountain terrain, visible mountain peak, visible mountain slope, mountain in a scene, mountain terrain in a scene, mountain peak in a scene, mountain slope in a scene|
|mouse|55.8503|60.5017|+4.6514|computer mouse|
|person|76.0512|74.0599|-1.9913|person, people, person wearing a shirt, person wearing jeans, person wearing a dress, person wearing a sweater, person wearing a skirt, person wearing a jacket, human, crowd, child, boy, girl, man, women, baby, human being, individual person|
|plate|44.7232|50.5921|+5.8689|plate|
|platform|21.8104|15.2810|-6.5294|railway platform, station platform, train platform, raised platform|
|pottedplant|44.1617|46.3481|+2.1864|potted plant, plant in a pot|
|road|42.3314|45.4917|+3.1603|road, roadway, street road, paved road|
|rock|29.1788|31.5865|+2.4077|rock, natural rock|
|sheep|84.5030|83.0967|-1.4063|sheep, ovine animal, domestic sheep|
|shelves|23.8299|21.8249|-2.0050|shelf, shelves|
|sidewalk|15.0331|13.1988|-1.8343|sidewalk, footpath, pedestrian walkway, street sidewalk|
|sign|35.5181|35.3282|-0.1899|sign, signboard, traffic sign|
|sky|72.0540|77.7742|+5.7202|sky, cloudy sky, blue sky, the sky above, clear sky, skyline, atmosphere|
|snow|64.7348|63.5228|-1.2120|snow|
|sofa|61.0352|59.5973|-1.4379|sofa, couch, settee, upholstered sofa|
|table|47.7483|47.6745|-0.0738|table, table furniture, tabletop furniture, desk|
|track|1.9396|45.2666|+43.3270|railway track, railroad track, train track, rail tracks, railroad rails|
|train|47.2403|60.7686|+13.5283|train, railway train, locomotive, passenger train|
|tree|52.9710|57.8259|+4.8549|tree, trees, forest tree, woody tree, tree canopy, branching tree|
|truck|17.0885|21.0204|+3.9319|truck, lorry, cargo truck|
|tvmonitor|50.9884|49.6971|-1.2913|television monitor, monitor, computer monitor, television|
|wall|28.9571|37.5031|+8.5460|wall, brick wall, wall surface, vertical wall, interior wall|
|water|76.3571|76.0081|-0.3490|water, body of water, lake, pond, sea, green water, blue water, water surface, open water|
|window|29.9164|31.2298|+1.3134|window, building window|
|wood|5.3425|7.6203|+2.2778|wood|
