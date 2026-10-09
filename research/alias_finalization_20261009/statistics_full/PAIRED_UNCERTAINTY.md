# Frozen finalization: paired uncertainty

Paired source-group bootstrap; filename groups are proxies, not verified independent acquisitions. VOC20/21 and PC59/60 reuse sources. Conditional on developed domains; not adjusted for method selection, not untouched validation. All classes with union>0 participate in each bootstrap draw. Intervals do not replace the frozen model-decision gate.

2000 paired bootstrap repetitions; no image/model inference.

| Protocol | Groups | Comparison | Delta pp | 95% paired interval |
| --- | ---: | --- | ---: | --- |
| vdd/vdd | 80 | Uniform_minus_Retained | -0.1077 | [-0.1401, -0.0812] |
| vdd/vdd | 80 | Uniform_minus_UniformMean | +2.0134 | [+1.5399, +2.5295] |
| vdd/vdd | 80 | Uniform_minus_UniformLocal | +2.1447 | [+0.5879, +3.9955] |
| vdd/vdd | 80 | Uniform_minus_UniformWide | +4.5220 | [+3.6503, +5.4559] |
| vdd/vdd | 80 | Retained_minus_UniformMean | +2.1211 | [+1.6364, +2.6427] |
| vdd/vdd | 80 | Retained_minus_UniformLocal | +2.2525 | [+0.7007, +4.1022] |
| vdd/vdd | 80 | Retained_minus_UniformWide | +4.6297 | [+3.7639, +5.5635] |
| potsdam/potsdam | 14 | Uniform_minus_Retained | -0.0610 | [-0.0948, -0.0317] |
| potsdam/potsdam | 14 | Uniform_minus_UniformMean | +2.7505 | [+2.3382, +3.1227] |
| potsdam/potsdam | 14 | Uniform_minus_UniformLocal | -0.5195 | [-1.2080, +0.2033] |
| potsdam/potsdam | 14 | Uniform_minus_UniformWide | +5.5791 | [+4.8609, +6.2204] |
| potsdam/potsdam | 14 | Retained_minus_UniformMean | +2.8115 | [+2.4121, +3.1794] |
| potsdam/potsdam | 14 | Retained_minus_UniformLocal | -0.4585 | [-1.1394, +0.2560] |
| potsdam/potsdam | 14 | Retained_minus_UniformWide | +5.6400 | [+4.9243, +6.2900] |
| voc20/voc20 | 1449 | Uniform_minus_Retained | +0.0000 | [+0.0000, +0.0000] |
| voc20/voc20 | 1449 | Uniform_minus_UniformMean | +0.1248 | [-0.3682, +0.6087] |
| voc20/voc20 | 1449 | Uniform_minus_UniformLocal | +9.8982 | [+8.4543, +11.4640] |
| voc20/voc20 | 1449 | Uniform_minus_UniformWide | +0.7867 | [+0.6778, +0.9025] |
| voc20/voc20 | 1449 | Retained_minus_UniformMean | +0.1248 | [-0.3682, +0.6087] |
| voc20/voc20 | 1449 | Retained_minus_UniformLocal | +9.8982 | [+8.4543, +11.4640] |
| voc20/voc20 | 1449 | Retained_minus_UniformWide | +0.7867 | [+0.6778, +0.9025] |
| voc21/voc21 | 1449 | Uniform_minus_Retained | +0.0000 | [+0.0000, +0.0000] |
| voc21/voc21 | 1449 | Uniform_minus_UniformMean | +7.0975 | [+6.1528, +8.1147] |
| voc21/voc21 | 1449 | Uniform_minus_UniformLocal | +43.1391 | [+41.7485, +44.6130] |
| voc21/voc21 | 1449 | Uniform_minus_UniformWide | +1.8985 | [+1.7394, +2.0847] |
| voc21/voc21 | 1449 | Retained_minus_UniformMean | +7.0975 | [+6.1528, +8.1147] |
| voc21/voc21 | 1449 | Retained_minus_UniformLocal | +43.1391 | [+41.7485, +44.6130] |
| voc21/voc21 | 1449 | Retained_minus_UniformWide | +1.8985 | [+1.7394, +2.0847] |
| ade150/ade150 | 2000 | Uniform_minus_Retained | +0.0000 | [+0.0000, +0.0000] |
| ade150/ade150 | 2000 | Uniform_minus_UniformMean | +1.3970 | [+0.9832, +1.7891] |
| ade150/ade150 | 2000 | Uniform_minus_UniformLocal | +3.7639 | [+3.2015, +4.3637] |
| ade150/ade150 | 2000 | Uniform_minus_UniformWide | +5.0372 | [+4.4300, +5.6641] |
| ade150/ade150 | 2000 | Retained_minus_UniformMean | +1.3970 | [+0.9832, +1.7891] |
| ade150/ade150 | 2000 | Retained_minus_UniformLocal | +3.7639 | [+3.2015, +4.3637] |
| ade150/ade150 | 2000 | Retained_minus_UniformWide | +5.0372 | [+4.4300, +5.6641] |
| context60/context60 | 5105 | Uniform_minus_Retained | +0.0000 | [+0.0000, +0.0000] |
| context60/context60 | 5105 | Uniform_minus_UniformMean | +0.6801 | [+0.6117, +0.7443] |
| context60/context60 | 5105 | Uniform_minus_UniformLocal | +5.4894 | [+5.0144, +5.9808] |
| context60/context60 | 5105 | Uniform_minus_UniformWide | +2.6419 | [+2.3734, +2.9018] |
| context60/context60 | 5105 | Retained_minus_UniformMean | +0.6801 | [+0.6117, +0.7443] |
| context60/context60 | 5105 | Retained_minus_UniformLocal | +5.4894 | [+5.0144, +5.9808] |
| context60/context60 | 5105 | Retained_minus_UniformWide | +2.6419 | [+2.3734, +2.9018] |
| udd5/udd5 | 40 | Uniform_minus_Retained | -0.0154 | [-0.0901, +0.0513] |
| udd5/udd5 | 40 | Uniform_minus_UniformMean | +1.5526 | [+0.8758, +2.2552] |
| udd5/udd5 | 40 | Uniform_minus_UniformLocal | -0.5898 | [-1.6709, +0.4880] |
| udd5/udd5 | 40 | Uniform_minus_UniformWide | +3.8035 | [+2.7827, +4.9469] |
| udd5/udd5 | 40 | Retained_minus_UniformMean | +1.5680 | [+0.8556, +2.3116] |
| udd5/udd5 | 40 | Retained_minus_UniformLocal | -0.5745 | [-1.6251, +0.4770] |
| udd5/udd5 | 40 | Retained_minus_UniformWide | +3.8188 | [+2.7557, +5.0138] |
| oem/oem | 75 | Uniform_minus_Retained | -0.1497 | [-0.1767, -0.1241] |
| oem/oem | 75 | Uniform_minus_UniformMean | +3.7807 | [+3.3422, +4.1975] |
| oem/oem | 75 | Uniform_minus_UniformLocal | -3.5493 | [-4.0476, -3.0355] |
| oem/oem | 75 | Uniform_minus_UniformWide | +7.4793 | [+6.7132, +8.2163] |
| oem/oem | 75 | Retained_minus_UniformMean | +3.9305 | [+3.4815, +4.3517] |
| oem/oem | 75 | Retained_minus_UniformLocal | -3.3995 | [-3.8880, -2.8982] |
| oem/oem | 75 | Retained_minus_UniformWide | +7.6290 | [+6.8529, +8.3700] |
| vaihingen/vaihingen | 17 | Uniform_minus_Retained | +0.0190 | [+0.0039, +0.0323] |
| vaihingen/vaihingen | 17 | Uniform_minus_UniformMean | +2.3181 | [+1.6779, +2.9726] |
| vaihingen/vaihingen | 17 | Uniform_minus_UniformLocal | +0.9353 | [+0.3118, +1.5905] |
| vaihingen/vaihingen | 17 | Uniform_minus_UniformWide | +5.1914 | [+3.9885, +6.3926] |
| vaihingen/vaihingen | 17 | Retained_minus_UniformMean | +2.2991 | [+1.6581, +2.9532] |
| vaihingen/vaihingen | 17 | Retained_minus_UniformLocal | +0.9163 | [+0.2930, +1.5662] |
| vaihingen/vaihingen | 17 | Retained_minus_UniformWide | +5.1724 | [+3.9667, +6.3692] |
| landcoverai/landcoverai | 41 | Uniform_minus_Retained | +0.0242 | [-0.0058, +0.0702] |
| landcoverai/landcoverai | 41 | Uniform_minus_UniformMean | +1.7496 | [+1.2252, +2.1872] |
| landcoverai/landcoverai | 41 | Uniform_minus_UniformLocal | +3.6236 | [+2.5075, +4.9306] |
| landcoverai/landcoverai | 41 | Uniform_minus_UniformWide | +4.5692 | [+3.6424, +5.3710] |
| landcoverai/landcoverai | 41 | Retained_minus_UniformMean | +1.7255 | [+1.2113, +2.1575] |
| landcoverai/landcoverai | 41 | Retained_minus_UniformLocal | +3.5994 | [+2.4820, +4.8970] |
| landcoverai/landcoverai | 41 | Retained_minus_UniformWide | +4.5451 | [+3.6139, +5.3480] |
| loveda/P | 1669 | Uniform_minus_Retained | -0.0323 | [-0.0630, +0.0043] |
| loveda/P | 1669 | Uniform_minus_UniformMean | +3.4269 | [+3.1662, +3.6937] |
| loveda/P | 1669 | Uniform_minus_UniformLocal | -3.1429 | [-3.7270, -2.5214] |
| loveda/P | 1669 | Uniform_minus_UniformWide | +7.3298 | [+6.8607, +7.8081] |
| loveda/P | 1669 | Retained_minus_UniformMean | +3.4592 | [+3.1750, +3.7401] |
| loveda/P | 1669 | Retained_minus_UniformLocal | -3.1106 | [-3.6733, -2.5202] |
| loveda/P | 1669 | Retained_minus_UniformWide | +7.3620 | [+6.8768, +7.8504] |
| loveda/D | 1669 | Uniform_minus_Retained | +0.0374 | [+0.0131, +0.0662] |
| loveda/D | 1669 | Uniform_minus_UniformMean | +1.7899 | [+1.6060, +1.9718] |
| loveda/D | 1669 | Uniform_minus_UniformLocal | -1.1163 | [-1.4115, -0.7985] |
| loveda/D | 1669 | Uniform_minus_UniformWide | +3.9125 | [+3.5670, +4.2522] |
| loveda/D | 1669 | Retained_minus_UniformMean | +1.7525 | [+1.5555, +1.9470] |
| loveda/D | 1669 | Retained_minus_UniformLocal | -1.1536 | [-1.4430, -0.8502] |
| loveda/D | 1669 | Retained_minus_UniformWide | +3.8751 | [+3.5269, +4.2275] |
| context59/context59 | 5105 | Uniform_minus_Retained | +0.0000 | [+0.0000, +0.0000] |
| context59/context59 | 5105 | Uniform_minus_UniformMean | +0.6500 | [+0.5829, +0.7217] |
| context59/context59 | 5105 | Uniform_minus_UniformLocal | +7.1999 | [+6.6945, +7.7025] |
| context59/context59 | 5105 | Uniform_minus_UniformWide | +2.8097 | [+2.5328, +3.0987] |
| context59/context59 | 5105 | Retained_minus_UniformMean | +0.6500 | [+0.5829, +0.7217] |
| context59/context59 | 5105 | Retained_minus_UniformLocal | +7.1999 | [+6.6945, +7.7025] |
| context59/context59 | 5105 | Retained_minus_UniformWide | +2.8097 | [+2.5328, +3.0987] |
| coco_object81/coco_object81 | 5000 | Uniform_minus_Retained | +0.0000 | [+0.0000, +0.0000] |
| coco_object81/coco_object81 | 5000 | Uniform_minus_UniformMean | +0.9648 | [+0.8841, +1.0508] |
| coco_object81/coco_object81 | 5000 | Uniform_minus_UniformLocal | +20.7132 | [+19.9073, +21.3457] |
| coco_object81/coco_object81 | 5000 | Uniform_minus_UniformWide | -3.1305 | [-3.6692, -2.4577] |
| coco_object81/coco_object81 | 5000 | Retained_minus_UniformMean | +0.9648 | [+0.8841, +1.0508] |
| coco_object81/coco_object81 | 5000 | Retained_minus_UniformLocal | +20.7132 | [+19.9073, +21.3457] |
| coco_object81/coco_object81 | 5000 | Retained_minus_UniformWide | -3.1305 | [-3.6692, -2.4577] |
| coco_stuff171/coco_stuff171 | 5000 | Uniform_minus_Retained | +0.0000 | [+0.0000, +0.0000] |
| coco_stuff171/coco_stuff171 | 5000 | Uniform_minus_UniformMean | +0.1549 | [-0.0257, +0.3714] |
| coco_stuff171/coco_stuff171 | 5000 | Uniform_minus_UniformLocal | +4.2850 | [+3.9715, +4.5345] |
| coco_stuff171/coco_stuff171 | 5000 | Uniform_minus_UniformWide | +1.8371 | [+1.5396, +2.2063] |
| coco_stuff171/coco_stuff171 | 5000 | Retained_minus_UniformMean | +0.1549 | [-0.0257, +0.3714] |
| coco_stuff171/coco_stuff171 | 5000 | Retained_minus_UniformLocal | +4.2850 | [+3.9715, +4.5345] |
| coco_stuff171/coco_stuff171 | 5000 | Retained_minus_UniformWide | +1.8371 | [+1.5396, +2.2063] |
| flair1/flair1 | 10 | Uniform_minus_Retained | -0.0596 | [-0.0786, -0.0381] |
| flair1/flair1 | 10 | Uniform_minus_UniformMean | +1.5402 | [+1.2695, +1.7637] |
| flair1/flair1 | 10 | Uniform_minus_UniformLocal | +1.4316 | [+0.9561, +2.0339] |
| flair1/flair1 | 10 | Uniform_minus_UniformWide | +3.7296 | [+3.1626, +4.1454] |
| flair1/flair1 | 10 | Retained_minus_UniformMean | +1.5998 | [+1.3312, +1.8209] |
| flair1/flair1 | 10 | Retained_minus_UniformLocal | +1.4912 | [+1.0186, +2.0983] |
| flair1/flair1 | 10 | Retained_minus_UniformWide | +3.7892 | [+3.2315, +4.1950] |

Panel means count LoveDA D once; resample groups independently within domains.

primary/Uniform_minus_Retained: -0.0422pp, [-0.0534, -0.0318].
primary/Uniform_minus_UniformMean: +3.3146pp, [+3.0130, +3.6391].
primary/Uniform_minus_UniformLocal: +12.1320pp, [+11.5670, +12.7525].
primary/Uniform_minus_UniformWide: +4.2592pp, [+3.9362, +4.6140].
primary/Retained_minus_UniformMean: +3.3568pp, [+3.0512, +3.6824].
primary/Retained_minus_UniformLocal: +12.1742pp, [+11.6089, +12.7998].
primary/Retained_minus_UniformWide: +4.3014pp, [+3.9784, +4.6556].
eight_rs/Uniform_minus_Retained: -0.0391pp, [-0.0522, -0.0260].
eight_rs/Uniform_minus_UniformMean: +2.1869pp, [+2.0106, +2.3440].
eight_rs/Uniform_minus_UniformLocal: +0.2951pp, [+0.0049, +0.6335].
eight_rs/Uniform_minus_UniformWide: +4.8483pp, [+4.5416, +5.1243].
eight_rs/Retained_minus_UniformMean: +2.2260pp, [+2.0467, +2.3861].
eight_rs/Retained_minus_UniformLocal: +0.3342pp, [+0.0465, +0.6692].
eight_rs/Retained_minus_UniformWide: +4.8874pp, [+4.5735, +5.1669].
