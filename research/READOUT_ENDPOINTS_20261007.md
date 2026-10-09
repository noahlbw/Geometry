# Selected task model: matched endpoint and transport audit

Same words, profiles, shared bounded image observations and stitching. Wide endpoint is sampled onto the local patch lattice, not official VIP inference. PC60 local residual preprocessing still uses the retained coupled foreground rival; this is a controlled final-readout endpoint, not an independent local encoder model. Labels score predictions only and do not select a route. Exact Frozen confusion replay, full unique coverage, paired target support and transition identities verified. Developed targets, exploratory diagnosis. No new production candidate or parameter tuning.

| Dataset | Images | Local endpoint | Wide endpoint | Frozen coupling |
| --- | ---: | ---: | ---: | ---: |
| vdd | 80 | 53.1452 | 50.6085 | 55.2382 |
| potsdam | 504 | 50.3615 | 44.1488 | 49.7888 |
| voc21 | 1449 | 27.3117 | 68.5522 | 70.4508 |
| context60 | 5105 | 36.1110 | 38.9585 | 41.6004 |
| ade150 | 2000 | 27.4223 | 26.1490 | 31.1862 |

## vdd

Counts are full-resolution valid target pixels. Fixed/broken are endpoint→Frozen transitions, not IoU gains. Correct-pixel gain can coexist with lower macro mIoU.

| Class | Local→Frozen fixed | Local→Frozen broken | Wide→Frozen fixed | Wide→Frozen broken |
| --- | ---: | ---: | ---: | ---: |
| other | 12793876 | 4565858 | 16302641 | 13812785 |
| wall | 133078 | 2752595 | 4648697 | 548298 |
| road | 306094 | 483210 | 3096975 | 246624 |
| vegetation | 4768972 | 12373445 | 31557432 | 11433692 |
| vehicle | 16796 | 284279 | 607709 | 88725 |
| roof | 5751315 | 152918 | 1927129 | 2015061 |
| water | 15403114 | 373271 | 823622 | 5529504 |

## potsdam

Counts are full-resolution valid target pixels. Fixed/broken are endpoint→Frozen transitions, not IoU gains. Correct-pixel gain can coexist with lower macro mIoU.

| Class | Local→Frozen fixed | Local→Frozen broken | Wide→Frozen fixed | Wide→Frozen broken |
| --- | ---: | ---: | ---: | ---: |
| impervious surface | 6337215 | 2052797 | 9309011 | 4029337 |
| building | 1715821 | 363451 | 1151181 | 1432874 |
| low vegetation | 1740458 | 10010403 | 14656362 | 2108802 |
| tree | 1369530 | 5071405 | 12581763 | 1562297 |
| car | 34987 | 44319 | 364427 | 58447 |
| clutter | 960249 | 1079471 | 1158493 | 1148724 |

## voc21

Counts are full-resolution valid target pixels. Fixed/broken are endpoint→Frozen transitions, not IoU gains. Correct-pixel gain can coexist with lower macro mIoU.

| Class | Local→Frozen fixed | Local→Frozen broken | Wide→Frozen fixed | Wide→Frozen broken |
| --- | ---: | ---: | ---: | ---: |
| background | 164624935 | 20 | 2550411 | 932813 |
| aeroplane | 2591 | 71310 | 32139 | 22047 |
| bicycle | 13247 | 173283 | 33241 | 39663 |
| bird | 186 | 255623 | 55772 | 50182 |
| boat | 715 | 286480 | 19053 | 17780 |
| bottle | 10360 | 472698 | 54174 | 70831 |
| bus | 404241 | 46637 | 67825 | 14497 |
| car | 7310 | 578023 | 97884 | 73097 |
| cat | 84334 | 80668 | 60849 | 25787 |
| chair | 377254 | 81787 | 62468 | 42136 |
| cow | 326283 | 40213 | 37750 | 14804 |
| diningtable | 24397 | 607432 | 67109 | 35350 |
| dog | 401 | 161013 | 44996 | 26002 |
| horse | 4864 | 147442 | 42829 | 29471 |
| motorbike | 9102 | 497412 | 92984 | 108827 |
| person | 1614158 | 824196 | 320238 | 223153 |
| pottedplant | 891 | 501681 | 32608 | 48122 |
| sheep | 13565 | 68485 | 32575 | 16301 |
| sofa | 22981 | 237826 | 47762 | 25457 |
| train | 24299 | 247647 | 48508 | 31401 |
| tvmonitor | 22960 | 193331 | 42669 | 31429 |

## context60

Counts are full-resolution valid target pixels. Fixed/broken are endpoint→Frozen transitions, not IoU gains. Correct-pixel gain can coexist with lower macro mIoU.

| Class | Local→Frozen fixed | Local→Frozen broken | Wide→Frozen fixed | Wide→Frozen broken |
| --- | ---: | ---: | ---: | ---: |
| background | 682197 | 27663886 | 9239919 | 279385 |
| aeroplane | 681217 | 10852 | 89171 | 42740 |
| bag | 182476 | 24350 | 108287 | 39631 |
| bed | 428943 | 40206 | 123721 | 134057 |
| bedclothes | 162836 | 1474532 | 478278 | 55403 |
| bench | 133477 | 20076 | 38797 | 10789 |
| bicycle | 1444695 | 63780 | 259527 | 263058 |
| bird | 338122 | 31577 | 322352 | 38048 |
| boat | 768447 | 28039 | 208066 | 87218 |
| book | 24670 | 77502 | 37892 | 33780 |
| bottle | 1759522 | 19227 | 307060 | 155560 |
| building | 14344581 | 160401 | 1829569 | 2535085 |
| bus | 1575338 | 10969 | 97759 | 209664 |
| cabinet | 1106917 | 134403 | 376986 | 176168 |
| car | 2281353 | 182522 | 628822 | 327369 |
| cat | 1787088 | 11396 | 183686 | 100840 |
| ceiling | 1312328 | 83374 | 327458 | 108055 |
| chair | 4156746 | 29821 | 459269 | 713550 |
| cloth | 1794024 | 21666 | 197612 | 234496 |
| computer | 158956 | 53981 | 42921 | 12680 |
| cow | 1383407 | 1232 | 59348 | 38265 |
| cup | 58004 | 37286 | 69594 | 14499 |
| curtain | 319534 | 27336 | 139068 | 47245 |
| dog | 337602 | 87473 | 317656 | 42729 |
| door | 623674 | 65271 | 273265 | 65128 |
| fence | 1994199 | 82288 | 344555 | 418925 |
| floor | 4602041 | 426393 | 1154105 | 631602 |
| flower | 2052 | 283433 | 141498 | 477 |
| food | 1504 | 612219 | 317031 | 285 |
| grass | 7585322 | 849308 | 2683726 | 796790 |
| ground | 396332 | 2905778 | 1275245 | 38691 |
| horse | 611998 | 23262 | 235991 | 53119 |
| keyboard | 13062 | 45480 | 13896 | 19098 |
| light | 1826 | 406221 | 134164 | 140 |
| motorbike | 1814615 | 116849 | 393245 | 143465 |
| mountain | 1735656 | 46438 | 398564 | 176053 |
| mouse | 126 | 5133 | 8698 | 281 |
| person | 57615082 | 2228 | 257669 | 2879760 |
| plate | 352850 | 7574 | 28754 | 30302 |
| platform | 11721 | 311823 | 77579 | 8259 |
| pottedplant | 1161128 | 133735 | 433471 | 172314 |
| road | 2396856 | 626781 | 1144475 | 311354 |
| rock | 403499 | 267647 | 294126 | 59386 |
| sheep | 464321 | 9940 | 172678 | 23357 |
| shelves | 94589 | 383405 | 241898 | 20945 |
| sidewalk | 614170 | 101267 | 108862 | 45017 |
| sign | 5464 | 310140 | 148468 | 7332 |
| sky | 19245374 | 423513 | 2165201 | 1828064 |
| snow | 164548 | 65954 | 220752 | 33369 |
| sofa | 1426707 | 423575 | 614781 | 229904 |
| table | 1370133 | 804079 | 1211348 | 230279 |
| track | 6 | 178834 | 10 | 0 |
| train | 678692 | 29162 | 127734 | 63777 |
| tree | 7106162 | 1835192 | 3582277 | 976570 |
| truck | 93622 | 31115 | 38531 | 38955 |
| tvmonitor | 3020387 | 2158 | 167969 | 525650 |
| wall | 14747926 | 674456 | 3884593 | 1277690 |
| water | 10857679 | 16247 | 383842 | 782430 |
| window | 460057 | 580323 | 715850 | 86290 |
| wood | 343004 | 134979 | 180780 | 65623 |

## ade150

Counts are full-resolution valid target pixels. Fixed/broken are endpoint→Frozen transitions, not IoU gains. Correct-pixel gain can coexist with lower macro mIoU.

| Class | Local→Frozen fixed | Local→Frozen broken | Wide→Frozen fixed | Wide→Frozen broken |
| --- | ---: | ---: | ---: | ---: |
| wall | 8028757 | 1765443 | 6498155 | 3931040 |
| building | 14658705 | 8613 | 418289 | 6863049 |
| sky | 8685869 | 82669 | 1417503 | 1145172 |
| floor | 1552485 | 519617 | 2004460 | 1364380 |
| tree | 3391679 | 520728 | 2207237 | 541665 |
| ceiling | 1402513 | 67841 | 618809 | 362195 |
| road | 421307 | 382615 | 1150216 | 174136 |
| bed | 93812 | 927824 | 3769640 | 74571 |
| windowpane | 742808 | 206632 | 1005968 | 424898 |
| grass | 3499488 | 42150 | 545865 | 1577381 |
| cabinet | 1619040 | 143625 | 752701 | 612826 |
| sidewalk | 751527 | 101323 | 416073 | 228982 |
| person | 4275383 | 4490 | 47699 | 1758045 |
| earth | 735645 | 2584 | 206773 | 362118 |
| door | 445576 | 47200 | 408956 | 212097 |
| table | 477507 | 112110 | 386867 | 155124 |
| mountain | 1049969 | 9085 | 247473 | 539319 |
| plant | 767516 | 104900 | 577702 | 349375 |
| curtain | 355380 | 12299 | 173327 | 149281 |
| chair | 1586993 | 106538 | 680859 | 261049 |
| car | 421971 | 26923 | 161137 | 92666 |
| water | 1471792 | 46930 | 217636 | 434107 |
| painting | 102266 | 246750 | 627473 | 47548 |
| sofa | 260791 | 712 | 22867 | 53534 |
| shelf | 351138 | 58256 | 226651 | 238092 |
| house | 10839 | 867901 | 873401 | 18 |
| sea | 171957 | 12994 | 102116 | 39144 |
| mirror | 12626 | 229796 | 426042 | 13647 |
| rug | 39101 | 44198 | 168189 | 32327 |
| field | 5293 | 428002 | 794923 | 889 |
| armchair | 26932 | 137315 | 131635 | 13026 |
| seat | 1196 | 743179 | 433168 | 8 |
| fence | 76187 | 35713 | 127747 | 34775 |
| desk | 226648 | 11181 | 73094 | 94806 |
| rock | 129424 | 67798 | 276800 | 21960 |
| wardrobe | 451426 | 410 | 15152 | 307107 |
| lamp | 100391 | 16721 | 127191 | 39108 |
| bathtub | 89876 | 3309 | 32363 | 61213 |
| railing | 35855 | 74048 | 209173 | 16278 |
| cushion | 2460 | 497593 | 381988 | 16 |
| base | 3555 | 14111 | 23844 | 5692 |
| box | 1471 | 188530 | 160075 | 740 |
| column | 16376 | 127423 | 301529 | 3819 |
| signboard | 45310 | 58954 | 164030 | 27124 |
| chest of drawers | 30425 | 8224 | 28888 | 27086 |
| counter | 99946 | 23029 | 124420 | 30263 |
| sand | 85409 | 1403 | 48083 | 15162 |
| sink | 50217 | 3162 | 25930 | 25408 |
| skyscraper | 20767 | 348916 | 303344 | 669 |
| fireplace | 24866 | 1306 | 12891 | 8782 |
| refrigerator | 74407 | 625 | 8949 | 48010 |
| grandstand | 85232 | 27966 | 106137 | 29414 |
| path | 927 | 264539 | 120081 | 780 |
| stairs | 43154 | 42859 | 185791 | 22739 |
| runway | 1231 | 350102 | 381075 | 79 |
| case | 136389 | 1773 | 31457 | 60234 |
| pool table | 127297 | 83 | 3560 | 35530 |
| pillow | 48115 | 55291 | 108343 | 44600 |
| screen door | 349 | 839 | 498 | 0 |
| stairway | 21642 | 41980 | 82800 | 16001 |
| river | 26943 | 2272 | 9249 | 10158 |
| bridge | 820 | 86764 | 241533 | 240 |
| bookcase | 1620 | 79229 | 109985 | 1541 |
| blind | 5382 | 284044 | 335464 | 1641 |
| coffee table | 47082 | 33460 | 116543 | 45027 |
| toilet | 50407 | 903 | 14793 | 4825 |
| flower | 8723 | 139869 | 95468 | 0 |
| book | 9513 | 25775 | 60099 | 2915 |
| hill | 30666 | 73182 | 126284 | 1422 |
| bench | 184466 | 5107 | 71674 | 10157 |
| countertop | 3472 | 71187 | 143759 | 1645 |
| stove | 103997 | 252 | 4960 | 63218 |
| palm | 40975 | 4167 | 22343 | 5843 |
| kitchen island | 42235 | 4223 | 42163 | 24052 |
| computer | 41193 | 9631 | 61401 | 52263 |
| swivel chair | 185 | 103124 | 36218 | 0 |
| boat | 121517 | 839 | 8629 | 16406 |
| bar | 17664 | 38653 | 92186 | 6983 |
| arcade machine | 680 | 27728 | 22139 | 78 |
| hovel | 109177 | 2084 | 36005 | 27262 |
| bus | 58965 | 433 | 4599 | 14122 |
| towel | 635 | 25186 | 57395 | 1059 |
| light | 218 | 131303 | 51297 | 0 |
| truck | 18450 | 1439 | 7739 | 9247 |
| tower | 1886 | 77178 | 206491 | 0 |
| chandelier | 691 | 8267 | 39859 | 2697 |
| awning | 8125 | 10033 | 45665 | 1771 |
| streetlight | 18079 | 42451 | 84920 | 7827 |
| booth | 75589 | 138 | 9187 | 31986 |
| television receiver | 39791 | 971 | 5886 | 14406 |
| airplane | 8029 | 51 | 3047 | 2921 |
| dirt track | 1221 | 2462 | 2227 | 2362 |
| apparel | 4373 | 25472 | 22200 | 6176 |
| pole | 50 | 79721 | 113313 | 59 |
| land | 4472 | 41434 | 65419 | 0 |
| bannister | 65214 | 582 | 7910 | 30371 |
| escalator | 42717 | 4821 | 41250 | 20788 |
| ottoman | 26259 | 12896 | 14439 | 4928 |
| bottle | 26826 | 8956 | 26337 | 15887 |
| buffet | 13809 | 8672 | 10268 | 35369 |
| poster | 41455 | 4729 | 20596 | 18726 |
| stage | 1733 | 20813 | 56618 | 65 |
| van | 878 | 46421 | 70771 | 770 |
| ship | 0 | 7294 | 26057 | 227 |
| fountain | 3066 | 56247 | 62183 | 1538 |
| conveyer belt | 32427 | 3752 | 37096 | 2805 |
| canopy | 3997 | 54092 | 113027 | 1698 |
| washer | 30037 | 36131 | 111981 | 2216 |
| plaything | 8122 | 27253 | 46526 | 687 |
| swimming pool | 3129 | 4548 | 14260 | 415 |
| stool | 664 | 30097 | 53937 | 203 |
| barrel | 0 | 104 | 776 | 0 |
| basket | 5851 | 16078 | 76467 | 1000 |
| waterfall | 1415 | 29138 | 43266 | 111 |
| tent | 7077 | 41 | 1035 | 33 |
| bag | 13012 | 28231 | 76454 | 1743 |
| minibike | 8008 | 4443 | 20309 | 2624 |
| cradle | 15576 | 119 | 3346 | 10799 |
| oven | 1392 | 6372 | 22514 | 1422 |
| ball | 4517 | 6066 | 25975 | 10625 |
| food | 3872 | 167075 | 257148 | 859 |
| step | 18 | 2930 | 6603 | 42 |
| tank | 5197 | 68418 | 119166 | 15 |
| trade name | 730 | 9963 | 8296 | 171 |
| microwave | 33190 | 6131 | 31762 | 8200 |
| pot | 18253 | 25155 | 93822 | 21385 |
| animal | 99863 | 10920 | 194481 | 4909 |
| bicycle | 2795 | 4529 | 16871 | 1654 |
| lake | 381 | 226382 | 38331 | 1 |
| dishwasher | 5521 | 15652 | 28553 | 3104 |
| screen | 21236 | 6672 | 43895 | 2647 |
| blanket | 7946 | 33029 | 39238 | 6279 |
| sculpture | 4453 | 18434 | 68453 | 151 |
| hood | 10536 | 7621 | 19839 | 3870 |
| sconce | 99 | 28263 | 73836 | 51 |
| vase | 2737 | 12820 | 31710 | 490 |
| traffic light | 19649 | 5377 | 26366 | 14421 |
| tray | 1082 | 28011 | 29887 | 618 |
| ashcan | 10524 | 1814 | 17421 | 5837 |
| fan | 3 | 53832 | 84039 | 0 |
| pier | 2892 | 271 | 3848 | 1398 |
| crt screen | 476 | 160 | 720 | 3435 |
| plate | 6526 | 4634 | 22767 | 5684 |
| monitor | 68618 | 3711 | 38400 | 19411 |
| bulletin board | 416 | 4787 | 11642 | 2235 |
| shower | 197 | 865 | 4741 | 134 |
| radiator | 118 | 10142 | 61918 | 272 |
| glass | 509 | 26926 | 15080 | 155 |
| clock | 6671 | 5342 | 20411 | 2293 |
| flag | 49 | 12611 | 33041 | 35 |
