# Background calibration feasibility

Both probes use existing64-image mask-free records and CPU only. No GT is loaded, no new alias or parameter is selected, and all running/full outputs remain unchanged. These are pseudo-witness diagnostics, not mIoU or held-out validation.

## Confidence retention frontier

The largest rejection threshold retaining at least95% of baseline-surviving pseudo true positives is the image-balanced lower5% confidence quantile. Positive and false-background-activation samples each give equal total weight to a contributing image. Ties use strict confidence < threshold rejection. Background here means the image-derived witness label, not GT. Rare image support is shown, not silently fitted.

| Protocol/profile | Class | Positive images | BG FP images | Old threshold | 95% guard | Pseudo TP lost (%) | Pseudo BG FP removed (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| voc21/default | aeroplane | 3 | 3 | 0.132531 | 0.155382 | 4.9687 | 60.5746 |
| voc21/default | bicycle | 3 | 0 | 0.132531 | 0.151915 | 4.6319 | n/a |
| voc21/default | bird | 6 | 0 | 0.132531 | 0.162598 | 4.9991 | n/a |
| voc21/default | boat | 4 | 5 | 0.132531 | 0.141531 | 4.7567 | 2.3380 |
| voc21/default | bottle | 5 | 0 | 0.132531 | 0.133045 | 0.0000 | n/a |
| voc21/default | bus | 7 | 1 | 0.132531 | 0.181542 | 4.9905 | 53.0743 |
| voc21/default | car | 8 | 7 | 0.116544 | 0.116544 | 0.0000 | 0.0000 |
| voc21/default | cat | 7 | 2 | 0.132531 | 0.195448 | 4.9760 | 100.0000 |
| voc21/default | chair | 3 | 1 | 0.132531 | 0.147235 | 0.6199 | 1.5793 |
| voc21/default | cow | 2 | 0 | 0.132531 | 0.168571 | 4.6482 | n/a |
| voc21/default | diningtable | 6 | 2 | 0.132531 | 0.136542 | 4.4854 | 50.0000 |
| voc21/default | dog | 10 | 9 | 0.123400 | 0.126095 | 4.6850 | 1.1745 |
| voc21/default | horse | 1 | 0 | 0.132531 | 0.152449 | 4.8117 | n/a |
| voc21/default | motorbike | 1 | 1 | 0.132531 | 0.158828 | 4.5110 | 95.6612 |
| voc21/default | person | 5 | 7 | 0.132531 | 0.154974 | 4.8445 | 28.0020 |
| voc21/default | pottedplant | 2 | 3 | 0.132531 | 0.176949 | 0.0000 | 100.0000 |
| voc21/default | sheep | 5 | 2 | 0.132531 | 0.180767 | 4.9987 | 100.0000 |
| voc21/default | sofa | 5 | 3 | 0.132531 | 0.232898 | 4.9796 | 100.0000 |
| voc21/default | train | 4 | 2 | 0.132531 | 0.161452 | 4.8296 | 17.8538 |
| voc21/default | tvmonitor | 3 | 3 | 0.132531 | 0.222230 | 4.8325 | 100.0000 |
| voc21/frozen | aeroplane | 3 | 0 | 0.237790 | 0.249908 | 4.5749 | n/a |
| voc21/frozen | bicycle | 3 | 0 | 0.237790 | 0.240017 | 4.1200 | n/a |
| voc21/frozen | bird | 6 | 0 | 0.237790 | 0.255557 | 4.8014 | n/a |
| voc21/frozen | boat | 3 | 0 | 0.237790 | 0.246176 | 4.9373 | n/a |
| voc21/frozen | bottle | 4 | 0 | 0.237790 | 0.243301 | 4.9475 | n/a |
| voc21/frozen | bus | 7 | 0 | 0.237790 | 0.244960 | 4.6520 | n/a |
| voc21/frozen | car | 3 | 0 | 0.237790 | 0.241575 | 4.0213 | n/a |
| voc21/frozen | cat | 7 | 0 | 0.237790 | 0.244646 | 4.9793 | n/a |
| voc21/frozen | chair | 2 | 0 | 0.237790 | 0.244213 | 4.6701 | n/a |
| voc21/frozen | cow | 2 | 0 | 0.237790 | 0.242658 | 4.2734 | n/a |
| voc21/frozen | diningtable | 4 | 0 | 0.237790 | 0.240243 | 0.0000 | n/a |
| voc21/frozen | dog | 9 | 1 | 0.237790 | 0.259783 | 4.9441 | 100.0000 |
| voc21/frozen | horse | 1 | 0 | 0.237790 | 0.242110 | 4.1626 | n/a |
| voc21/frozen | motorbike | 1 | 0 | 0.237790 | 0.241414 | 4.9470 | n/a |
| voc21/frozen | person | 4 | 1 | 0.237790 | 0.242076 | 4.7704 | 3.6138 |
| voc21/frozen | pottedplant | 2 | 0 | 0.237790 | 0.238267 | 0.0000 | n/a |
| voc21/frozen | sheep | 5 | 0 | 0.237790 | 0.255757 | 3.4733 | n/a |
| voc21/frozen | sofa | 5 | 1 | 0.237790 | 0.299841 | 4.9911 | 100.0000 |
| voc21/frozen | train | 4 | 1 | 0.237790 | 0.241258 | 4.8611 | 0.0000 |
| voc21/frozen | tvmonitor | 3 | 0 | 0.237790 | 0.275723 | 4.9311 | n/a |
| coco_object81/default | person | 25 | 13 | 0.062131 | 0.065025 | 4.3688 | 10.7409 |
| coco_object81/default | bicycle | 2 | 1 | 0.046046 | 0.065521 | 3.4125 | 44.8049 |
| coco_object81/default | car | 6 | 4 | 0.046046 | 0.050944 | 2.9042 | 13.5847 |
| coco_object81/default | motorcycle | 3 | 0 | 0.046046 | 0.067470 | 4.9464 | n/a |
| coco_object81/default | airplane | 1 | 0 | 0.046046 | 0.087296 | 4.8770 | n/a |
| coco_object81/default | bus | 4 | 4 | 0.046046 | 0.096094 | 4.9843 | 100.0000 |
| coco_object81/default | train | 1 | 1 | 0.046046 | 0.093772 | 4.7804 | 61.8117 |
| coco_object81/default | truck | 2 | 2 | 0.046046 | 0.046765 | 0.0000 | 10.5708 |
| coco_object81/default | boat | 1 | 1 | 0.046046 | 0.046321 | 0.0000 | 0.0000 |
| coco_object81/default | traffic light | 0 | 5 | 0.046046 | 0.046046 | n/a | 0.0000 |
| coco_object81/default | fire hydrant | 1 | 2 | 0.046046 | 0.074952 | 4.4583 | 93.7947 |
| coco_object81/default | stop sign | 4 | 4 | 0.046046 | 0.048857 | 4.9203 | 10.2744 |
| coco_object81/default | parking meter | 1 | 2 | 0.046046 | 0.067954 | 4.3961 | 100.0000 |
| coco_object81/default | bench | 3 | 2 | 0.046046 | 0.062650 | 4.0902 | 100.0000 |
| coco_object81/default | bird | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | cat | 1 | 0 | 0.046046 | 0.066295 | 4.6594 | n/a |
| coco_object81/default | dog | 2 | 2 | 0.046046 | 0.088555 | 2.1310 | 91.1899 |
| coco_object81/default | horse | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | sheep | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | cow | 1 | 1 | 0.046046 | 0.092587 | 4.8908 | 100.0000 |
| coco_object81/default | elephant | 1 | 1 | 0.046046 | 0.069567 | 4.7513 | 100.0000 |
| coco_object81/default | bear | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | zebra | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | giraffe | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | backpack | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | umbrella | 3 | 0 | 0.046046 | 0.057149 | 3.3935 | n/a |
| coco_object81/default | handbag | 1 | 0 | 0.046046 | 0.068871 | 0.0000 | n/a |
| coco_object81/default | tie | 1 | 0 | 0.046046 | 0.062840 | 0.0000 | n/a |
| coco_object81/default | suitcase | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | frisbee | 1 | 0 | 0.046046 | 0.070938 | 3.5812 | n/a |
| coco_object81/default | skis | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | snowboard | 0 | 1 | 0.046046 | 0.046046 | n/a | 0.0000 |
| coco_object81/default | sports ball | 2 | 0 | 0.046046 | 0.049774 | 0.0000 | n/a |
| coco_object81/default | kite | 1 | 0 | 0.046046 | 0.068959 | 3.9605 | n/a |
| coco_object81/default | baseball bat | 1 | 0 | 0.046046 | 0.058129 | 0.0000 | n/a |
| coco_object81/default | baseball glove | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | skateboard | 3 | 0 | 0.046046 | 0.059832 | 4.8800 | n/a |
| coco_object81/default | surfboard | 2 | 1 | 0.046046 | 0.056962 | 0.0000 | 100.0000 |
| coco_object81/default | tennis racket | 3 | 0 | 0.046046 | 0.058737 | 2.1574 | n/a |
| coco_object81/default | bottle | 6 | 0 | 0.046046 | 0.048872 | 4.4077 | n/a |
| coco_object81/default | wine glass | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | cup | 2 | 0 | 0.046046 | 0.058457 | 0.0000 | n/a |
| coco_object81/default | fork | 1 | 0 | 0.046046 | 0.056818 | 0.0000 | n/a |
| coco_object81/default | knife | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | spoon | 1 | 0 | 0.046046 | 0.081068 | 0.0000 | n/a |
| coco_object81/default | bowl | 3 | 0 | 0.046046 | 0.064675 | 1.9443 | n/a |
| coco_object81/default | banana | 1 | 0 | 0.046046 | 0.068634 | 4.8286 | n/a |
| coco_object81/default | apple | 1 | 0 | 0.046046 | 0.067132 | 4.8262 | n/a |
| coco_object81/default | sandwich | 2 | 0 | 0.046046 | 0.047589 | 1.8303 | n/a |
| coco_object81/default | orange | 2 | 0 | 0.046046 | 0.053294 | 4.6545 | n/a |
| coco_object81/default | broccoli | 4 | 0 | 0.046046 | 0.063362 | 4.9682 | n/a |
| coco_object81/default | carrot | 2 | 0 | 0.046046 | 0.058018 | 4.4915 | n/a |
| coco_object81/default | hot dog | 1 | 0 | 0.046046 | 0.046046 | 0.0000 | n/a |
| coco_object81/default | pizza | 3 | 0 | 0.046046 | 0.052459 | 0.0000 | n/a |
| coco_object81/default | donut | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | cake | 3 | 0 | 0.046046 | 0.046550 | 2.3510 | n/a |
| coco_object81/default | chair | 3 | 1 | 0.046046 | 0.073216 | 0.6961 | 100.0000 |
| coco_object81/default | couch | 4 | 1 | 0.046046 | 0.053855 | 1.0863 | 0.0000 |
| coco_object81/default | potted plant | 6 | 0 | 0.046046 | 0.047218 | 0.6182 | n/a |
| coco_object81/default | bed | 2 | 1 | 0.046046 | 0.088802 | 4.8673 | 100.0000 |
| coco_object81/default | dining table | 7 | 9 | 0.046046 | 0.073332 | 4.3733 | 69.0789 |
| coco_object81/default | toilet | 3 | 1 | 0.046046 | 0.047629 | 0.0000 | 0.0000 |
| coco_object81/default | tv | 5 | 13 | 0.046046 | 0.052766 | 0.0000 | 27.7079 |
| coco_object81/default | laptop | 3 | 1 | 0.046046 | 0.078741 | 4.9798 | 100.0000 |
| coco_object81/default | mouse | 1 | 2 | 0.046046 | 0.139546 | 3.9519 | 100.0000 |
| coco_object81/default | remote | 1 | 0 | 0.046046 | 0.059817 | 4.6928 | n/a |
| coco_object81/default | keyboard | 1 | 0 | 0.046046 | 0.105523 | 0.0000 | n/a |
| coco_object81/default | cell phone | 4 | 1 | 0.046046 | 0.078064 | 4.4027 | 100.0000 |
| coco_object81/default | microwave | 1 | 0 | 0.046046 | 0.142707 | 4.5653 | n/a |
| coco_object81/default | oven | 2 | 0 | 0.046046 | 0.080914 | 4.9590 | n/a |
| coco_object81/default | toaster | 1 | 0 | 0.046046 | 0.049289 | 0.0000 | n/a |
| coco_object81/default | sink | 3 | 0 | 0.046046 | 0.054798 | 4.9397 | n/a |
| coco_object81/default | refrigerator | 1 | 0 | 0.046046 | 0.054298 | 0.0000 | n/a |
| coco_object81/default | book | 2 | 0 | 0.046046 | 0.067879 | 4.9762 | n/a |
| coco_object81/default | clock | 4 | 2 | 0.046046 | 0.082515 | 0.1075 | 75.2697 |
| coco_object81/default | vase | 1 | 0 | 0.046046 | 0.054654 | 0.0000 | n/a |
| coco_object81/default | scissors | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | teddy bear | 1 | 0 | 0.046046 | 0.078515 | 4.2220 | n/a |
| coco_object81/default | hair drier | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/default | toothbrush | 0 | 0 | 0.046046 | 0.046046 | n/a | n/a |
| coco_object81/frozen | person | 20 | 7 | 0.093264 | 0.095671 | 4.9599 | 2.2816 |
| coco_object81/frozen | bicycle | 2 | 1 | 0.093264 | 0.093985 | 2.8374 | 100.0000 |
| coco_object81/frozen | car | 3 | 0 | 0.093264 | 0.094303 | 3.7051 | n/a |
| coco_object81/frozen | motorcycle | 3 | 0 | 0.093264 | 0.098656 | 4.9960 | n/a |
| coco_object81/frozen | airplane | 1 | 0 | 0.093264 | 0.101093 | 4.6027 | n/a |
| coco_object81/frozen | bus | 4 | 0 | 0.093264 | 0.104619 | 4.9554 | n/a |
| coco_object81/frozen | train | 1 | 1 | 0.093264 | 0.103199 | 4.7495 | 36.7703 |
| coco_object81/frozen | truck | 1 | 0 | 0.093264 | 0.098547 | 4.7350 | n/a |
| coco_object81/frozen | boat | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | traffic light | 0 | 1 | 0.093264 | 0.093264 | n/a | 0.0000 |
| coco_object81/frozen | fire hydrant | 1 | 1 | 0.093264 | 0.098649 | 4.5747 | 30.0136 |
| coco_object81/frozen | stop sign | 2 | 1 | 0.093264 | 0.095804 | 0.0000 | 38.8744 |
| coco_object81/frozen | parking meter | 1 | 0 | 0.093264 | 0.093289 | 0.0000 | n/a |
| coco_object81/frozen | bench | 3 | 0 | 0.093264 | 0.118083 | 1.7838 | n/a |
| coco_object81/frozen | bird | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | cat | 1 | 0 | 0.093264 | 0.093521 | 4.8898 | n/a |
| coco_object81/frozen | dog | 2 | 1 | 0.093264 | 0.096299 | 4.9750 | 78.6505 |
| coco_object81/frozen | horse | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | sheep | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | cow | 1 | 0 | 0.093264 | 0.101830 | 4.9199 | n/a |
| coco_object81/frozen | elephant | 1 | 0 | 0.093264 | 0.100442 | 4.7734 | n/a |
| coco_object81/frozen | bear | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | zebra | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | giraffe | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | backpack | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | umbrella | 3 | 0 | 0.093264 | 0.099742 | 0.0000 | n/a |
| coco_object81/frozen | handbag | 1 | 0 | 0.093264 | 0.103201 | 0.0000 | n/a |
| coco_object81/frozen | tie | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | suitcase | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | frisbee | 1 | 0 | 0.093264 | 0.132984 | 3.2619 | n/a |
| coco_object81/frozen | skis | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | snowboard | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | sports ball | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | kite | 1 | 0 | 0.093264 | 0.096748 | 4.3077 | n/a |
| coco_object81/frozen | baseball bat | 1 | 0 | 0.093264 | 0.098949 | 0.0000 | n/a |
| coco_object81/frozen | baseball glove | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | skateboard | 3 | 0 | 0.093264 | 0.093650 | 0.1907 | n/a |
| coco_object81/frozen | surfboard | 2 | 0 | 0.093264 | 0.093502 | 2.1361 | n/a |
| coco_object81/frozen | tennis racket | 2 | 0 | 0.093264 | 0.098787 | 4.7318 | n/a |
| coco_object81/frozen | bottle | 3 | 0 | 0.093264 | 0.102302 | 3.4276 | n/a |
| coco_object81/frozen | wine glass | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | cup | 1 | 0 | 0.093264 | 0.110821 | 4.3494 | n/a |
| coco_object81/frozen | fork | 1 | 0 | 0.093264 | 0.103308 | 0.0000 | n/a |
| coco_object81/frozen | knife | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | spoon | 1 | 0 | 0.093264 | 0.103963 | 0.0000 | n/a |
| coco_object81/frozen | bowl | 2 | 0 | 0.093264 | 0.103604 | 4.9818 | n/a |
| coco_object81/frozen | banana | 1 | 0 | 0.093264 | 0.100516 | 4.8929 | n/a |
| coco_object81/frozen | apple | 1 | 0 | 0.093264 | 0.102148 | 4.9335 | n/a |
| coco_object81/frozen | sandwich | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | orange | 2 | 0 | 0.093264 | 0.097284 | 4.8978 | n/a |
| coco_object81/frozen | broccoli | 4 | 0 | 0.093264 | 0.099422 | 4.3883 | n/a |
| coco_object81/frozen | carrot | 2 | 0 | 0.093264 | 0.094889 | 0.0000 | n/a |
| coco_object81/frozen | hot dog | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | pizza | 2 | 0 | 0.093264 | 0.100877 | 3.1581 | n/a |
| coco_object81/frozen | donut | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | cake | 2 | 0 | 0.093264 | 0.097537 | 0.7793 | n/a |
| coco_object81/frozen | chair | 3 | 0 | 0.093264 | 0.095516 | 3.9719 | n/a |
| coco_object81/frozen | couch | 3 | 0 | 0.093264 | 0.123230 | 3.2035 | n/a |
| coco_object81/frozen | potted plant | 3 | 0 | 0.093264 | 0.096344 | 4.7963 | n/a |
| coco_object81/frozen | bed | 2 | 0 | 0.093264 | 0.098235 | 4.7870 | n/a |
| coco_object81/frozen | dining table | 7 | 1 | 0.093264 | 0.095533 | 0.4402 | 0.0000 |
| coco_object81/frozen | toilet | 1 | 0 | 0.093264 | 0.106098 | 4.3540 | n/a |
| coco_object81/frozen | tv | 2 | 0 | 0.093264 | 0.097879 | 3.9404 | n/a |
| coco_object81/frozen | laptop | 3 | 0 | 0.093264 | 0.097189 | 4.6864 | n/a |
| coco_object81/frozen | mouse | 1 | 1 | 0.093264 | 0.153985 | 3.9519 | 100.0000 |
| coco_object81/frozen | remote | 1 | 0 | 0.093264 | 0.094601 | 4.8822 | n/a |
| coco_object81/frozen | keyboard | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | cell phone | 4 | 0 | 0.093264 | 0.093331 | 0.0000 | n/a |
| coco_object81/frozen | microwave | 1 | 0 | 0.093264 | 0.159793 | 4.9075 | n/a |
| coco_object81/frozen | oven | 1 | 0 | 0.093264 | 0.095254 | 4.7066 | n/a |
| coco_object81/frozen | toaster | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | sink | 2 | 0 | 0.093264 | 0.098764 | 4.9016 | n/a |
| coco_object81/frozen | refrigerator | 1 | 0 | 0.093264 | 0.093868 | 0.0000 | n/a |
| coco_object81/frozen | book | 2 | 0 | 0.093264 | 0.094528 | 0.0000 | n/a |
| coco_object81/frozen | clock | 4 | 0 | 0.093264 | 0.104407 | 0.5195 | n/a |
| coco_object81/frozen | vase | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | scissors | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | teddy bear | 1 | 0 | 0.093264 | 0.117070 | 4.2672 | n/a |
| coco_object81/frozen | hair drier | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |
| coco_object81/frozen | toothbrush | 0 | 0 | 0.093264 | 0.093264 | n/a | n/a |

## Full Residual Union Compensation

The existing score is S=G+H(W-G+V), and its admission potential V does not depend on G. Replacing only the residual-background local log-mean-exp by log-sum-exp changes G by log(n_bg). The exact central-tile score difference is therefore (1-H1)*log(n_bg) in the background column. All other logits, words, templates and reader equations stay fixed in this algebraic counterfactual. This is full normalization compensation, not an arbitrary learned bias. It is not a claim that background words are equally valid, nor that weaker compensation is ruled out.

| Protocol | Rejection | Original pseudo error (%) | Union pseudo error (%) | Paired delta (pp) | Original BG pseudo error (%) | Union BG pseudo error (%) |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| voc21 | argmax | 4.0949 | 47.6961 | +43.6012 | 8.0302 | 0.0228 |
| voc21 | rejected | 20.7375 | 48.0649 | +27.3274 | 0.4610 | 0.0228 |
| coco_object81 | argmax | 9.4310 | 51.7180 | +42.2870 | 12.7224 | 0.0790 |
| coco_object81 | rejected | 34.2086 | 52.1796 | +17.9710 | 2.0903 | 0.0790 |

The full background compensation strongly harms the combined pseudo objective despite reducing background error. It is not promoted to a GPU experiment. The threshold frontier is in-sample feasibility only and does not establish improved segmentation; several severe-error classes have overlapping confidence or sparse support. The separate fixed-word qualification full suite remains queued with its original rules.

Reproduce: F:/APP/codetool/anaconda3/python.exe tools/background_replay_feasibility.py (add --collect to retrieve the original image-only witness records). Inputs and detailed JSON: research/natural_background_replay_feasibility_20261004/.
