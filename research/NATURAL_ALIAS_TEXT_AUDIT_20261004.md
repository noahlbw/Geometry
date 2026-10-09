# Frozen natural alias text audit

No images or masks are loaded. These cosine diagnostics do not measure whether an alias helps segmentation. Residual background is exempt from canonical text ownership tests: its concrete scene words are not synonyms of the abstract word background. Running vocabularies and protected anchors are unchanged. Separate, untested curated input manifests are saved without replacing the sources.

| Protocol | Aliases | Noncanonical aliases closer to a rival anchor | Text-only reanchor proposals |
| --- | ---: | ---: | ---: |
| ade150 | 382 | 40 | 64 |
| coco_object81 | 325 | 1 | 74 |
| coco_stuff171 | 639 | 43 | 142 |
| voc20 | 116 | 1 | 17 |
| voc21 | 172 | 1 | 17 |

## Taxonomy-Supported Input Repairs

Only the following explicit sense errors and empty strings are removed in separate taxonomy_curated_selection.json candidates. ADE objectInfo150.txt category 41 defines base as base/pedestal/stand; category 89 defines booth as booth/cubicle/stall/kiosk. Category 93 defines apparel as wearing apparel/dress/clothes, while category 36 defines wardrobe as wardrobe/closet/press. No class-frequency fields, images or masks select these repairs. Existing template, tau, tem and background threshold stay frozen. The candidates have not been evaluated and are not automatically promoted.

| Protocol | Parent | Removed alias | Reason |
| --- | --- | --- | --- |
| ade150 | base | stall | taxonomy_word_sense_conflict |
| ade150 | base | sales booth | taxonomy_word_sense_conflict |
| ade150 | apparel | clothes closet | taxonomy_word_sense_conflict |
| ade150 | apparel | clothespress | taxonomy_word_sense_conflict |
| coco_stuff171 | floor-marble | (empty string) | empty_query |

## Cross-Class Conflicts

A negative margin can also reflect a weak canonical label or an ontology overlap; it is not sufficient grounds for deletion.

| Protocol | Parent | Alias | Rival | Own cosine | Rival cosine | Margin |
| --- | --- | --- | --- | ---: | ---: | ---: |
| ade150 | base | sales booth | booth | 0.6694 | 0.8809 | -0.2116 |
| ade150 | crt screen | crtscreen | screen | 0.7727 | 0.9614 | -0.1887 |
| ade150 | base | stall | booth | 0.7389 | 0.8713 | -0.1324 |
| ade150 | apparel | clothes closet | wardrobe | 0.8064 | 0.9286 | -0.1223 |
| ade150 | minibike | motorcycle | bicycle | 0.8075 | 0.9197 | -0.1121 |
| ade150 | minibike | motorbike | bicycle | 0.8120 | 0.9157 | -0.1037 |
| ade150 | earth | ground | land | 0.7848 | 0.8856 | -0.1008 |
| ade150 | cradle | baby’s bed | bed  | 0.8055 | 0.8911 | -0.0856 |
| ade150 | light | skylight | ceiling | 0.7253 | 0.8086 | -0.0833 |
| ade150 | case | vitrine | cabinet | 0.7545 | 0.8275 | -0.0730 |
| ade150 | apparel | clothespress | bicycle | 0.6544 | 0.7229 | -0.0685 |
| ade150 | earth | exposed soil | land | 0.7481 | 0.8148 | -0.0667 |
| ade150 | cradle | baby bed | bed  | 0.8265 | 0.8901 | -0.0635 |
| ade150 | buffet | china closet | cabinet | 0.7410 | 0.8031 | -0.0621 |
| ade150 | building | frontal | base | 0.7004 | 0.7621 | -0.0616 |
| coco_object81 | motorcycle | motorized bike | bicycle | 0.8958 | 0.9078 | -0.0120 |
| coco_stuff171 | floor-marble |  | solid-other | 0.6504 | 0.9007 | -0.2504 |
| coco_stuff171 | desk-stuff | tabletop | table | 0.7994 | 0.9359 | -0.1364 |
| coco_stuff171 | ground-other | playground surface | playingfield | 0.6783 | 0.7983 | -0.1201 |
| coco_stuff171 | furniture-other | tv stand | tv | 0.7801 | 0.8887 | -0.1086 |
| coco_stuff171 | ground-other | concrete ground | wall-concrete | 0.7823 | 0.8720 | -0.0897 |
| coco_stuff171 | furniture-other | shoe cabinet | cupboard | 0.7532 | 0.8391 | -0.0859 |
| coco_stuff171 | structural-other | building structure | building-other | 0.8601 | 0.9458 | -0.0857 |
| coco_stuff171 | furniture-other | sideboard | cupboard | 0.8173 | 0.8921 | -0.0747 |
| coco_stuff171 | ground-other | cement ground | wall-concrete | 0.7743 | 0.8490 | -0.0747 |
| coco_stuff171 | solid-other | solid surface | counter | 0.6992 | 0.7723 | -0.0732 |
| coco_stuff171 | building-other | frontal | solid-other | 0.7192 | 0.7922 | -0.0730 |
| coco_stuff171 | plant-other | plant not in a pot | potted plant | 0.8585 | 0.9250 | -0.0665 |
| coco_stuff171 | rug | floor carpet | carpet | 0.9227 | 0.9792 | -0.0565 |
| coco_stuff171 | furniture-other | dresser | cabinet | 0.8267 | 0.8800 | -0.0533 |
| coco_stuff171 | structural-other | structural element of the building | building-other | 0.8345 | 0.8877 | -0.0532 |
| voc20 | motorbike | motorized bike | bicycle | 0.8912 | 0.9078 | -0.0166 |
| voc21 | motorbike | motorized bike | bicycle | 0.8912 | 0.9078 | -0.0166 |

## Representative Protected Anchors

These are untested text-only proposals. Current protected anchors remain unchanged. Canonical self-similarity is always one, so comparing an alias against its current canonical anchor cannot diagnose whether that anchor itself has the right sense.

| Protocol | Current anchor | Proposed representative | Current centrality | Proposed centrality |
| --- | --- | --- | ---: | ---: |
| ade150 | case | display case | 0.7962 | 0.8813 |
| ade150 | ashcan | trash bin | 0.8275 | 0.9120 |
| ade150 | base | stand | 0.7484 | 0.8319 |
| ade150 | buffet | china closet | 0.7632 | 0.8386 |
| ade150 | earth | soil surface | 0.7824 | 0.8437 |
| ade150 | windowpane | window pane | 0.9092 | 0.9430 |
| ade150 | step | plinth | 0.7301 | 0.7595 |
| ade150 | animal | dog | 0.8574 | 0.8802 |
| ade150 | bag | carrying bag | 0.9458 | 0.9620 |
| ade150 | sink | sink basin | 0.9505 | 0.9647 |
| coco_object81 | mouse | computer mouse | 0.9012 | 0.9273 |
| coco_object81 | frisbee | flying disc | 0.8798 | 0.8966 |
| coco_object81 | hair drier | hair dryer | 0.9570 | 0.9725 |
| coco_object81 | train | railroad train | 0.9224 | 0.9370 |
| coco_object81 | remote | remote control | 0.9540 | 0.9683 |
| coco_object81 | sink | sink basin | 0.9505 | 0.9647 |
| coco_object81 | tv | television screen | 0.9445 | 0.9544 |
| coco_object81 | surfboard | surf board | 0.9681 | 0.9760 |
| coco_object81 | cup | drinking cup | 0.9449 | 0.9524 |
| coco_object81 | toothbrush | tooth brush | 0.9285 | 0.9354 |
| coco_stuff171 | mirror-stuff | wall mirror | 0.8332 | 0.9087 |
| coco_stuff171 | light | lamp | 0.8418 | 0.9013 |
| coco_stuff171 | food-other | meal | 0.8790 | 0.9066 |
| coco_stuff171 | ground-other | outdoor ground | 0.8144 | 0.8417 |
| coco_stuff171 | dirt | bare soil | 0.8862 | 0.9129 |
| coco_stuff171 | mouse | computer mouse | 0.9012 | 0.9273 |
| coco_stuff171 | wall-other | wall | 0.9367 | 0.9608 |
| coco_stuff171 | window-blind | window blinds | 0.9160 | 0.9397 |
| coco_stuff171 | furniture-other | dresser | 0.8307 | 0.8520 |
| coco_stuff171 | branch | tree branches | 0.9146 | 0.9331 |
| voc20 | tvmonitor | tv monitor | 0.9028 | 0.9549 |
| voc20 | train | railroad train | 0.9212 | 0.9421 |
| voc20 | pottedplant | plant in a pot | 0.9062 | 0.9258 |
| voc20 | car | automobile | 0.9281 | 0.9278 |
| voc20 | sheep | domestic sheep | 0.9027 | 0.8915 |
| voc20 | bicycle | pedal bicycle | 0.9080 | 0.8947 |
| voc20 | bus | city bus | 0.9349 | 0.9215 |
| voc20 | diningtable | meal table | 0.9121 | 0.8957 |
| voc20 | horse | domestic horse | 0.9408 | 0.9241 |
| voc20 | sofa | upholstered sofa | 0.9621 | 0.9448 |
| voc21 | tvmonitor | tv monitor | 0.9028 | 0.9549 |
| voc21 | train | railroad train | 0.9212 | 0.9421 |
| voc21 | pottedplant | plant in a pot | 0.9062 | 0.9258 |
| voc21 | car | automobile | 0.9281 | 0.9278 |
| voc21 | bicycle | pedal bicycle | 0.9080 | 0.8947 |
| voc21 | bus | city bus | 0.9349 | 0.9215 |
| voc21 | diningtable | meal table | 0.9280 | 0.9145 |
| voc21 | horse | domestic horse | 0.9408 | 0.9241 |
| voc21 | sofa | upholstered sofa | 0.9621 | 0.9448 |
| voc21 | boat | small boat | 0.9464 | 0.9288 |
