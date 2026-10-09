# Natural wide-view production timing

Three fixed-spread images/domain; one warm pass then three synchronized repeats. Includes encoding/readout/stitch/restoration/CPU argmax, excludes image loading/text/cache/disk. Not full-data latency or matched VIP timing. No count correction or envelope.
Single selected wide-view encoding only; no old+new wide observations inside measured inference. Frozen visual weights/head states checked. Nine fixed-spread real images replay paired-reference probabilities exactly. This is not full9138-image API probability replay, full-dataset mean latency or matched VIP timing.

| Protocol | Old long448 ms | Natural short336/cap672 ms | Delta ms |
| --- | ---: | ---: | ---: |
| voc21 | 199.58 | 181.71 | -17.87 |
| context60 | 223.00 | 235.52 | +12.51 |
| ade150 | 238.54 | 238.62 | +0.08 |
