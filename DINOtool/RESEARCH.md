# Method selection

Research date: 2026-08-21.

## Selected method

The default implementation follows **DinoSplat-OV**, "Standalone DINOv3 for
Training-Free Open-Vocabulary Semantic Segmentation in Remote Sensing"
([arXiv:2608.03023](https://arxiv.org/abs/2608.03023), 2026-08-04).

It is the closest fit to this repository because it is:

- DINOv3-only for semantic recognition through the official dino.txt head;
- explicitly designed for remote-sensing imagery;
- training-free, including its high-resolution reconstruction stage;
- compatible with the local ViT-L/16 LVD-1689M and dino.txt checkpoints.

The implementation covers the paper's four inference components: remote-sensing
prompt/synonym aggregation, Text-aware Laplacian Propagation (TLP), Gaussian
Splatting Upsampling (GSUP), and global-anchor sliding-window blending.

## Fast deployment path

`infer-dino-dense` and `--serve-dino-dense` are a separate latency-oriented
ablation. They load only the frozen DINOv3 dino.txt model, run one resized image
forward, batch and cache the requested text prototypes, and upsample one dense
class-cost map. The default 512-pixel long side produces a 32x32 ViT patch grid
before bilinear output upsampling for an 800x800 image. It is training-free and
does not load SAM3, TLP, GSUP, SAT, or per-class prompt decoders. Its purpose is
agent responsiveness; quality comparisons should use the full DinoSplat-OV or
SAM3+DINO paths. An external CAFe cost-aggregation checkpoint is accepted as an
opt-in quality experiment, with an explicit latency trade-off.

## Alternatives reviewed

- [DINOv3](https://github.com/facebookresearch/dinov3) and its official
  `dinotxt_segmentation_inference.ipynb` provide the reliable baseline and local
  loading API. The SAT-493M model is used only by the explicitly experimental
  `dinosplat-sat` mode because its text head was released for the LVD backbone.
- [GeoSeg-OV](https://arxiv.org/abs/2608.10426) reports stronger trained remote-
  sensing results and released code, but depends on learned cost aggregation and
  decoder components. It is not the preferred training-free deployment route.
- [Pi-Seg / OVRSISBenchV2](https://arxiv.org/abs/2604.15652) is a strong trained
  CAT-Seg baseline with remote-sensing checkpoints. It is useful for benchmark
  comparison, but requires its learned perturbation and decoder weights.
- [SegEarth-OV](https://arxiv.org/abs/2410.01768) is a mature training-free
  remote-sensing system, but its pipeline is CLIP-based and relies on a pretrained
  feature upsampler.
- [CLIP-DINOiser](https://arxiv.org/abs/2403.14539) and LPOSS use DINO as an
  auxiliary spatial prior for CLIP. They do not satisfy the standalone-DINO goal.
- Seg-Probe, described in [arXiv:2608.09101](https://arxiv.org/abs/2608.09101),
  is a newer training-free probe built on SegEarth-OV3/SAM 3. Its released claim
  targets mask auditing, and the required SAM 3 stack is not present locally.

## Reimplementation boundary

No official DinoSplat-OV code was public in GitHub search on 2026-08-21. The
paper also omits the TLP smoothing strength, semantic diagonal boost, GSUP
optimizer/lr, the full-image anchor resize, and the global similarity
temperature. It does specify a ViT CLS token for each window, which this
implementation takes from DINO.txt's vision head. Unspecified values are
exposed as CLI configuration where applicable and recorded in `metadata.json`;
the defaults are implementation choices, not claimed author settings.

The paper says GSUP upsamples features. This implementation upsamples class
logits after the text dot product. GSUP is a fixed linear weighted sum once its
per-image Gaussian parameters are fitted, so the operations commute exactly:
`sum(w * feature) dot text == sum(w * (feature dot text))`. This avoids allocating
a full-resolution 1024-channel feature tensor.
