# Bounded contrast coupling: fixed finite mechanism study

The final eight-domain accuracy/latency objective remains unachieved. This is a
mechanism test, not a final model nomination or an independent validation set.
The existing bounded896 suite, all outputs and model rules remain unchanged.

## Hypothesis

The unscreened coupled reconstruction is `L + H(B-L)`, with the original
Geometry operator `H = (I + G^T G)^-1 G^T G`. It transports both spatial broad
innovation and window-wide class offsets. Those offsets can change competition
even without useful spatial evidence. The OEM Geometry-to-coupling drop and
car/vehicle precision deficits motivate separating those components; they do not
prove that the offsets are harmful on every domain.

For valid patch indicator `v`, define `P X` as subtracting the mean of `X` over
valid patches and zeroing invalid rows. Primary: `L + P H P(B-L)`. This ignores
uniform broad class offsets and has exactly zero valid-patch mean correction.
It preserves the intermediate local mean, NOT necessarily final class area or
mIoU after interpolation, softmax, stitching and argmax.

## Fixed Arms And Protocol

- Original Geometry local-only and original no-admission coupled endpoints.
- Primary Geometry contrast coupling with no added parameter.
- Published SCLIP_Two local readout with original H, coupled and contrast-coupled.
  These are head controls, not a claim of our innovation or official SCLIP scores.
- Geometry long edge896, four or fewer real512 crops/stride384; wide original RGB
  independently resized to448, four or fewer336 crops/stride112. No extra backbone
  observation, fine encoding, native-resolution window or per-domain routing.
- Same frozen checkpoints, Geometry depth2, fixed20 alias pools/class, local RS
  and wide ImageNet text, probability stitching and original-size restoration.
- No alias weighting in these arms. No threshold, template or hyperparameter fit.
- Full VDD80, Potsdam504, UDD540 and OEM384:1008 unique images. Only after mask-free
  full-image smoke equivalence and singleton/control identity. Do not run a new
  eight-domain suite unless full results provide relevant improvement.
- Masks are loaded only after predictions. All four domains were developed on
  previously; this is exploratory development, not untouched independent evidence.

## Evidence Required

Check original endpoint full confusion equality and identities versus frozen
bounded896 outputs, complete unique coverage, per-image confusion sums, constant
offset invariance, zero mean correction, valid padding, actual encoding budgets,
frozen parameters and singleton equivalence. Compare full mIoU and car/vehicle
class outcomes with both same20 VIP and stronger official-query VIP.

Standalone timing is warmed synchronized complete-image singleton inference on
the same fixed first/middle/last images, with loading and masks excluded. No
extra GPU worker is permitted during a new timing panel. Existing suite timing
finishes before the four new full-domain workers launch.
