"""Query-conditioned, reconstructible multi-scale cost decoding.

The decoder operates on CAFe's dense cost tensor rather than proposing object
masks. A learned lifting analysis keeps one coarse approximation and all three
2x2 detail streams at every level. The same local P/U kernels are used for
synthesis, so an inactive update stack reconstructs the input cost exactly up
to floating-point round-off. This is an interface invariant, not a claim that
the complete OVSS model is invertible.
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _groups(channels: int) -> int:
    return math.gcd(channels, 32)


@dataclass(frozen=True)
class QLiftConfig:
    """Architecture choices shared by all controlled Q-Lift arms."""

    arm: str = "query_lift"
    levels: int = 2
    guide_dim: int = 128
    hidden_dim: int = 256
    kernel: int = 3
    query_chunk: int = 8
    tune_visual_blocks: int = 0

    def validate(self) -> None:
        if self.arm not in {"skip", "haar", "image_lift", "query_lift"}:
            raise ValueError("QLift arm must be skip, haar, image_lift, or query_lift")
        if not 1 <= self.levels <= 3:
            raise ValueError("QLift supports one to three analysis levels")
        if self.guide_dim < 16 or self.hidden_dim < 32:
            raise ValueError("QLift guide_dim and hidden_dim are too small")
        if self.kernel < 3 or self.kernel % 2 == 0:
            raise ValueError("QLift kernel must be odd and at least 3")
        if self.query_chunk < 1:
            raise ValueError("QLift query_chunk must be positive")
        if self.tune_visual_blocks not in {0, 2}:
            raise ValueError("QLift tune_visual_blocks must be either 0 or 2")


@dataclass(frozen=True)
class _Pad:
    height: int
    width: int
    pad_height: int
    pad_width: int


@dataclass(frozen=True)
class _LiftCache:
    p: Tensor
    u: Tensor


def _pad_even(value: Tensor) -> tuple[Tensor, _Pad]:
    """Replicate-pad a BCHW state to an even grid, remembering valid extent."""
    height, width = value.shape[-2:]
    pad_height, pad_width = height % 2, width % 2
    if pad_height or pad_width:
        value = F.pad(value, (0, pad_width, 0, pad_height), mode="replicate")
    return value, _Pad(height, width, pad_height, pad_width)


def _crop(value: Tensor, pad: _Pad) -> Tensor:
    return value[..., :pad.height, :pad.width]


def _split_2x2(value: Tensor) -> tuple[Tensor, Tensor]:
    """Return anchor and the three non-anchor parity streams from an even grid."""
    anchor = value[..., 0::2, 0::2]
    details = torch.stack((value[..., 0::2, 1::2], value[..., 1::2, 0::2], value[..., 1::2, 1::2]), dim=1)
    return anchor, details


def _interleave_2x2(anchor: Tensor, details: Tensor) -> Tensor:
    if details.ndim != anchor.ndim + 1 or details.shape[1] != 3:
        raise ValueError("Expected exactly three detail streams")
    batch, channels, height, width = anchor.shape
    output = anchor.new_empty(batch, channels, height * 2, width * 2)
    output[..., 0::2, 0::2] = anchor
    output[..., 0::2, 1::2] = details[:, 0]
    output[..., 1::2, 0::2] = details[:, 1]
    output[..., 1::2, 1::2] = details[:, 2]
    return output


def _masked_softmax(values: Tensor, valid: Tensor, dimension: int) -> Tensor:
    """Float32 masked softmax with a defined all-invalid fallback."""
    minimum = torch.finfo(torch.float32).min
    masked = values.float().masked_fill(~valid, minimum)
    weights = torch.softmax(masked, dim=dimension)
    has_valid = valid.any(dim=dimension, keepdim=True)
    return torch.where(has_valid, weights, torch.zeros_like(weights)).to(values.dtype)


class ImageGuidance(nn.Module):
    """Image-shared guidance built once from DINO's paired X/Y token fields."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(2048, dim, 1),
            nn.GroupNorm(_groups(dim), dim),
            nn.GELU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(_groups(dim), dim),
            nn.GELU(),
        )

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        return self.stem(torch.cat((F.normalize(x, dim=1), F.normalize(y, dim=1)), dim=1))


class ResidualCostUpdate(nn.Module):
    """A zero-start state update; context has a wider local receptive field."""

    def __init__(self, cost_dim: int, guide_dim: int, hidden_dim: int, kernel: int) -> None:
        super().__init__()
        self.state_norm = nn.GroupNorm(_groups(cost_dim), cost_dim)
        self.guide = nn.Conv2d(guide_dim, hidden_dim, 1)
        self.text = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden_dim), nn.GELU())
        self.input = nn.Conv2d(cost_dim + 2 * hidden_dim, hidden_dim, 1)
        self.norm = nn.GroupNorm(_groups(hidden_dim), hidden_dim)
        self.depthwise = nn.Conv2d(hidden_dim, hidden_dim, kernel, padding=kernel // 2, groups=hidden_dim)
        self.expand = nn.Conv2d(hidden_dim, 2 * hidden_dim, 1)
        self.output = nn.Conv2d(2 * hidden_dim, cost_dim, 1)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    def forward(self, state: Tensor, guide: Tensor, text: Tensor) -> Tensor:
        if state.shape[-2:] != guide.shape[-2:] or state.shape[0] != len(text):
            raise ValueError("State, guidance, and query batches must align")
        text_field = self.text(text).to(state.dtype)[:, :, None, None].expand(-1, -1, *state.shape[-2:])
        hidden = self.input(torch.cat((self.state_norm(state), self.guide(guide), text_field), dim=1))
        hidden = self.depthwise(F.gelu(self.norm(hidden)))
        return state + self.output(F.gelu(self.expand(hidden)))


class DynamicLifting(nn.Module):
    """Generate local P/U kernels from image guidance and optionally a query."""

    def __init__(self, guide_dim: int, hidden_dim: int, kernel: int) -> None:
        super().__init__()
        self.kernel = kernel
        self.image = nn.Conv2d(guide_dim, hidden_dim, 1)
        self.image_token = nn.Linear(guide_dim, 1024)
        self.text = nn.Sequential(nn.LayerNorm(1024), nn.Linear(1024, hidden_dim), nn.GELU())
        self.norm = nn.GroupNorm(_groups(hidden_dim), hidden_dim)
        self.hidden = nn.Conv2d(hidden_dim, hidden_dim, 1)
        self.weights = nn.Conv2d(hidden_dim, 6 * kernel * kernel, 1)

    def _weights(self, guide: Tensor, text: Tensor, *, query_conditioned: bool) -> tuple[Tensor, Tensor]:
        token = self.image_token(guide.float().mean((2, 3))).to(guide.dtype)
        if query_conditioned:
            token = token + text
        hidden = self.image(guide) + self.text(token).to(guide.dtype)[:, :, None, None]
        hidden = self.hidden(F.gelu(self.norm(hidden)))
        batch, _, height, width = hidden.shape
        logits = self.weights(F.gelu(hidden)).reshape(batch, 6, self.kernel * self.kernel, height, width)
        valid = F.unfold(hidden.new_ones((1, 1, height, width)), self.kernel, padding=self.kernel // 2)
        valid = valid.reshape(1, self.kernel * self.kernel, height, width).bool()
        prediction = _masked_softmax(logits[:, :3], valid[:, None], 2)
        update_logits = logits[:, 3:].reshape(batch, 3 * self.kernel * self.kernel, height, width)
        update_valid = valid[:, None].expand(1, 3, -1, -1, -1).reshape(1, 3 * self.kernel * self.kernel, height, width)
        update = _masked_softmax(update_logits, update_valid, 1).reshape(batch, 3, self.kernel * self.kernel, height, width)
        return prediction, update

    def analyze(self, value: Tensor, guide: Tensor, text: Tensor, *, query_conditioned: bool) -> tuple[Tensor, Tensor, _LiftCache]:
        anchor, odd = _split_2x2(value)
        prediction_weights, update_weights = self._weights(guide, text, query_conditioned=query_conditioned)
        batch, channels, height, width = anchor.shape
        bank = F.unfold(anchor, self.kernel, padding=self.kernel // 2).reshape(batch, channels, self.kernel * self.kernel, height, width)
        prediction = (bank[:, None] * prediction_weights[:, :, None]).sum(3)
        details = odd - prediction
        detail_bank = F.unfold(details.reshape(batch, 3 * channels, height, width), self.kernel, padding=self.kernel // 2)
        detail_bank = detail_bank.reshape(batch, 3, channels, self.kernel * self.kernel, height, width)
        update = (detail_bank * update_weights[:, :, None]).sum((1, 3))
        return anchor + update, details, _LiftCache(prediction_weights, update_weights)

    def synthesize(self, approximation: Tensor, details: Tensor, cache: _LiftCache) -> Tensor:
        batch, channels, height, width = approximation.shape
        detail_bank = F.unfold(details.reshape(batch, 3 * channels, height, width), self.kernel, padding=self.kernel // 2)
        detail_bank = detail_bank.reshape(batch, 3, channels, self.kernel * self.kernel, height, width)
        update = (detail_bank * cache.u[:, :, None]).sum((1, 3))
        anchor = approximation - update
        bank = F.unfold(anchor, self.kernel, padding=self.kernel // 2).reshape(batch, channels, self.kernel * self.kernel, height, width)
        prediction = (bank[:, None] * cache.p[:, :, None]).sum(3)
        return _interleave_2x2(anchor, details + prediction)


class FixedHaar:
    """Parameter-free 2D Haar analysis/synthesis for the fixed-wavelet control."""

    @staticmethod
    def analyze(value: Tensor) -> tuple[Tensor, Tensor, None]:
        anchor, odd = _split_2x2(value)
        o0, o1, o2 = odd.unbind(1)
        approximation = (anchor + o0 + o1 + o2) * 0.5
        details = torch.stack((
            (anchor + o0 - o1 - o2) * 0.5,
            (anchor - o0 + o1 - o2) * 0.5,
            (anchor - o0 - o1 + o2) * 0.5,
        ), dim=1)
        return approximation, details, None

    @staticmethod
    def synthesize(approximation: Tensor, details: Tensor, cache: None) -> Tensor:
        del cache
        d0, d1, d2 = details.unbind(1)
        anchor = (approximation + d0 + d1 + d2) * 0.5
        o0 = (approximation + d0 - d1 - d2) * 0.5
        o1 = (approximation - d0 + d1 - d2) * 0.5
        o2 = (approximation - d0 - d1 + d2) * 0.5
        return _interleave_2x2(anchor, torch.stack((o0, o1, o2), dim=1))


class QLiftLevel(nn.Module):
    """One analysis level with separate context and detail state updates."""

    def __init__(self, cost_dim: int, config: QLiftConfig) -> None:
        super().__init__()
        self.lifting: DynamicLifting | None = None
        self.context = ResidualCostUpdate(cost_dim, config.guide_dim, config.hidden_dim, kernel=5)
        self.detail = ResidualCostUpdate(cost_dim, config.guide_dim, config.hidden_dim, kernel=3)

    def update_details(self, details: Tensor, guide: Tensor, text: Tensor) -> Tensor:
        batch, streams, channels, height, width = details.shape
        repeated_guide = guide.repeat_interleave(streams, dim=0)
        repeated_text = text.repeat_interleave(streams, dim=0)
        updated = self.detail(details.reshape(batch * streams, channels, height, width), repeated_guide, repeated_text)
        return updated.reshape(batch, streams, channels, height, width)

    @staticmethod
    def modulate_update(original: Tensor, updated: Tensor, prediction_weights: Tensor) -> Tensor:
        """Use the capacity-control P/U field without changing a control split.

        ``skip`` and ``haar`` use this only to modulate their existing state
        update. Their analysis/reconstruction stays standard skip/Haar. The
        learned-lifting arms use the same field both here and in P/U analysis.
        """
        center = prediction_weights.shape[2] // 2
        baseline = 1.0 / prediction_weights.shape[2]
        modulation = (prediction_weights[:, :, center].mean(1, keepdim=True) - baseline)
        gain = 1.0 + 0.25 * torch.tanh(modulation * prediction_weights.shape[2])
        if original.ndim == 5:
            gain = gain[:, None]
        return original + (updated - original) * gain.to(updated.dtype)


class QLiftDecoder(nn.Module):
    """Four controlled evidence organizations over a dense query-cost grid."""

    def __init__(self, cost_dim: int, config: QLiftConfig) -> None:
        super().__init__()
        config.validate()
        self.cost_dim = cost_dim
        self.config = config
        self.guidance = ImageGuidance(config.guide_dim)
        self.levels = nn.ModuleList(QLiftLevel(cost_dim, config) for _ in range(config.levels))
        # All shared modules are constructed before the P/U banks. Every arm
        # owns the same P/U capacity; skip/Haar use it only as a state-update
        # modulator, whereas learned lifting also uses it for analysis/synthesis.
        for level in self.levels:
            level.lifting = DynamicLifting(config.guide_dim, config.hidden_dim, config.kernel)

    def _lift(
        self,
        state: Tensor,
        guide: Tensor,
        text: Tensor,
        *,
        mode: str,
        lifting_text: Tensor,
        lifting_query_conditioned: bool,
        gain_text: Tensor,
        gain_query_conditioned: bool,
        separate_update_gain: bool,
        detail_mode: str,
    ) -> Tensor:
        entries: list[tuple[QLiftLevel, Tensor, _LiftCache | None, _Pad]] = []
        current, current_guide = state, guide
        for level in self.levels:
            padded, padding = _pad_even(current)
            padded_guide, _ = _pad_even(current_guide)
            coarse_guide = F.avg_pool2d(padded_guide, 2)
            assert level.lifting is not None
            if mode == "haar":
                approximation, details, cache = FixedHaar.analyze(padded)
                prediction_weights, _ = level.lifting._weights(
                    coarse_guide, gain_text, query_conditioned=gain_query_conditioned
                )
            else:
                approximation, details, cache = level.lifting.analyze(
                    padded,
                    coarse_guide,
                    lifting_text,
                    query_conditioned=lifting_query_conditioned,
                )
                prediction_weights = (
                    level.lifting._weights(coarse_guide, gain_text, query_conditioned=gain_query_conditioned)[0]
                    if separate_update_gain else cache.p
                )
            approximation = level.modulate_update(
                approximation, level.context(approximation, coarse_guide, text), prediction_weights
            )
            details = level.modulate_update(
                details, level.update_details(details, coarse_guide, text), prediction_weights
            )
            if detail_mode == "zero":
                # Audit-only intervention: preserve the coarse path while
                # removing the updated detail tensor before inverse synthesis.
                details = torch.zeros_like(details)
            entries.append((level, details, cache, padding))
            current, current_guide = approximation, coarse_guide
        for level, details, cache, padding in reversed(entries):
            if mode == "haar":
                current = FixedHaar.synthesize(current, details, cache)
            else:
                assert cache is not None
                assert level.lifting is not None
                current = level.lifting.synthesize(current, details, cache)
            current = _crop(current, padding)
        return current

    def _skip(self, state: Tensor, guide: Tensor, text: Tensor) -> Tensor:
        entries: list[tuple[QLiftLevel, Tensor, Tensor, _Pad, Tensor]] = []
        current, current_guide = state, guide
        for level in self.levels:
            padded, padding = _pad_even(current)
            padded_guide, _ = _pad_even(current_guide)
            coarse = F.avg_pool2d(padded, 2)
            coarse_guide = F.avg_pool2d(padded_guide, 2)
            entries.append((level, padded, coarse, padding, padded_guide))
            assert level.lifting is not None
            prediction_weights, _ = level.lifting._weights(coarse_guide, text, query_conditioned=False)
            current = level.modulate_update(coarse, level.context(coarse, coarse_guide, text), prediction_weights)
            current_guide = coarse_guide
        for level, skip, coarse, padding, fine_guide in reversed(entries):
            reconstructed = skip + F.interpolate(current - coarse, size=skip.shape[-2:], mode="bilinear", align_corners=False)
            current = _crop(level.detail(reconstructed, fine_guide, text), padding)
        return current

    def forward(
        self,
        cost: Tensor,
        x: Tensor,
        y: Tensor,
        text: Tensor,
        *,
        return_trace: bool = False,
        lifting_text: Tensor | None = None,
        lifting_query_conditioned: bool | None = None,
        gain_text: Tensor | None = None,
        gain_query_conditioned: bool | None = None,
        detail_mode: str = "native",
    ) -> tuple[Tensor, dict[str, Tensor]]:
        if cost.ndim != 5 or x.ndim != 4 or y.shape != x.shape or text.ndim != 2 or text.shape[1] != 1024:
            raise ValueError("Unexpected QLift cost, DINO feature, or text shape")
        if lifting_text is None:
            lifting_text = text
        if lifting_text.shape != text.shape:
            raise ValueError("P/U lifting text must match the semantic text tensor")
        separate_update_gain = gain_text is not None or gain_query_conditioned is not None
        if gain_text is None:
            gain_text = lifting_text
        if gain_text.shape != text.shape:
            raise ValueError("Update-gain text must match the semantic text tensor")
        if detail_mode not in {"native", "zero"}:
            raise ValueError("detail_mode must be native or zero")
        if detail_mode != "native" and self.config.arm == "skip":
            raise ValueError("Detail interventions require a lifting or Haar arm")
        if lifting_query_conditioned is None:
            lifting_query_conditioned = self.config.arm == "query_lift"
        if gain_query_conditioned is None:
            gain_query_conditioned = lifting_query_conditioned
        if self.config.arm != "query_lift" and lifting_query_conditioned:
            raise ValueError("Only a query_lift checkpoint may condition P/U on audit text")
        if self.config.arm != "query_lift" and gain_query_conditioned:
            raise ValueError("Only a query_lift checkpoint may condition update gain on audit text")
        batch, channels, queries, height, width = cost.shape
        if channels != self.cost_dim or x.shape != (batch, 1024, height, width):
            raise ValueError("QLift feature and cost grids must share the DINO patch resolution")
        image_guide = self.guidance(x, y)
        pieces, errors = [], []
        for start in range(0, queries, self.config.query_chunk):
            stop = min(queries, start + self.config.query_chunk)
            count = stop - start
            state = cost[:, :, start:stop].permute(0, 2, 1, 3, 4).reshape(batch * count, channels, height, width)
            guide = image_guide[:, None].expand(-1, count, -1, -1, -1).reshape(batch * count, image_guide.shape[1], height, width)
            query = text[start:stop][None].expand(batch, -1, -1).reshape(batch * count, -1)
            lifting_query = lifting_text[start:stop][None].expand(batch, -1, -1).reshape(batch * count, -1)
            gain_query = gain_text[start:stop][None].expand(batch, -1, -1).reshape(batch * count, -1)
            output = (self._skip(state, guide, query) if self.config.arm == "skip" else
                      self._lift(state, guide, query, mode=self.config.arm, lifting_text=lifting_query,
                                 lifting_query_conditioned=bool(lifting_query_conditioned), gain_text=gain_query,
                                 gain_query_conditioned=bool(gain_query_conditioned),
                                 separate_update_gain=separate_update_gain, detail_mode=detail_mode))
            pieces.append(output.reshape(batch, count, channels, height, width).permute(0, 2, 1, 3, 4))
            if return_trace:
                errors.append((output - state).float().abs().amax())
        result = torch.cat(pieces, dim=2)
        trace = {"max_cost_change": torch.stack(errors).amax() if errors else result.new_zeros(())}
        return result, trace
