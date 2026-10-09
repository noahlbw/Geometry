from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


def default_code_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class CheckpointConfig:
    dinov3_repo: Path
    checkpoint_dir: Path
    dinotxt_weights: Path
    lvd_weights: Path
    sat_weights: Path
    bpe_path: Path

    @classmethod
    def from_roots(
        cls,
        code_root: str | Path | None = None,
        checkpoint_dir: str | Path | None = None,
    ) -> "CheckpointConfig":
        root = Path(code_root) if code_root else default_code_root()
        ckpt = Path(checkpoint_dir) if checkpoint_dir else root / "ckpt" / "DINO"
        return cls(
            dinov3_repo=root / "dinov3",
            checkpoint_dir=ckpt,
            dinotxt_weights=ckpt / "dinov3_vitl16_dinotxt_vision_head_and_text_encoder-a442d8f5.pth",
            lvd_weights=ckpt / "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth",
            sat_weights=ckpt / "dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth",
            bpe_path=ckpt / "bpe_simple_vocab_16e6.txt.gz",
        )

    def required(self, use_satellite: bool = False) -> tuple[Path, ...]:
        paths = (
            self.dinov3_repo,
            self.dinotxt_weights,
            self.lvd_weights,
            self.bpe_path,
        )
        return paths + ((self.sat_weights,) if use_satellite else ())


@dataclass(frozen=True)
class TLPConfig:
    smoothing_strength: float = 0.25
    semantic_temperature: float = 0.07
    probability_temperature: float = 0.07
    diagonal_boost: float = 1.0
    image_edge_scale: float = 5.0
    min_confidence: float = 0.05
    structure_temperature: float = 0.15
    cg_max_iterations: int = 50
    cg_tolerance: float = 1e-4

    def validate(self) -> None:
        if self.smoothing_strength < 0:
            raise ValueError("TLP smoothing_strength must be non-negative.")
        if self.semantic_temperature <= 0 or self.probability_temperature <= 0:
            raise ValueError("TLP temperatures must be positive.")
        if self.image_edge_scale < 0:
            raise ValueError("TLP image_edge_scale must be non-negative.")
        if not 0 < self.min_confidence <= 1:
            raise ValueError("TLP min_confidence must be in (0, 1].")
        if self.cg_max_iterations < 1 or self.cg_tolerance <= 0:
            raise ValueError("TLP CG settings must be positive.")


@dataclass(frozen=True)
class GSUPConfig:
    steps: int = 10
    learning_rate: float = 0.05
    neighbors: int = 16
    optimization_size: int = 256
    chunk_pixels: int = 65536
    initial_spatial_scale: float = 1.0
    initial_color_sigma: float = 0.10

    def validate(self) -> None:
        side = int(self.neighbors**0.5)
        if side * side != self.neighbors or side % 2:
            raise ValueError("GSUP neighbors must be an even square, for example 4 or 16.")
        if self.steps < 0 or self.learning_rate <= 0:
            raise ValueError("GSUP steps must be non-negative and learning_rate positive.")
        if self.optimization_size < 32 or self.chunk_pixels < 1:
            raise ValueError("GSUP optimization_size or chunk_pixels is invalid.")
        if self.initial_spatial_scale <= 0 or self.initial_color_sigma <= 0:
            raise ValueError("GSUP initial scales must be positive.")


@dataclass(frozen=True)
class InferenceConfig:
    mode: str = "dinosplat"
    tile_size: int = 512
    overlap: int = 128
    output_temperature: float = 0.07
    confidence_threshold: float | None = None
    use_global_anchor: bool = True
    global_anchor_temperature: float = 0.07
    global_anchor_sigma: float = 0.5
    max_in_memory_mb: int = 2048
    bands: tuple[int, int, int] = (1, 2, 3)
    amp: bool = True

    @property
    def use_tlp(self) -> bool:
        return self.mode in {"tlp", "dinosplat", "dinosplat-sat"}

    @property
    def use_gsup(self) -> bool:
        return self.mode in {"dinosplat", "dinosplat-sat"}

    @property
    def use_satellite(self) -> bool:
        return self.mode == "dinosplat-sat"

    def validate(self, patch_size: int = 16) -> None:
        if self.mode not in {"baseline", "tlp", "dinosplat", "dinosplat-sat"}:
            raise ValueError(f"Unknown inference mode: {self.mode}")
        if self.tile_size < patch_size or self.tile_size % patch_size:
            raise ValueError(f"tile_size must be divisible by patch size {patch_size}.")
        if not 0 <= self.overlap < self.tile_size:
            raise ValueError("overlap must be non-negative and smaller than tile_size.")
        if self.output_temperature <= 0:
            raise ValueError("output_temperature must be positive.")
        if self.confidence_threshold is not None and not 0 <= self.confidence_threshold <= 1:
            raise ValueError("confidence_threshold must be in [0, 1].")
        if self.global_anchor_temperature <= 0 or self.global_anchor_sigma <= 0:
            raise ValueError("global anchor temperature and sigma must be positive.")
        if self.max_in_memory_mb < 64:
            raise ValueError("max_in_memory_mb must be at least 64.")
        if len(set(self.bands)) != 3 or min(self.bands) < 1:
            raise ValueError("bands must contain three distinct 1-based band indices.")


def dataclass_dict(value: Any) -> dict[str, Any]:
    result = asdict(value)
    return {key: str(item) if isinstance(item, Path) else item for key, item in result.items()}
