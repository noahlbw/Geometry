from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RasterMetadata:
    path: Path
    width: int
    height: int
    is_geotiff: bool
    transform: object | None = None
    crs: object | None = None


class RasterSource:
    """Windowed RGB access with one global radiometric stretch per raster."""

    def __init__(self, path: str | Path, bands: tuple[int, int, int] = (1, 2, 3)) -> None:
        self.path = Path(path)
        if not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.bands = bands
        self._dataset = None
        self._image = None
        self._ranges: tuple[tuple[float, float], ...] | None = None
        if self.path.suffix.lower() in {".tif", ".tiff"}:
            import rasterio

            self._dataset = rasterio.open(self.path)
            if self._dataset.count < 3 or max(bands) > self._dataset.count:
                self.close()
                raise ValueError(f"{self.path} does not contain requested RGB bands {bands}.")
            self.metadata = RasterMetadata(
                path=self.path,
                width=self._dataset.width,
                height=self._dataset.height,
                is_geotiff=True,
                transform=self._dataset.transform,
                crs=self._dataset.crs,
            )
            if any(self._dataset.dtypes[index - 1] != "uint8" for index in bands):
                self._ranges = self._estimate_ranges()
        else:
            self._image = Image.open(self.path).convert("RGB")
            self.metadata = RasterMetadata(
                path=self.path,
                width=self._image.width,
                height=self._image.height,
                is_geotiff=False,
            )

    def read_window(self, left: int, top: int, width: int, height: int) -> Image.Image:
        if width < 1 or height < 1:
            raise ValueError("Raster windows must have positive dimensions.")
        if self._dataset is None:
            assert self._image is not None
            return self._image.crop((left, top, left + width, top + height)).convert("RGB")
        from rasterio.windows import Window

        array = self._dataset.read(self.bands, window=Window(left, top, width, height))
        return Image.fromarray(np.moveaxis(self._to_uint8(array), 0, -1), mode="RGB")

    def read_preview(self, maximum: int = 1600) -> Image.Image:
        scale = min(1.0, maximum / max(self.metadata.width, self.metadata.height))
        target = (max(1, round(self.metadata.width * scale)), max(1, round(self.metadata.height * scale)))
        if self._dataset is None:
            assert self._image is not None
            return self._image.resize(target, Image.Resampling.BILINEAR)
        from rasterio.enums import Resampling

        array = self._dataset.read(self.bands, out_shape=(3, target[1], target[0]), resampling=Resampling.bilinear)
        return Image.fromarray(np.moveaxis(self._to_uint8(array), 0, -1), mode="RGB")

    def close(self) -> None:
        if self._dataset is not None:
            self._dataset.close()
            self._dataset = None
        if self._image is not None:
            self._image.close()
            self._image = None

    def __enter__(self) -> "RasterSource":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def _estimate_ranges(self) -> tuple[tuple[float, float], ...]:
        from rasterio.enums import Resampling

        sample_scale = min(1.0, 1024 / max(self.metadata.width, self.metadata.height))
        sample_h = max(1, round(self.metadata.height * sample_scale))
        sample_w = max(1, round(self.metadata.width * sample_scale))
        sample = self._dataset.read(
            self.bands,
            out_shape=(3, sample_h, sample_w),
            resampling=Resampling.bilinear,
            masked=True,
        ).astype(np.float32)
        ranges: list[tuple[float, float]] = []
        for band in sample:
            values = band.compressed() if np.ma.isMaskedArray(band) else band[np.isfinite(band)]
            if not values.size:
                ranges.append((0.0, 1.0))
                continue
            low, high = np.percentile(values, (2, 98))
            if not np.isfinite(low) or not np.isfinite(high) or high <= low:
                low, high = float(np.nanmin(values)), float(np.nanmax(values))
            if high <= low:
                high = low + 1.0
            ranges.append((float(low), float(high)))
        return tuple(ranges)

    def _to_uint8(self, array: np.ndarray) -> np.ndarray:
        if array.dtype == np.uint8 and self._ranges is None:
            return array
        result = np.empty(array.shape, dtype=np.uint8)
        ranges = self._ranges
        if ranges is None:
            ranges = tuple((float(np.nanmin(band)), float(np.nanmax(band))) for band in array)
        for index, (band, (low, high)) in enumerate(zip(array.astype(np.float32), ranges)):
            result[index] = np.clip((band - low) * 255.0 / max(high - low, 1e-6), 0, 255).astype(np.uint8)
        return result


def write_geotiff(
    path: str | Path,
    array: np.ndarray,
    metadata: RasterMetadata,
    nodata: int | float | None = None,
) -> None:
    if not metadata.is_geotiff:
        return
    import rasterio

    path = Path(path)
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=metadata.width,
        height=metadata.height,
        count=1,
        dtype=array.dtype,
        transform=metadata.transform,
        crs=metadata.crs,
        compress="deflate",
        predictor=2,
        tiled=True,
        blockxsize=256,
        blockysize=256,
        nodata=nodata,
    ) as destination:
        destination.write(array, 1)

