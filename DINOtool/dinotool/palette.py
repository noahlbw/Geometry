from __future__ import annotations

import colorsys
from typing import Sequence

import numpy as np
from PIL import Image


BASE_COLORS: tuple[tuple[int, int, int], ...] = (
    (48, 48, 48),
    (39, 125, 161),
    (239, 103, 72),
    (59, 158, 97),
    (246, 190, 63),
    (144, 98, 180),
    (229, 130, 178),
    (101, 101, 101),
    (61, 183, 195),
    (177, 89, 40),
    (88, 140, 55),
    (209, 77, 98),
    (84, 103, 171),
    (165, 122, 79),
    (141, 199, 63),
    (196, 156, 148),
)


def make_palette(count: int) -> list[tuple[int, int, int]]:
    colors = list(BASE_COLORS[:count])
    index = len(colors)
    while len(colors) < count:
        hue = (index * 0.618033988749895) % 1.0
        red, green, blue = colorsys.hsv_to_rgb(hue, 0.62, 0.88)
        colors.append((round(red * 255), round(green * 255), round(blue * 255)))
        index += 1
    return colors


def colorize(labels: np.ndarray, colors: Sequence[tuple[int, int, int]]) -> np.ndarray:
    result = np.zeros((*labels.shape, 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        result[labels == index] = color
    return result


def indexed_image(labels: np.ndarray, colors: Sequence[tuple[int, int, int]]) -> Image.Image:
    image = Image.fromarray(labels.astype(np.uint8), mode="P")
    flattened = [component for color in colors for component in color]
    flattened.extend([0] * (768 - len(flattened)))
    image.putpalette(flattened)
    image.info["transparency"] = 255
    return image

