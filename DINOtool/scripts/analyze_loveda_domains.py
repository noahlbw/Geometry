from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image


CLASS_NAMES = ("background", "building", "road", "water", "barren", "forest", "agricultural")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze saved LoveDA predictions by Rural/Urban domain.")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--predictions-root", required=True)
    parser.add_argument("--modes", nargs="+", default=("baseline", "tlp", "dinosplat"))
    args = parser.parse_args()
    root = Path(args.data_root).expanduser().resolve()
    prediction_root = Path(args.predictions_root).expanduser().resolve()
    matrices = {mode: {} for mode in args.modes}
    all_matrices = {mode: np.zeros((7, 7), dtype=np.int64) for mode in args.modes}
    samples = 0
    for image_dir in sorted(root.rglob("images_png")):
        mask_dir = image_dir.parent / "masks_png"
        if not mask_dir.is_dir():
            continue
        domain = "rural" if "rural" in {part.casefold() for part in image_dir.parts} else "urban"
        matrix_by_mode = {
            mode: matrices[mode].setdefault(domain, np.zeros((7, 7), dtype=np.int64)) for mode in args.modes
        }
        for mask_path in sorted(mask_dir.glob("*.png")):
            key = (image_dir.parent.relative_to(root) / mask_path.name).as_posix()
            target = np.asarray(Image.open(mask_path), dtype=np.uint8).astype(np.int64, copy=False)
            valid = (target >= 1) & (target <= 7)
            if not bool(valid.any()):
                continue
            target = target[valid] - 1
            samples += 1
            for mode in args.modes:
                prediction_path = prediction_root / mode / key
                if not prediction_path.is_file():
                    raise FileNotFoundError(prediction_path)
                prediction = np.asarray(Image.open(prediction_path), dtype=np.uint8).astype(np.int64, copy=False)[valid]
                if prediction.size and (prediction.min() < 0 or prediction.max() >= 7):
                    raise ValueError(f"Invalid prediction IDs in {prediction_path}")
                encoded = target * 7 + prediction
                counts = np.bincount(encoded, minlength=49).reshape(7, 7)
                matrix_by_mode[mode] += counts
                all_matrices[mode] += counts
    print(f"samples={samples}")
    for mode in args.modes:
        print(f"[{mode}]")
        _print_metrics("all", all_matrices[mode])
        for domain, matrix in matrices[mode].items():
            _print_metrics(domain, matrix)


def _print_metrics(name: str, matrix: np.ndarray) -> None:
    diagonal = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - diagonal
    iou = np.divide(diagonal, union, out=np.full(7, np.nan), where=union > 0)
    accuracy = diagonal.sum() / max(matrix.sum(), 1)
    values = ", ".join(f"{class_name}={score * 100:.2f}" for class_name, score in zip(CLASS_NAMES, iou))
    print(f"  {name}: mIoU={np.nanmean(iou) * 100:.3f}, acc={accuracy * 100:.3f}; {values}")


if __name__ == "__main__":
    main()
