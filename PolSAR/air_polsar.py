"""
AIR-PolarSAR-Seg tile loader — turns the 4-band amplitude tiffs + color-coded
ground-truth PNGs in test/train_set into the (H, W, k, k) covariance format
the PolSAR forest consumes, for MULTI-IMAGE training (each tile is a
different part of the scene).

The dataset ships amplitude-only bands (HH, HV, VH, VV as uint16), not
single-look complex data, so a true polarimetric covariance is not
recoverable. We build the standard pragmatic substitute: the outer product
of the 4-channel amplitude vector, boxcar-averaged over a small window
(full-rank after averaging) plus a small identity regularization so every
matrix is strictly positive-definite (logm/inv/det in distances.py need
that). All distances operate on real symmetric PD matrices just as well as
on complex Hermitian ones.

Typical usage (multiple tiles as one training pool):

    from air_polsar import load_air_tiles, AIR_CLASS_NAMES, AIR_IGNORE
    from example import build_reference_pool, sample_training_patches

    images, labels = load_air_tiles("../test/train_set", indices=[1, 55, 100])
    pool = build_reference_pool(images, n=200, rng=rng)
    patches, patch_labels = sample_training_patches(
        images, labels, n=2000, patch_size=9, rng=rng,
        ignore_label=AIR_IGNORE,
        balanced=True,   # tiles are dominated by "natural" — equalize classes
    )
"""

from __future__ import annotations

import os

import numpy as np
from scipy.ndimage import uniform_filter
from skimage import io

# Class palette of the AIR-PolarSAR-Seg ground truth (color-coded PNGs).
# Black pixels are unlabeled -> AIR_IGNORE. Adjust here if your copy of the
# dataset uses a different legend.
AIR_CLASSES: list[tuple[str, tuple[int, int, int]]] = [
    ("housing",    (255, 255,   0)),
    ("industrial", (255,   0,   0)),
    ("natural",    (  0, 255,   0)),
    ("land_use",   (  0, 255, 255)),
    ("water",      (  0,   0, 255)),
    ("other",      (255, 255, 255)),
]
AIR_CLASS_NAMES = [name for name, _ in AIR_CLASSES]
AIR_IGNORE = 255  # label for unlabeled (black) ground-truth pixels


def decode_air_gt(gt_rgb: np.ndarray) -> np.ndarray:
    """Color-coded (H, W, 3) ground truth -> (H, W) uint8 class indices.

    Unknown colors (including black) map to AIR_IGNORE.
    """
    labels = np.full(gt_rgb.shape[:2], AIR_IGNORE, dtype=np.uint8)
    for idx, (_, color) in enumerate(AIR_CLASSES):
        labels[np.all(gt_rgb[..., :3] == color, axis=-1)] = idx
    return labels


def bands_to_covariance(
    bands: np.ndarray,
    boxcar: int = 3,
    scale: float = 1000.0,
    eps: float = 1e-3,
) -> np.ndarray:
    """(4, H, W) amplitude bands -> (H, W, 4, 4) real SPD pseudo-covariances.

    C(r, c) = boxcar-mean of s s^T with s = bands / scale, plus
    eps * mean-span * I. The boxcar makes the rank-1 single-pixel outer
    products full-rank; the identity term guarantees strict PD for logm.
    """
    s = bands.astype(np.float64) / scale
    k, H, W = s.shape

    C = np.empty((H, W, k, k), dtype=np.float64)
    for i in range(k):
        for j in range(i, k):
            avg = uniform_filter(s[i] * s[j], size=boxcar, mode="nearest")
            C[:, :, i, j] = avg
            if j != i:
                C[:, :, j, i] = avg

    mean_span = float(np.trace(C, axis1=2, axis2=3).mean())
    C += np.eye(k) * (eps * max(mean_span, 1e-12))
    return C


def load_air_tile(
    train_dir: str,
    index: int,
    boxcar: int = 3,
    scale: float = 1000.0,
    eps: float = 1e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """Load one AIR-PolarSAR-Seg tile as (image, labels).

    image:  (H, W, 4, 4) float64 SPD covariance stack
    labels: (H, W) uint8 class indices, AIR_IGNORE where unlabeled
    """
    import rasterio as rs

    stem = os.path.join(train_dir, f"AIR-PolarSAR-Seg-{index}")
    combined = f"{stem}_Combined.tiff"
    if os.path.exists(combined):
        with rs.open(combined) as t:
            bands = t.read()  # (4, H, W)
    else:
        band_data = []
        for pol in ("HH", "HV", "VH", "VV"):
            with rs.open(f"{stem}_{pol}.tiff") as t:
                band_data.append(t.read(1))
        bands = np.stack(band_data)

    labels = decode_air_gt(io.imread(f"{stem}_gt.png"))
    return bands_to_covariance(bands, boxcar=boxcar, scale=scale, eps=eps), labels


def load_air_tiles(
    train_dir: str,
    indices: list,
    boxcar: int = 3,
    scale: float = 1000.0,
    eps: float = 1e-3,
) -> tuple[list, list]:
    """Load several tiles -> (images, labels) lists, ready for
    build_reference_pool / sample_training_patches."""
    images, labels = [], []
    for i in indices:
        img, lbl = load_air_tile(train_dir, i, boxcar=boxcar, scale=scale, eps=eps)
        images.append(img)
        labels.append(lbl)
    return images, labels
