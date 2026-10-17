"""
End-to-end example: train and evaluate the PolSAR RF on synthetic data.

For real data:
  - Load your PolSAR image as a (H, W, k, k) complex numpy array of
    covariance/coherency matrices (e.g. from PolSARpro .bin files).
  - Replace `make_synthetic_image` with your loader.
  - Replace the label array with your ground-truth mask.
"""

import numpy as np
import sys
import os

# Allow imports from the same directory
sys.path.insert(0, os.path.dirname(__file__))

from distances import ALL_DISTANCES
from node_tests import NodeTestSampler
from polsar_rf import PolSARRandomForest
import polars as pl

# ---------------------------------------------------------------------------
# Synthetic data generator (placeholder — swap with real PolSAR loader)
# ---------------------------------------------------------------------------
def load_image(path,cats) -> tuple[np.ndarray, np.ndarray]:
    file_list = os.listdir(path)
    valid_img = []
    for fname in file_list:
        if any(fname.startswith(cat) for cat in cats):
            valid_img.append(fname)

    return image, labels
def make_synthetic_image(
    height: int = 64,
    width: int = 64,
    k: int = 3,
    n_classes: int = 5,
    rng: np.random.Generator = None,
    class_means: list | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Create a tiny synthetic PolSAR image and a label mask.

    Each class gets a random mean covariance matrix.  Pixels are drawn from
    a complex-Gaussian with that covariance (Wishart noise around the mean).

    class_means: optional list of k×k SPD matrices, one per class. Pass the
    SAME list to several calls to generate multiple scenes that share one
    class set (multi-image training) — otherwise each image would invent its
    own, semantically unrelated classes.

    Returns
    -------
    image : ndarray, shape (H, W, k, k), complex128
        Stack of Hermitian positive-definite covariance matrices.
    labels : ndarray, shape (H, W), int
        Class label per pixel.
    """
    rng = rng or np.random.default_rng(0)

    if class_means is None:
        # One random SPD mean per class
        class_means = []
        for _ in range(n_classes):
            A = rng.standard_normal((k, k)) + 1j * rng.standard_normal((k, k))
            class_means.append((A @ A.conj().T) / k + np.eye(k) * 0.5)

    # Assign each pixel a class (simple 5-band vertical stripes)
    labels = (np.arange(height * width).reshape(height, width) * n_classes // (height * width)).astype(int)
    labels = np.clip(labels, 0, n_classes - 1)

    # Sample noisy covariance per pixel
    image = np.zeros((height, width, k, k), dtype=complex)
    for c in range(n_classes):
        mask = labels == c
        n_pix = mask.sum()
        mean = class_means[c]
        # Add noise: random Hermitian perturbation
        for i, (r, col) in enumerate(zip(*np.where(mask))):
            noise = rng.standard_normal((k, k)) + 1j * rng.standard_normal((k, k))
            noise = (noise @ noise.conj().T) * 0.1
            image[r, col] = mean + noise

    return image, labels


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def extract_patch(image: np.ndarray, r: int, c: int, half: int) -> np.ndarray:
    """Extract a (2*half+1, 2*half+1, k, k) patch centred at (r, c)k."""
    return image[r - half:r + half + 1, c - half:c + half + 1]


def _as_image_list(images) -> list:
    """One (H, W, k, k) ndarray → [ndarray]; a list/tuple passes through."""
    if isinstance(images, np.ndarray):
        return [images]
    return list(images)


def build_reference_pool(images, n: int = 500, rng=None, per_image: bool = False) -> list:
    """
    Sample n random covariance matrices for 1-pt projections.

    images : one (H, W, k, k) ndarray, or a list of them (different scenes /
             parts of a city or landscape; sizes may differ).
    n      : total pool size, across all images.
    per_image : False (default) — pixels are drawn proportionally to each
             image's pixel count, i.e. uniformly over the union of all pixels.
             True — an equal quota per image, so small scenes contribute as
             many references as large ones (more diversity, less fidelity to
             the overall pixel distribution).
    """
    rng = rng or np.random.default_rng(42)
    images = _as_image_list(images)
    if not images:
        return []

    sizes = np.array([img.shape[0] * img.shape[1] for img in images], dtype=float)
    if per_image:
        quotas = np.full(len(images), n // len(images), dtype=int)
        quotas[: n % len(images)] += 1
    else:
        # Multinomial draw ≙ sampling each of the n pool entries uniformly
        # over the union of all pixels.
        quotas = rng.multinomial(n, sizes / sizes.sum())

    pool = []
    for img, quota in zip(images, quotas):
        H, W = img.shape[:2]
        rows = rng.integers(0, H, size=quota)
        cols = rng.integers(0, W, size=quota)
        pool.extend(img[r, c] for r, c in zip(rows, cols))
    return pool


def sample_training_patches(
    images,
    labels,
    n: int,
    patch_size: int,
    rng=None,
    ignore_label: int | None = None,
    balanced: bool = False,
) -> tuple[list, np.ndarray]:
    """
    Sample n labeled training patches across one or several images.

    Each image is edge-padded once, pixel locations are drawn uniformly over
    the union of all images' pixels (so scenes contribute proportionally to
    their size), and the result is shuffled — consecutive partial_fit batches
    therefore mix scenes instead of exhausting one image before the next.

    images : one (H, W, k, k) ndarray or a list of them.
    labels : the matching (H, W) int label mask, or list of masks.
    ignore_label : label value to skip (e.g. 255 for unlabeled ground-truth
             pixels); sampling retries elsewhere so n patches are returned
             as long as any labeled pixels exist.
    balanced : False (default) — sample uniformly over labeled pixels, so the
             stream mirrors the true class distribution. True — split n into
             equal per-class quotas over the classes observed anywhere in the
             masks; rare classes are oversampled with replacement. Use this
             when a dominant class (e.g. AIR-PolarSAR-Seg's "natural") would
             otherwise drown out the others.

    Returns (patches, labels): list of (patch_size, patch_size, k, k) arrays
    and the aligned (n,) int label vector.
    """
    rng = rng or np.random.default_rng(42)
    images = _as_image_list(images)
    label_masks = [labels] if isinstance(labels, np.ndarray) and images and labels.ndim == 2 else list(labels)
    if len(images) != len(label_masks):
        raise ValueError(f"{len(images)} images but {len(label_masks)} label masks")

    half = patch_size // 2
    # Pad the two spatial axes and leave every trailing axis alone, whatever
    # their number: (H, W, k, k) PolSAR covariances, (H, W, 3) RGB, or a plain
    # (H, W) mask. Equivalent to the old hard-coded 4-D pad for PolSAR input.
    padded = [
        np.pad(img, [(half, half), (half, half)] + [(0, 0)] * (img.ndim - 2), mode="edge")
        for img in images
    ]

    # Sampling groups: each is (per-image coordinate arrays, total quota).
    # Unbalanced = one group over all labeled pixels; balanced = one group per
    # observed class with an equal share of n.
    groups: list[tuple[list, int]] = []
    if balanced:
        observed: set = set()
        for lbl in label_masks:
            observed |= set(np.unique(lbl).tolist())
        observed.discard(ignore_label)
        classes = sorted(observed)
        if not classes:
            raise ValueError("no labeled pixels in any image")
        base, rem = divmod(n, len(classes))
        for i, c in enumerate(classes):
            coords = [np.where(lbl == c) for lbl in label_masks]
            groups.append((coords, base + (1 if i < rem else 0)))
    else:
        coords = []
        for lbl in label_masks:
            if ignore_label is not None:
                coords.append(np.where(lbl != ignore_label))
            else:
                rr, cc = np.indices(lbl.shape).reshape(2, -1)
                coords.append((rr, cc))
        groups.append((coords, n))

    patches, out_labels = [], []
    for coords, group_n in groups:
        counts = np.array([len(rr) for rr, _ in coords], dtype=float)
        if counts.sum() == 0:
            raise ValueError("no labeled pixels in any image")
        # Multinomial over images ≙ drawing each patch uniformly over the
        # group's pixels across all images.
        quotas = rng.multinomial(group_n, counts / counts.sum())
        for pad_img, lbl, (rr, cc), quota in zip(padded, label_masks, coords, quotas):
            if quota == 0 or len(rr) == 0:
                continue
            pick = rng.integers(0, len(rr), size=quota)
            for r, c in zip(rr[pick], cc[pick]):
                patches.append(pad_img[r:r + patch_size, c:c + patch_size])
                out_labels.append(int(lbl[r, c]))

    order = rng.permutation(len(patches))
    return [patches[i] for i in order], np.array(out_labels)[order]


def balanced_accuracy(predictions: np.ndarray, labels: np.ndarray, n_classes: int) -> float:
    """Average per-class detection rate (balanced accuracy)."""
    per_class = []
    for c in range(n_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        per_class.append((predictions[mask] == c).mean())
    return float(np.mean(per_class))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    rng = np.random.default_rng(0)

    print("=" * 60)
    print("PolSAR RF — skipping explicit feature extraction")
    print("=" * 60)

    # ---- 1. Data ----------------------------------------------------------
    print("\n[1/4] Generating synthetic PolSAR image …")
    H, W, k, n_classes = 64, 64, 3, 5
    image, labels = make_synthetic_image(H, W, k, n_classes, rng)
    print(f"      Image: {image.shape}, Classes: {n_classes}")

    # Train / test split: top half trains, bottom half tests
    half_h = H // 2
    train_img = image[:half_h]
    train_lbl = labels[:half_h]
    test_img  = image[half_h:]
    test_lbl  = labels[half_h:]

    # ---- 2. Build training patches ----------------------------------------
    print("\n[2/4] Extracting training patches …")
    patch_size = 11          # spatial size of each patch
    half = patch_size // 2
    train_patches = []
    train_labels_flat = []

    # Pad image so border pixels can also be classified
    padded_train = np.pad(
        train_img,
        ((half, half), (half, half), (0, 0), (0, 0)),
        mode="edge",
    )

    for r in range(train_img.shape[0]):
        for c in range(train_img.shape[1]):
            p = padded_train[r:r + patch_size, c:c + patch_size]
            train_patches.append(p)
            train_labels_flat.append(train_lbl[r, c])

    train_labels_flat = np.array(train_labels_flat)
    print(f"      {len(train_patches)} training patches of shape {train_patches[0].shape}")

    # ---- 3. Build reference pool for 1-pt projections ---------------------
    ref_pool = build_reference_pool(train_img, n=200, rng=rng)

    # ---- 4. Construct sampler + forest ------------------------------------
    print("\n[3/4] Building Random Forest …")
    sampler = NodeTestSampler(
        patch_size=patch_size,
        region_size_min=2,
        region_size_max=4,
        max_offset=4,
        reference_pool=ref_pool,
        distance_names=list(ALL_DISTANCES.keys()),   # use all distances
        # --- OR restrict to the paper's best performer: ---
        # distance_names=["log_euclid"],
    )

    rf = PolSARRandomForest(
        sampler=sampler,
        n_trees=5,           # use 30 for real experiments (paper setting)
        n_candidates=20,     # use 50–200 for real experiments
        max_depth=10,        # use 50 for real experiments
        min_samples_split=5,
        n_classes=n_classes,
        random_state=42,
    )
    rf.fit(train_patches, train_labels_flat)

    # ---- 5. Classify test image -------------------------------------------
    print("\n[4/4] Classifying test image …")
    padded_test = np.pad(
        test_img,
        ((half, half), (half, half), (0, 0), (0, 0)),
        mode="edge",
    )
    test_H, test_W = test_img.shape[:2]
    predictions = np.zeros((test_H, test_W), dtype=int)

    for r in range(test_H):
        for c in range(test_W):
            patch = padded_test[r:r + patch_size, c:c + patch_size]
            predictions[r, c] = rf.predict(patch)
        print(f"  Row {r + 1}/{test_H}", end="\r")
    print()

    ba = balanced_accuracy(predictions, test_lbl, n_classes)
    print(f"\nBalanced accuracy on test set: {ba * 100:.1f}%")
    print("(Synthetic data — expect ~20% random baseline with 5 classes)")


if __name__ == "__main__":
    main()
