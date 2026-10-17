"""
Scene-classification datasets (AID / UC-Merced) and Marée-style patch sampling.

These datasets label a whole tile, not a pixel: the class is encoded in the
filename. That makes them immediately usable with no ground-truth masks, and it
is ideal for the visualization goal — every reference patch pushed to TreeFlow is a
recognizable aerial scene rather than an anonymous pixel neighbourhood.

Patch sampling follows

    Marée, Geurts, Piater & Wehenkel, "Random Subwindows for Robust Image
    Classification", CVPR 2005

— random subwindows of random size, each resized to a fixed patch size. That
gives many samples per tile plus scale robustness, and (because the windows are
biased large) the resulting patches stay readable.
"""

from __future__ import annotations

import os
import re
import warnings
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from rgb_patch import PreparedPatch

# ---------------------------------------------------------------------------
# Dataset registry
# ---------------------------------------------------------------------------

# Both patterns verified against every file in the two directories, no outliers.
# AID is anchored on "_<digits>_adv" rather than "strip trailing digits" so that
# run-together class names survive intact: denseresidential, storagetanks,
# railwaystation, baseballfield, ...
AID_RE = re.compile(r"^(?P<cls>[A-Za-z]+)_\d+_adv\.png$", re.IGNORECASE)
# UC-Merced numbers each class with exactly two digits: agricultural01_adv.png
UCM_RE = re.compile(r"^(?P<cls>[A-Za-z]+)\d{2}_adv\.png$", re.IGNORECASE)

# UAE-RS location: $UAE_RS_ROOT if set, else a "UAE-RS" folder next to the repo.
UAE_RS_ROOT = os.environ.get("UAE_RS_ROOT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "UAE-RS")

DATASETS = {
    "AID": {"root": os.path.join(UAE_RS_ROOT, "AID"), "pattern": AID_RE},   # 5000 imgs, 30 classes
    "UCM": {"root": os.path.join(UAE_RS_ROOT, "UCM"), "pattern": UCM_RE},   # 1050 imgs, 21 classes
}


@dataclass
class SceneDataset:
    """A list of image paths with one integer class label each."""

    name: str
    paths: list = field(default_factory=list)
    labels: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))
    class_names: list = field(default_factory=list)

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.paths)

    def __repr__(self) -> str:
        return f"SceneDataset({self.name}: {len(self.paths)} images, {self.n_classes} classes)"


def load_scene_dataset(
    name: str = "AID",
    root: Optional[str] = None,
    pattern: Optional[re.Pattern] = None,
    max_per_class: Optional[int] = None,
    rng=None,
) -> SceneDataset:
    """Scan a scene-classification directory and parse the class of each file.

    name:          "AID" or "UCM" (or any name, if root and pattern are given).
    max_per_class: keep at most this many images per class — the quickest way
                   to shorten an epoch while keeping every class represented.

    Files that do not match the pattern are skipped with a warning rather than
    raising, so a partially re-downloaded directory still runs.
    """
    spec = DATASETS.get(name.upper(), {})
    root = root or spec.get("root")
    pattern = pattern or spec.get("pattern")
    if root is None or pattern is None:
        raise ValueError(f"unknown dataset {name!r}; pass root= and pattern= explicitly")
    if not os.path.isdir(root):
        raise FileNotFoundError(f"dataset directory not found: {root}")

    by_class: dict[str, list] = {}
    skipped = 0
    for fname in sorted(os.listdir(root)):
        m = pattern.match(fname)
        if not m:
            if fname.lower().endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
                skipped += 1
            continue
        by_class.setdefault(m.group("cls").lower(), []).append(os.path.join(root, fname))
    if skipped:
        warnings.warn(f"{skipped} image file(s) in {root} did not match the {name} naming pattern "
                      "and were skipped", stacklevel=2)
    if not by_class:
        raise ValueError(f"no images matched the {name} naming pattern in {root}")

    rng = rng or np.random.default_rng(0)
    class_names = sorted(by_class)          # sorted -> stable, reproducible indices
    paths, labels = [], []
    for idx, cls in enumerate(class_names):
        files = by_class[cls]
        if max_per_class is not None and len(files) > max_per_class:
            pick = rng.choice(len(files), size=max_per_class, replace=False)
            files = [files[i] for i in sorted(pick)]
        paths.extend(files)
        labels.extend([idx] * len(files))

    return SceneDataset(name=name.upper(), paths=paths,
                        labels=np.array(labels, dtype=int), class_names=class_names)


def train_test_split_scenes(
    ds: SceneDataset, test_frac: float = 0.2, rng=None,
) -> tuple[SceneDataset, SceneDataset]:
    """Class-stratified split BY IMAGE.

    This is the one non-obvious correctness requirement of the whole loader:
    subwindows drawn from the same tile overlap heavily, so splitting at the
    PATCH level would leak training pixels into the test set and report a
    wildly inflated accuracy. Splitting by image is what makes the reported
    numbers mean anything.
    """
    rng = rng or np.random.default_rng(0)
    tr_paths, tr_labels, te_paths, te_labels = [], [], [], []
    for c in range(ds.n_classes):
        idx = np.where(ds.labels == c)[0]
        idx = idx[rng.permutation(len(idx))]
        n_test = max(1, int(round(len(idx) * test_frac))) if len(idx) > 1 else 0
        for i in idx[:n_test]:
            te_paths.append(ds.paths[i]); te_labels.append(c)
        for i in idx[n_test:]:
            tr_paths.append(ds.paths[i]); tr_labels.append(c)

    train = SceneDataset(ds.name, tr_paths, np.array(tr_labels, dtype=int), ds.class_names)
    test = SceneDataset(ds.name, te_paths, np.array(te_labels, dtype=int), ds.class_names)
    return train, test


# ---------------------------------------------------------------------------
# Image loading and subwindow sampling
# ---------------------------------------------------------------------------

def load_image(path: str) -> np.ndarray:
    """Read one image as an (H, W, 3) uint8 array."""
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.uint8)


def _resize(arr: np.ndarray, size: int, resample: str) -> np.ndarray:
    """Resize an (h, w, 3) uint8 crop to (size, size, 3).

    PIL is the default and is ~35x faster than skimage's anti-aliased resize
    (measured 0.35 ms vs 12.3 ms), which decides whether building a 20k-patch
    epoch takes seconds or minutes. PIL scales its filter support when
    downscaling, so BILINEAR here IS anti-aliased — which matters, since these
    crops are typically downscaled 2-4x.
    """
    if arr.shape[0] == size and arr.shape[1] == size:
        return np.ascontiguousarray(arr)
    if resample == "pil":
        from PIL import Image

        return np.asarray(
            Image.fromarray(arr).resize((size, size), Image.BILINEAR), dtype=np.uint8
        )
    if resample == "nearest":
        rows = (np.arange(size) * arr.shape[0] // size).clip(0, arr.shape[0] - 1)
        cols = (np.arange(size) * arr.shape[1] // size).clip(0, arr.shape[1] - 1)
        return np.ascontiguousarray(arr[np.ix_(rows, cols)])
    if resample == "skimage":
        from skimage.transform import resize as sk_resize

        out = sk_resize(arr, (size, size), anti_aliasing=True, preserve_range=True)
        return np.clip(out, 0, 255).astype(np.uint8)
    raise ValueError(f"unknown resample backend {resample!r}")


def sample_subwindows(
    img: np.ndarray,
    n: int,
    patch_size: int = 64,
    min_frac: float = 0.5,
    max_frac: float = 1.0,
    rng=None,
    square: bool = True,
    resample: str = "pil",
) -> list:
    """Marée-style random subwindows, each resized to patch_size x patch_size.

        side ~ U(min_frac, max_frac) * min(H, W)
        r    ~ U(0, H - side),  c ~ U(0, W - side)
        out  = resize(img[r:r+side, c:c+side], (patch_size, patch_size))

    The size range is biased large (50-100% by default) so each patch still
    shows enough of the scene to carry the tile's label — and to stay
    recognizable in NodeView.

    square=True keeps Marée's square windows. Drawing height and width
    independently and then resizing to a square patch would distort texture
    statistics anisotropically, which is a different (and here unrequested)
    kind of augmentation from the non-square RECTANGLE TESTS the forest uses.
    """
    rng = rng or np.random.default_rng()
    H, W = img.shape[:2]
    short = min(H, W)
    out = []
    for _ in range(n):
        if square:
            side = int(round(rng.uniform(min_frac, max_frac) * short))
            side = max(2, min(side, short))
            sh = sw = side
        else:
            sh = max(2, min(int(round(rng.uniform(min_frac, max_frac) * H)), H))
            sw = max(2, min(int(round(rng.uniform(min_frac, max_frac) * W)), W))
        r = int(rng.integers(0, H - sh + 1))
        c = int(rng.integers(0, W - sw + 1))
        out.append(_resize(img[r:r + sh, c:c + sw], patch_size, resample))
    return out


def sample_patch_set(
    ds: SceneDataset,
    n: int,
    patch_size: int = 64,
    rng=None,
    prepared: bool = True,
    **subwindow_kw,
) -> tuple[list, np.ndarray]:
    """Eagerly draw n patches spread over the dataset's images.

    Used for the fixed evaluation set and the reference pool: those are routed
    again at every push, so they are prepared ONCE here and reused.
    """
    rng = rng or np.random.default_rng()
    if len(ds) == 0:
        raise ValueError("empty dataset")
    order = rng.permutation(len(ds))
    per_image = max(1, int(np.ceil(n / len(ds))))

    patches, labels = [], []
    for i in order:
        img = load_image(ds.paths[i])
        take = min(per_image, n - len(patches))
        for w in sample_subwindows(img, take, patch_size, rng=rng, **subwindow_kw):
            patches.append(PreparedPatch(w) if prepared else w)
            labels.append(int(ds.labels[i]))
        if len(patches) >= n:
            break

    idx = rng.permutation(len(patches))
    return [patches[i] for i in idx], np.array(labels, dtype=int)[idx]


def prepare_patch_stream(
    ds: SceneDataset,
    patch_size: int = 64,
    per_image: int = 4,
    batch_size: int = 2000,
    epochs: int = 1,
    rng=None,
    **subwindow_kw,
) -> Iterator[tuple[list, np.ndarray]]:
    """Yield (patches, labels) batches, streaming tiles from disk.

    One tile is loaded, `per_image` subwindows are drawn and prepared, then the
    tile is dropped. Holding all of AID in RAM would be ~1 GB of tiles before
    any patches exist; streaming keeps the footprint at roughly
    batch_size x 75 KB (2000 -> ~150 MB) plus one tile.

    Batches mix classes because the file order is shuffled per epoch. epochs>1
    re-streams a freshly shuffled list — with median splits (~50/50) the
    reachable depth grows with the total stream, so more epochs is the primary
    knob for deeper trees.
    """
    rng = rng or np.random.default_rng()
    patches: list = []
    labels: list = []

    for _ in range(max(1, epochs)):
        for i in rng.permutation(len(ds)):
            img = load_image(ds.paths[i])
            lbl = int(ds.labels[i])
            for w in sample_subwindows(img, per_image, patch_size, rng=rng, **subwindow_kw):
                patches.append(PreparedPatch(w))
                labels.append(lbl)
            if len(patches) >= batch_size:
                idx = rng.permutation(len(patches))
                yield [patches[j] for j in idx], np.array(labels, dtype=int)[idx]
                patches, labels = [], []

    if patches:
        idx = rng.permutation(len(patches))
        yield [patches[j] for j in idx], np.array(labels, dtype=int)[idx]
