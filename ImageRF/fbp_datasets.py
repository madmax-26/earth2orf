"""
Five-Billion-Pixels (GF-2) semantic-segmentation dataset and patch sampling.

Where rgb_datasets.py handles SCENE datasets — one label per file, parsed out of
the filename — this module handles the opposite arrangement: 150 very large
GF-2 tiles (6908 x 7300 px, 4 m GSD) each paired with a full-resolution mask
whose pixel VALUES are the class indexes. A random subwindow of such a tile has
no single label, so the label has to be read out of the mask crop.

Everything downstream is unchanged: this module still yields
(list[PreparedPatch], labels) batches shaped exactly like
rgb_datasets.prepare_patch_stream, so ImageOnlineRandomForest, the TreeFlow
interface and the frontend need no modification at all.

Three things are specific to this dataset and are the reason the sampler is not
just "crop and take the mask":

  * Index 0 means UNLABELED, not a class. Training on it would teach the forest
    a 25th category that means "the annotator stopped here". Patches are
    rejected on unlabeled coverage, and the surviving indexes 1..24 are shifted
    down to 0..23 so they index class_names directly.

  * A crop straddling a class boundary is genuinely ambiguous. `purity` is the
    fraction of labeled pixels belonging to the majority class; crops below
    min_purity are rejected, which keeps a whole-patch label honest.

  * Band order is NOT RGB, and the 8-bit tiles are stranger than their folder
    name suggests. Verified against the files themselves rather than the readme:

      Image_16bit_BGRNir  4 real uint16 bands in (B, G, R, Nir) order. Taking
                          (R, G, B) gives a genuine true-colour scene.
      Image__8bit_NirRGB  tagged photometric=CMYK and ADOBE-INVERTED: the true
                          value is 255 - stored. After inverting, the three
                          usable planes are (Nir, R, G) and the fourth is a
                          constant K plane — checked over every available tile.
                          There is NO blue band here, so true colour is not
                          reachable from these files; they give a colour-
                          infrared composite (vegetation bright red).

    Hence `composite`: 'rgb' (true colour, needs bits=16) or 'cir' (Nir,R,G,
    available from either). Reading the 8-bit tiles naively — no inversion,
    bands 1-3 as RGB — produces a photographic negative, which is what a
    correlation of -0.95 against the 16-bit tile of the same scene revealed.

Layout expected under `root` (as produced by big_datasets/extract_merge.ps1):

    Image_16bit_BGRNir/<stem>.tiff       150 16-bit tiles  (bits=16, default)
    Image__8bit_NirRGB/<stem>.tif        150 8-bit tiles   (bits=8)
    Annotation__index/<stem>_24label.png 150 index masks
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Iterator, Optional

import numpy as np

from rgb_patch import PreparedPatch

# Dataset location: $FBP_ROOT if set, else <repo>/big_datasets/Five-Billion-Pixels.
FBP_ROOT = os.environ.get("FBP_ROOT") or os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "big_datasets", "Five-Billion-Pixels")

# ---------------------------------------------------------------------------
# Classes — verbatim from the dataset's "readme (important).txt"
# ---------------------------------------------------------------------------

# Ordered by mask index 1..24; index 0 ("unlabeled") is deliberately absent, so
# position i here is the TRAINING label i and the mask value i + 1.
FBP_CLASSES = [
    ("industrial area",   (200,   0,   0)),
    ("paddy field",       (  0, 200,   0)),
    ("irrigated field",   (150, 250,   0)),
    ("dry cropland",      (150, 200, 150)),
    ("garden land",       (200,   0, 200)),
    ("arbor forest",      (150,   0, 250)),
    ("shrub forest",      (150, 150, 250)),
    ("park",              (200, 150, 200)),
    ("natural meadow",    (250, 200,   0)),
    ("artificial meadow", (200, 200,   0)),
    ("river",             (  0,   0, 200)),
    ("urban residential", (250,   0, 150)),
    ("lake",              (  0, 150, 200)),
    ("pond",              (  0, 200, 250)),
    ("fish pond",         (150, 200, 250)),
    ("snow",              (250, 250, 250)),
    ("bareland",          (200, 200, 200)),
    ("rural residential", (200, 150, 150)),
    ("stadium",           (250, 200, 150)),
    ("square",            (150, 150,   0)),
    ("road",              (250, 150, 150)),
    ("overpass",          (250, 150,   0)),
    ("railway station",   (250, 200, 250)),
    ("airport",           (200, 150,   0)),
]

FBP_CLASS_NAMES = [n for n, _ in FBP_CLASSES]
N_FBP_CLASSES = len(FBP_CLASS_NAMES)          # 24
UNLABELED = 0                                  # mask value, never a train label

# The 30 tiles held out for the baseline in Table 4 of the paper. Using this
# rather than a random split is what makes a number here comparable to theirs.
FBP_TEST_STEMS = frozenset([
    "GF2_PMS1__L1A0001064454-MSS1", "GF2_PMS1__L1A0001118839-MSS1",
    "GF2_PMS1__L1A0001344822-MSS1", "GF2_PMS1__L1A0001348919-MSS1",
    "GF2_PMS1__L1A0001366278-MSS1", "GF2_PMS1__L1A0001366284-MSS1",
    "GF2_PMS1__L1A0001395956-MSS1", "GF2_PMS1__L1A0001432972-MSS1",
    "GF2_PMS1__L1A0001670888-MSS1", "GF2_PMS1__L1A0001680857-MSS1",
    "GF2_PMS1__L1A0001680858-MSS1", "GF2_PMS1__L1A0001757429-MSS1",
    "GF2_PMS1__L1A0001765574-MSS1", "GF2_PMS2__L1A0000607677-MSS2",
    "GF2_PMS2__L1A0000607681-MSS2", "GF2_PMS2__L1A0000718813-MSS2",
    "GF2_PMS2__L1A0001038935-MSS2", "GF2_PMS2__L1A0001038936-MSS2",
    "GF2_PMS2__L1A0001119060-MSS2", "GF2_PMS2__L1A0001367840-MSS2",
    "GF2_PMS2__L1A0001378491-MSS2", "GF2_PMS2__L1A0001378501-MSS2",
    "GF2_PMS2__L1A0001396036-MSS2", "GF2_PMS2__L1A0001396037-MSS2",
    "GF2_PMS2__L1A0001416129-MSS2", "GF2_PMS2__L1A0001471436-MSS2",
    "GF2_PMS2__L1A0001517494-MSS2", "GF2_PMS2__L1A0001591676-MSS2",
    "GF2_PMS2__L1A0001787564-MSS2", "GF2_PMS2__L1A0001821754-MSS2",
])

# subdir, extension, whether values are Adobe-inverted, and the position of
# each band within the file. See the module docstring — these were verified
# against the files, and the 8-bit layout is not the one the folder name implies.
_IMAGE_SPECS = {
    8:  dict(subdir="Image__8bit_NirRGB", ext=".tif",  invert=True,
             bands={"Nir": 0, "R": 1, "G": 2}),        # band 4 is a constant K plane
    16: dict(subdir="Image_16bit_BGRNir", ext=".tiff", invert=False,
             bands={"B": 0, "G": 1, "R": 2, "Nir": 3}),
}

# Which three bands each composite displays, in output R,G,B order.
_COMPOSITES = {
    "rgb": ("R", "G", "B"),      # true colour
    "cir": ("Nir", "R", "G"),    # colour infrared: vegetation reads bright red
}
_MASK_DIR = "Annotation__index"
_MASK_SUFFIX = "_24label.png"


def fbp_label_map(class_names: Optional[list] = None) -> dict:
    """{0: 'Industrial Area', ...} for treeflow.push(label_map=...).

    The FBP names already contain spaces, so unlike rgb_interface._prettify
    (which exists to split run-together AID names) this is plain title-casing.
    """
    names = class_names or FBP_CLASS_NAMES
    return {i: n.title() for i, n in enumerate(names)}


def fbp_class_colors() -> list:
    """The dataset's own 24 class colours as hex, for Forest.class_colors.

    Preferred over rgb_interface.scene_class_colors here: these are the exact
    colours used by Annotation__color, so a node's class mix in the frontend
    reads the same as the published label masks.
    """
    return ["#{:02x}{:02x}{:02x}".format(*rgb) for _, rgb in FBP_CLASSES]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

@dataclass
class FBPDataset:
    """Image/mask path pairs. Unlike SceneDataset there are no per-file labels."""

    name: str = "FBP"
    image_paths: list = field(default_factory=list)
    mask_paths: list = field(default_factory=list)
    stems: list = field(default_factory=list)
    class_names: list = field(default_factory=lambda: list(FBP_CLASS_NAMES))
    bits: int = 16
    composite: str = "rgb"

    @property
    def n_classes(self) -> int:
        return len(self.class_names)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __repr__(self) -> str:
        return (f"FBPDataset({self.name}: {len(self.image_paths)} tiles, "
                f"{self.n_classes} classes, {self.bits}-bit {self.composite})")


def load_fbp_dataset(
    root: str = FBP_ROOT,
    bits: int = 16,
    composite: str = "rgb",
    max_tiles: Optional[int] = None,
    rng=None,
) -> FBPDataset:
    """Pair every image tile with its index mask.

    bits:      16 (default) reads the raw tiles and stretches them at load
               time — the only source of a true-colour composite, since the
               8-bit tiles carry no blue band. 8 reads the re-quantized tiles
               the paper used, which only support composite='cir'.
    composite: 'rgb' true colour, or 'cir' colour-infrared (Nir, R, G).
    max_tiles: keep at most this many tiles — the quickest way to shorten a run.

    A tile whose mask is missing is skipped with a warning rather than raising,
    so a partially extracted directory still runs.
    """
    resolve_composite(bits, composite)      # fail fast on an impossible pairing
    spec = _IMAGE_SPECS[bits]
    subdir, ext = spec["subdir"], spec["ext"]
    img_dir = os.path.join(root, subdir)
    mask_dir = os.path.join(root, _MASK_DIR)
    for d in (img_dir, mask_dir):
        if not os.path.isdir(d):
            raise FileNotFoundError(f"dataset directory not found: {d}")

    stems, images, masks, missing = [], [], [], 0
    for fname in sorted(os.listdir(img_dir)):
        if not fname.endswith(ext):
            continue
        stem = fname[: -len(ext)]
        mask = os.path.join(mask_dir, stem + _MASK_SUFFIX)
        if not os.path.isfile(mask):
            missing += 1
            continue
        stems.append(stem)
        images.append(os.path.join(img_dir, fname))
        masks.append(mask)

    if missing:
        warnings.warn(f"{missing} tile(s) in {img_dir} have no matching mask in "
                      f"{mask_dir} and were skipped", stacklevel=2)
    if not images:
        raise ValueError(f"no image/mask pairs found under {root}")

    if max_tiles is not None and len(images) > max_tiles:
        rng = rng or np.random.default_rng(0)
        pick = sorted(rng.choice(len(images), size=max_tiles, replace=False))
        stems = [stems[i] for i in pick]
        images = [images[i] for i in pick]
        masks = [masks[i] for i in pick]

    return FBPDataset(name="FBP", image_paths=images, mask_paths=masks,
                      stems=stems, class_names=list(FBP_CLASS_NAMES), bits=bits,
                      composite=composite)


def train_test_split_fbp(ds: FBPDataset, test_stems=FBP_TEST_STEMS):
    """Split BY TILE using the paper's official held-out list.

    Splitting by tile (not by patch) is the same correctness requirement as in
    rgb_datasets.train_test_split_scenes, and it bites harder here: two crops
    from one 6908x7300 tile share illumination, season, sensor gain and often
    the very same fields, so a patch-level split would report a badly inflated
    accuracy.
    """
    def sub(keep):
        idx = [i for i, s in enumerate(ds.stems) if (s in test_stems) == keep]
        return FBPDataset(name=ds.name,
                          image_paths=[ds.image_paths[i] for i in idx],
                          mask_paths=[ds.mask_paths[i] for i in idx],
                          stems=[ds.stems[i] for i in idx],
                          class_names=ds.class_names, bits=ds.bits,
                          composite=ds.composite)

    train, test = sub(False), sub(True)
    if not len(test):
        warnings.warn("none of the official test tiles are present; the test "
                      "split is empty", stacklevel=2)
    return train, test


# ---------------------------------------------------------------------------
# Tile and mask loading
# ---------------------------------------------------------------------------

def resolve_composite(bits: int, composite: str) -> tuple:
    """(bits, composite) -> the three band positions to read, in R,G,B order.

    Raises on 'rgb' with bits=8: those files have no blue band, and silently
    substituting something else would put a mislabelled picture in front of
    both the forest and the user.
    """
    if bits not in _IMAGE_SPECS:
        raise ValueError(f"bits must be 8 or 16, got {bits!r}")
    if composite not in _COMPOSITES:
        raise ValueError(f"composite must be one of {sorted(_COMPOSITES)}, got {composite!r}")
    available = _IMAGE_SPECS[bits]["bands"]
    wanted = _COMPOSITES[composite]
    missing = [b for b in wanted if b not in available]
    if missing:
        raise ValueError(
            f"the {bits}-bit tiles have no {'/'.join(missing)} band, so "
            f"composite={composite!r} is not available from them "
            f"(they carry {'/'.join(available)}). Use bits=16 for true colour, "
            f"or composite='cir'.")
    return tuple(available[b] for b in wanted)


def load_tile_rgb(path: str, bits: int = 16, composite: str = "rgb") -> np.ndarray:
    """Read one GF-2 tile as a displayable (H, W, 3) uint8 array.

    rasterio reads only the three bands the composite needs, so a tile costs
    ~150 MB rather than the ~200 MB (8-bit) or ~400 MB (16-bit) of a full read.

    Two corrections happen here, both established by inspecting the files (see
    the module docstring): 8-bit tiles are un-inverted out of their CMYK
    encoding, and 16-bit tiles are stretched to 8 bits per tile on the 2nd/98th
    percentiles. The readme notes that re-quantization method measurably affects
    domain-adaptation results, so this stretch is a sane default, NOT the
    paper's — comparing against published numbers argues for bits=8.
    """
    import rasterio

    spec = _IMAGE_SPECS[bits]
    idx = resolve_composite(bits, composite)
    with rasterio.open(path) as src:
        arr = src.read([i + 1 for i in idx])              # 1-based; (3, H, W)
    arr = np.transpose(arr, (1, 2, 0))                    # (H, W, 3)

    if spec["invert"]:
        # Adobe-inverted CMYK: true value is 255 - stored. Without this the
        # tile is a photographic negative.
        arr = 255 - arr

    if arr.dtype == np.uint8:
        return np.ascontiguousarray(arr)

    out = np.empty(arr.shape, dtype=np.uint8)
    for c in range(arr.shape[2]):
        band = arr[:, :, c]
        lo, hi = np.percentile(band, (2.0, 98.0))
        if hi <= lo:
            hi = lo + 1.0
        scaled = (band.astype(np.float32) - lo) * (255.0 / (hi - lo))
        out[:, :, c] = np.clip(scaled, 0, 255).astype(np.uint8)
    return out


def load_mask(path: str) -> np.ndarray:
    """Read one index mask as an (H, W) uint8 array of values 0..24."""
    from PIL import Image

    # These masks are ~50 M pixels, well past PIL's decompression-bomb guard.
    # The files are ours and their size is known, so the guard is just noise.
    prev, Image.MAX_IMAGE_PIXELS = Image.MAX_IMAGE_PIXELS, None
    try:
        with Image.open(path) as im:
            if im.mode not in ("L", "P", "I;16", "I"):
                im = im.convert("L")
            arr = np.asarray(im)
    finally:
        Image.MAX_IMAGE_PIXELS = prev

    if arr.ndim == 3:               # a stray RGB-encoded index mask
        arr = arr[:, :, 0]
    arr = np.ascontiguousarray(arr.astype(np.uint8))

    # An index mask must hold 0..24. A colour mask read by mistake would sail
    # through everything above and then quietly produce labels past the end of
    # class_names, so it is rejected here rather than downstream.
    hi = int(arr.max()) if arr.size else 0
    if hi > N_FBP_CLASSES:
        raise ValueError(f"{os.path.basename(path)}: mask holds value {hi}, above the "
                         f"maximum class index {N_FBP_CLASSES} — is this an "
                         f"Annotation__index mask and not Annotation__color?")
    return arr


def _resize_rgb(arr: np.ndarray, size: int) -> np.ndarray:
    """(h, w, 3) uint8 -> (size, size, 3), anti-aliased on downscale.

    Same PIL BILINEAR choice as rgb_datasets._resize and for the same measured
    reason; PIL scales its filter support when downscaling, so this is properly
    anti-aliased for the 2-3x reductions scale jitter produces.
    """
    if arr.shape[0] == size and arr.shape[1] == size:
        return np.ascontiguousarray(arr)
    from PIL import Image

    return np.asarray(
        Image.fromarray(arr).resize((size, size), Image.BILINEAR), dtype=np.uint8
    )


# ---------------------------------------------------------------------------
# Patch sampling
# ---------------------------------------------------------------------------

def sample_labeled_patches(
    img: np.ndarray,
    mask: np.ndarray,
    n: int,
    patch_size: int = 64,
    rng=None,
    scale_jitter: tuple = (1.0, 2.0),
    min_purity: float = 0.6,
    max_unlabeled: float = 0.2,
    max_attempts: int = 30,
    label_rule: str = "majority",
) -> tuple[list, list]:
    """Draw up to n (patch, label) pairs from one tile.

        side  ~ U(scale_jitter) * patch_size
        crop  = img[r:r+side, c:c+side]  ->  resized to patch_size
        label = majority class of mask[r:r+side, c:c+side], ignoring unlabeled

    scale_jitter borrows Marée's random-size subwindows for scale robustness;
    (1.0, 1.0) gives pure fixed-scale crops. Note the trade-off it introduces
    HERE that does not exist for scene data: a larger window carries more
    context but is likelier to straddle a boundary and be rejected by
    min_purity.

    Rejection is capped at max_attempts per patch, so a tile that is mostly
    unlabeled or mostly boundary yields fewer than n patches instead of
    spinning. Returns (patches, labels) with labels already shifted to 0..23.
    """
    if label_rule not in ("majority", "center"):
        raise ValueError(f"unknown label_rule {label_rule!r}")
    rng = rng or np.random.default_rng()
    # Bound by BOTH arrays, not just the mask: align_to_mask normally makes
    # them equal, but this function is also callable directly, and drawing an
    # origin valid only for the mask would run the crop off the tile.
    H = min(mask.shape[0], img.shape[0])
    W = min(mask.shape[1], img.shape[1])
    lo, hi = scale_jitter
    short = min(H, W)

    patches, labels = [], []
    for _ in range(n):
        for _attempt in range(max_attempts):
            side = int(round(rng.uniform(lo, hi) * patch_size))
            side = max(2, min(side, short))
            r = int(rng.integers(0, H - side + 1))
            c = int(rng.integers(0, W - side + 1))

            m = mask[r:r + side, c:c + side]
            # bincount over 0..24 is far cheaper than np.unique here, and it is
            # run once per attempt so it sits on the hot path.
            counts = np.bincount(m.ravel(), minlength=N_FBP_CLASSES + 1)
            total = m.size
            if counts[UNLABELED] / total > max_unlabeled:
                continue
            labeled = total - counts[UNLABELED]
            if labeled == 0:
                continue

            if label_rule == "majority":
                cls = int(np.argmax(counts[1:])) + 1
                if counts[cls] / labeled < min_purity:
                    continue
            else:
                cls = int(m[side // 2, side // 2])
                if cls == UNLABELED:
                    continue

            crop = img[r:r + side, c:c + side]
            patches.append(_resize_rgb(crop, patch_size))
            labels.append(cls - 1)              # 1..24 -> 0..23
            break

    return patches, labels


def sample_patch_set(
    ds: FBPDataset,
    n: int,
    patch_size: int = 64,
    rng=None,
    prepared: bool = True,
    **kw,
) -> tuple[list, np.ndarray]:
    """Eagerly draw n patches spread over the dataset's tiles.

    Used for the fixed evaluation set and the reference pool, which are routed
    again at every push and so are prepared ONCE here. Mirrors
    rgb_datasets.sample_patch_set, but each tile costs a ~200 MB decode, so it
    takes as many patches as it can from a tile before moving on.
    """
    rng = rng or np.random.default_rng()
    if len(ds) == 0:
        raise ValueError("empty dataset")
    order = rng.permutation(len(ds))
    per_tile = max(1, int(np.ceil(n / len(ds))))

    patches, labels = [], []
    for i in order:
        img = load_tile_rgb(ds.image_paths[i], ds.bits, ds.composite)
        mask = load_mask(ds.mask_paths[i])
        img, mask = align_to_mask(img, mask, ds.stems[i])
        take = min(per_tile, n - len(patches))
        ps, ls = sample_labeled_patches(img, mask, take, patch_size, rng=rng, **kw)
        patches.extend(PreparedPatch(p) if prepared else p for p in ps)
        labels.extend(ls)
        del img, mask
        if len(patches) >= n:
            break

    if not patches:
        raise ValueError("no patch passed the purity/unlabeled filters; try "
                         "min_purity lower or max_unlabeled higher")
    idx = rng.permutation(len(patches))
    return [patches[i] for i in idx], np.array(labels, dtype=int)[idx]


def prepare_patch_stream(
    ds: FBPDataset,
    patch_size: int = 64,
    per_tile: int = 400,
    batch_size: int = 2000,
    epochs: int = 1,
    rng=None,
    verbose: bool = False,
    **kw,
) -> Iterator[tuple[list, np.ndarray]]:
    """Yield (patches, labels) batches, streaming tiles from disk.

    Shape-compatible with rgb_datasets.prepare_patch_stream, so
    rf.partial_fit(patches, labels) is unchanged.

    per_tile is much higher than the scene version's per_image for a concrete
    reason: decoding one tile costs ~200 MB and a second or two, versus ~1 ms
    for an AID thumbnail, so the cost is amortized by drawing many patches
    before dropping the tile. The price is correlation WITHIN a batch — crops
    from one tile share sensor and season — which is why tiles are reshuffled
    every epoch and a batch spans several tiles.
    """
    rng = rng or np.random.default_rng()
    patches: list = []
    labels: list = []

    for ep in range(max(1, epochs)):
        for i in rng.permutation(len(ds)):
            img = load_tile_rgb(ds.image_paths[i], ds.bits, ds.composite)
            mask = load_mask(ds.mask_paths[i])
            img, mask = align_to_mask(img, mask, ds.stems[i])
            ps, ls = sample_labeled_patches(img, mask, per_tile, patch_size,
                                            rng=rng, **kw)
            del img, mask
            if verbose:
                print(f"  [epoch {ep}] {ds.stems[i]}: {len(ps)}/{per_tile} patches")
            patches.extend(PreparedPatch(p) for p in ps)
            labels.extend(ls)
            while len(patches) >= batch_size:
                idx = rng.permutation(batch_size)
                yield ([patches[j] for j in idx],
                       np.array(labels[:batch_size], dtype=int)[idx])
                patches, labels = patches[batch_size:], labels[batch_size:]

    if patches:
        idx = rng.permutation(len(patches))
        yield [patches[j] for j in idx], np.array(labels, dtype=int)[idx]


# A handful of tiles ship one row shorter than their mask (6907 vs 6908 rows,
# e.g. GF2_PMS1__L1A0001344822-MSS1). Anything larger than this is not a
# known packaging quirk and more likely a wrong pairing, so it still raises.
_MAX_SIZE_SLACK = 4


def align_to_mask(img: np.ndarray, mask: np.ndarray, stem: str = "") -> tuple:
    """Crop tile and mask to their common size, top-left aligned.

    A mask that does not line up would silently mislabel every patch drawn from
    it, so this is checked per tile rather than trusted. Several FBP tiles are
    off by a single row; both are cropped from the top-left, which is the
    convention the geotransform origin implies and which measured marginally
    better than bottom-alignment when scoring label boundaries against image
    gradients. A discrepancy beyond _MAX_SIZE_SLACK px is treated as a wrong
    pairing and raises.
    """
    ih, iw = img.shape[:2]
    mh, mw = mask.shape[:2]
    if (ih, iw) == (mh, mw):
        return img, mask
    if abs(ih - mh) > _MAX_SIZE_SLACK or abs(iw - mw) > _MAX_SIZE_SLACK:
        raise ValueError(f"{stem}: image {(ih, iw)} and mask {(mh, mw)} differ by "
                         f"more than {_MAX_SIZE_SLACK} px — these are probably "
                         f"not the same scene")
    h, w = min(ih, mh), min(iw, mw)
    return img[:h, :w], mask[:h, :w]


def class_histogram(labels) -> dict:
    """{class_name: count} for a label array — the quickest imbalance check.

    FBP is heavily imbalanced (a lot of cropland and residential, very little
    stadium or railway station), so it is worth looking at this before reading
    anything into an accuracy figure.
    """
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=N_FBP_CLASSES)
    return {FBP_CLASS_NAMES[i]: int(counts[i]) for i in range(N_FBP_CLASSES)}
