"""
Assertion suite for fbp_datasets.

Runs standalone:  python test_fbp_datasets.py

Sections 1-5 use synthetic tiles and masks, so they run anywhere. Section 6
only runs if the extracted dataset is present, and checks the things that
CANNOT be checked synthetically: that the band bookkeeping matches the real
files, and that image and mask line up.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fbp_datasets as F  # noqa: E402

_failures = []


def check(cond, label):
    if cond:
        print(f"  ok   {label}")
    else:
        print(f"  FAIL {label}")
        _failures.append(label)


def synthetic(h=300, w=400, seed=0):
    """A tile whose left half is class 3 and right half class 7, plus an
    unlabeled stripe down the middle — enough to exercise every filter."""
    rng = np.random.default_rng(seed)
    img = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
    mask = np.zeros((h, w), dtype=np.uint8)
    mask[:, : w // 2] = 3
    mask[:, w // 2:] = 7
    mask[:, w // 2 - 10: w // 2 + 10] = F.UNLABELED
    return img, mask


# ── 1. Class table ───────────────────────────────────────────────────────────
print("\n1. class table")
check(len(F.FBP_CLASS_NAMES) == 24, "24 classes, unlabeled excluded")
check(F.N_FBP_CLASSES == 24, "N_FBP_CLASSES agrees")
check(F.FBP_CLASS_NAMES[0] == "industrial area", "index 0 is mask value 1")
check(F.FBP_CLASS_NAMES[23] == "airport", "index 23 is mask value 24")
colors = F.fbp_class_colors()
check(len(colors) == 24 and len(set(colors)) == 24, "24 distinct hex colours")
check(colors[0] == "#c80000", "industrial area is (200,0,0)")
check(all(c.startswith("#") and len(c) == 7 for c in colors), "hex format")
lm = F.fbp_label_map()
check(lm[0] == "Industrial Area" and len(lm) == 24, "label map is title-cased")
check(len(F.FBP_TEST_STEMS) == 30, "30 official test tiles")

# ── 2. Composite resolution ──────────────────────────────────────────────────
print("\n2. composite resolution")
check(F.resolve_composite(16, "rgb") == (2, 1, 0), "16-bit rgb -> R,G,B positions")
check(F.resolve_composite(16, "cir") == (3, 2, 1), "16-bit cir -> Nir,R,G positions")
check(F.resolve_composite(8, "cir") == (0, 1, 2), "8-bit cir -> Nir,R,G positions")
try:
    F.resolve_composite(8, "rgb")
    check(False, "8-bit rgb refused")
except ValueError as e:
    # The 8-bit tiles have no blue band; silently substituting one would put a
    # mislabelled picture in front of the forest.
    check("no B band" in str(e), "8-bit rgb refused with a specific message")
for bad in [(4, "rgb"), (8, "nir")]:
    try:
        F.resolve_composite(*bad)
        check(False, f"{bad} rejected")
    except ValueError:
        check(True, f"{bad} rejected")

# ── 3. Labelling rules ───────────────────────────────────────────────────────
print("\n3. labelling rules")
img, mask = synthetic()
rng = np.random.default_rng(0)
ps, ls = F.sample_labeled_patches(img, mask, 200, patch_size=32, rng=rng,
                                  scale_jitter=(1.0, 1.0))
check(len(ps) == len(ls), "one label per patch")
check(all(p.shape == (32, 32, 3) for p in ps), "patches are patch_size square")
check(all(p.dtype == np.uint8 for p in ps), "patches are uint8")
# Mask values 3 and 7 must arrive as training labels 2 and 6.
check(set(ls) <= {2, 6}, "labels shifted from 1..24 to 0..23")
check(all(0 <= l < 24 for l in ls), "labels within range")

# Unlabeled must never become a class, however lenient the filters are.
allzero = np.zeros_like(mask)
ps0, ls0 = F.sample_labeled_patches(img, allzero, 50, patch_size=32, rng=rng,
                                    max_unlabeled=1.0, min_purity=0.0)
check(len(ps0) == 0, "a fully unlabeled tile yields no patches")

# purity: a fine checkerboard makes EVERY 32 px crop half one class and half
# another, so min_purity=1.0 must reject all of them while a lenient filter
# takes them all. (A single boundary would not test this: on a 400 px-wide tile
# almost every random crop lands inside one half, so both settings return 100
# and the assertion passes vacuously.)
board = np.indices((256, 256)).sum(0) // 8 % 2
checker = np.where(board == 0, 3, 7).astype(np.uint8)
cimg = np.zeros((256, 256, 3), np.uint8)
pure, _ = F.sample_labeled_patches(cimg, checker, 40, patch_size=32, rng=rng,
                                   scale_jitter=(1.0, 1.0), min_purity=1.0,
                                   max_unlabeled=0.0)
mixed, lm_ = F.sample_labeled_patches(cimg, checker, 40, patch_size=32, rng=rng,
                                      scale_jitter=(1.0, 1.0), min_purity=0.0,
                                      max_unlabeled=0.0)
check(len(pure) == 0, "min_purity=1.0 rejects every mixed crop")
check(len(mixed) == 40, "min_purity=0.0 accepts them all")
check(set(lm_) <= {2, 6}, "mixed crops still get a valid majority label")

# max_unlabeled is enforced independently of purity.
half = np.full((256, 256), 3, np.uint8)
half[:, 128:] = F.UNLABELED           # exactly 50% unlabeled everywhere
strict_u, _ = F.sample_labeled_patches(cimg, half, 40, patch_size=256, rng=rng,
                                       scale_jitter=(1.0, 1.0), max_unlabeled=0.2)
loose_u, _ = F.sample_labeled_patches(cimg, half, 40, patch_size=256, rng=rng,
                                      scale_jitter=(1.0, 1.0), max_unlabeled=0.6)
check(len(strict_u) == 0, "max_unlabeled rejects a 50%-unlabeled crop at 0.2")
check(len(loose_u) == 40, "and accepts it at 0.6")

# A patch drawn wholly inside one class must carry that class, not the other.
solid = np.full((200, 200), 7, dtype=np.uint8)
_, ls_solid = F.sample_labeled_patches(img[:200, :200], solid, 30, patch_size=32,
                                       rng=rng, scale_jitter=(1.0, 1.0))
check(set(ls_solid) == {6}, "a uniform region labels every patch class 7 -> 6")

# center rule reads the middle pixel
ctr_img, ctr_mask = synthetic()
ps_c, ls_c = F.sample_labeled_patches(ctr_img, ctr_mask, 50, patch_size=32,
                                      rng=rng, scale_jitter=(1.0, 1.0),
                                      label_rule="center", min_purity=0.0)
check(set(ls_c) <= {2, 6}, "center rule also shifts labels")
try:
    F.sample_labeled_patches(img, mask, 1, rng=rng, label_rule="nonsense")
    check(False, "unknown label_rule rejected")
except ValueError:
    check(True, "unknown label_rule rejected")

# ── 4. Scale jitter and resizing ─────────────────────────────────────────────
print("\n4. scale jitter")
ps_j, _ = F.sample_labeled_patches(img, mask, 100, patch_size=16, rng=rng,
                                   scale_jitter=(1.0, 4.0), min_purity=0.5)
check(all(p.shape == (16, 16, 3) for p in ps_j), "jittered crops resized to patch_size")
check(len(ps_j) > 0, "jitter still yields patches")

# ── 5. Tile/mask alignment ───────────────────────────────────────────────────────────
print("\n5. tile/mask alignment")
a, b = F.align_to_mask(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.uint8))
check(a.shape[:2] == b.shape[:2] == (10, 10), "matching image/mask passes through")

# The real off-by-one: several FBP tiles are 6907 rows to a 6908-row mask.
a, b = F.align_to_mask(np.zeros((6907, 7300, 3), np.uint8),
                       np.zeros((6908, 7300), np.uint8), "offbyone")
check(a.shape[:2] == (6907, 7300) and b.shape[:2] == (6907, 7300),
      "one-row mismatch is cropped to the common size")

# Top-left alignment, not bottom: the mask's LAST row is the one dropped.
m = np.zeros((12, 10), np.uint8)
m[-1, :] = 9
_, b = F.align_to_mask(np.zeros((11, 10, 3), np.uint8), m, "topleft")
check(b.shape[:2] == (11, 10) and not (b == 9).any(),
      "alignment crops the mask bottom row, keeping the top-left origin")

# A big discrepancy is a wrong pairing, not a packaging quirk.
try:
    F.align_to_mask(np.zeros((10, 10, 3), np.uint8), np.zeros((200, 200), np.uint8), "x")
    check(False, "a large mismatch raises")
except ValueError:
    # A mask that does not line up would silently mislabel every patch.
    check(True, "a large mismatch raises")

# A colour mask read as an index mask must be refused, not silently turned
# into labels past the end of class_names.
import tempfile
from PIL import Image as _I
with tempfile.TemporaryDirectory() as td:
    bad = os.path.join(td, "colour_by_mistake.png")
    _I.fromarray(np.full((8, 8), 200, np.uint8)).save(bad)
    try:
        F.load_mask(bad)
        check(False, "an out-of-range mask is refused")
    except ValueError as e:
        check("above the maximum class index" in str(e),
              "an out-of-range mask is refused")
    good = os.path.join(td, "index.png")
    _I.fromarray(np.full((8, 8), 24, np.uint8)).save(good)
    check(F.load_mask(good).max() == 24, "a value of exactly 24 is accepted")

# Mismatched sizes passed straight to the sampler must not run off the tile.
ps_m, ls_m = F.sample_labeled_patches(np.zeros((99, 100, 3), np.uint8),
                                      np.full((100, 100), 3, np.uint8),
                                      20, patch_size=32, rng=rng,
                                      scale_jitter=(1.0, 1.0))
check(len(ps_m) == 20 and all(p.shape == (32, 32, 3) for p in ps_m),
      "sampler is bounded by the smaller of tile and mask")

hist = F.class_histogram([0, 0, 5, 23])
check(hist["industrial area"] == 2 and hist["airport"] == 1, "class_histogram counts")
check(len(hist) == 24, "class_histogram covers every class")

# ── 6. Real files, if extracted ──────────────────────────────────────────────
print("\n6. real dataset")
if not os.path.isdir(F.FBP_ROOT):
    print("  skipped — dataset not found at", F.FBP_ROOT)
else:
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ds16 = F.load_fbp_dataset(bits=16, composite="rgb", max_tiles=2)
        except (ValueError, FileNotFoundError) as e:
            ds16 = None
            print("  skipped 16-bit —", e)

    if ds16 is not None and len(ds16):
        check(len(ds16.image_paths) == len(ds16.mask_paths), "paths pair up")
        i = 0
        img_r = F.load_tile_rgb(ds16.image_paths[i], 16, "rgb")
        mask_r = F.load_mask(ds16.mask_paths[i])
        check(img_r.dtype == np.uint8 and img_r.ndim == 3 and img_r.shape[2] == 3,
              "16-bit tile loads as (H,W,3) uint8")
        img_r, mask_r = F.align_to_mask(img_r, mask_r, ds16.stems[i])
        check(img_r.shape[:2] == mask_r.shape[:2], "tile and mask align")
        check(mask_r.max() <= 24, "mask values within 0..24")
        # A stretched tile that came out constant would mean the percentile
        # stretch collapsed, which no visual check downstream would catch.
        check(img_r.std() > 5, "stretched tile has real contrast")

        ps_r, ls_r = F.sample_labeled_patches(img_r, mask_r, 50, 64,
                                              rng=np.random.default_rng(0))
        check(all(0 <= l < 24 for l in ls_r), "real labels within range")
        check(all(p.shape == (64, 64, 3) for p in ps_r), "real patches shaped right")

        # The train/test split must not overlap.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            full = F.load_fbp_dataset(bits=16, composite="rgb")
            tr, te = F.train_test_split_fbp(full)
        check(not (set(tr.stems) & set(te.stems)), "train/test tiles disjoint")
        check(len(tr) + len(te) == len(full), "split covers every tile")
        check(all(s in F.FBP_TEST_STEMS for s in te.stems), "test split is the official list")

    # The 8-bit inversion is the correction most likely to regress silently.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            ds8 = F.load_fbp_dataset(bits=8, composite="cir", max_tiles=1)
        except (ValueError, FileNotFoundError) as e:
            ds8 = None
            print("  skipped 8-bit —", e)
    if ds8 is not None and len(ds8):
        import rasterio

        p = ds8.image_paths[0]
        with rasterio.open(p) as src:
            check(src.count == 4, "8-bit tile has 4 bands")
            b4 = src.read(4, out_shape=(200, 200))
        # Band 4 constant is what makes true colour unreachable; if a future
        # re-download has a real blue band, this test should be revisited.
        check(b4.min() == b4.max() == 255, "8-bit band 4 is a constant K plane")
        cir8 = F.load_tile_rgb(p, 8, "cir")
        with rasterio.open(p) as src:
            raw1 = src.read(1)
        check(np.array_equal(cir8[:, :, 0], 255 - raw1), "8-bit values are un-inverted")

print("\n" + "=" * 60)
if _failures:
    print(f"{len(_failures)} FAILURE(S):")
    for f in _failures:
        print("  -", f)
    sys.exit(1)
print("all checks passed")
