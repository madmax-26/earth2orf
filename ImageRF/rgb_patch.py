"""
RGB patch preparation: summed-area tables for O(1) rectangle aggregates.

A patch is an (h, w, 3) uint8 image. Every node test in rgb_node_tests.py is a
comparison of rectangle aggregates, so the whole forest rests on one primitive:
the mean of an arbitrary rectangle, in constant time.

That is what the summed-area table (integral image) gives:

    S    = I[r+rh, c+rw] - I[r, c+rw] - I[r+rh, c] + I[r, c]
    mean = S / (rh * rw)

four lookups regardless of rectangle size. The table is built ONCE per patch
(PreparedPatch) and reused across every tree, node and candidate of the forest
— the structural advantage over the PolSAR forest, where each projection
recomputes its region operators from scratch.
"""

from __future__ import annotations

import array

import numpy as np

N_CHANNELS = 3


def integral_image(arr: np.ndarray) -> np.ndarray:
    """(h, w, C) uint8 -> (h+1, w+1, C) int32 summed-area table.

    The zero first row/column is what lets rectangles touching the top/left
    edge be read without any boundary special-casing. int32 is exact here:
    the largest possible sum is 255 * h * w (255*64*64 = 1_044_480 << 2**31).
    """
    h, w, c = arr.shape
    out = np.zeros((h + 1, w + 1, c), dtype=np.int32)
    np.cumsum(np.cumsum(arr, axis=0, dtype=np.int32), axis=1, out=out[1:, 1:])
    return out


class PreparedPatch:
    """An (h, w, 3) uint8 patch plus everything a RectTest needs to evaluate.

    Built once per patch and handed to the forest in place of the raw array;
    the reused PolSAR forest core passes patches through opaquely, so it never
    needs to know.

    Storage choices are measured, not guessed (see bench_rgb_rf.py):
      * the SAT is one flat array.array('i') per channel — ~3x faster per
        projection than scalar indexing into a 3-D ndarray, at the same memory.
      * pixels are one interleaved `bytes`, so the `center` operator gets a
        plain int without numpy scalar overhead.
    Budget is roughly 75 KB per prepared 64x64 patch, so batch size is a memory
    knob as much as a training one.
    """

    __slots__ = ("arr", "h", "w", "_sat", "_px", "_stride")

    def __init__(self, arr: np.ndarray):
        a = np.asarray(arr)
        if a.ndim == 2:                      # grayscale -> replicate to RGB
            a = np.repeat(a[:, :, None], N_CHANNELS, axis=2)
        if a.ndim != 3 or a.shape[2] != N_CHANNELS:
            raise ValueError(f"expected an (h, w, 3) RGB patch, got {np.asarray(arr).shape}")
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        # Copy: a patch is usually a view into a much larger tile, and keeping
        # the view alive would pin the whole tile in memory.
        a = np.ascontiguousarray(a)

        self.arr = a
        self.h, self.w = int(a.shape[0]), int(a.shape[1])
        self._stride = self.w + 1

        table = integral_image(a)
        self._sat = [
            array.array("i", np.ascontiguousarray(table[:, :, ch]).ravel().tolist())
            for ch in range(N_CHANNELS)
        ]
        self._px = a.tobytes()               # index: (r * w + c) * 3 + ch

    # ── O(1) region operators ────────────────────────────────────────────────

    def rect_mean(self, r: int, c: int, rh: int, rw: int, ch: int) -> float:
        """Mean of arr[r:r+rh, c:c+rw, ch] — four SAT lookups, any size.

        Exact and deterministic: the table is integer, so this is an integer
        sum divided by an integer area.
        """
        f = self._sat[ch]
        s = self._stride
        top = r * s + c
        bot = (r + rh) * s + c
        return (f[bot + rw] - f[top + rw] - f[bot] + f[top]) / (rh * rw)

    def rect_center(self, r: int, c: int, rh: int, rw: int, ch: int) -> int:
        """Value of the rectangle's centre pixel.

        Deliberately does NOT use the SAT (it is a single sample, not a sum).
        The `rh // 2` convention mirrors PolSAR's apply_operator "center"
        (region[rw // 2, rw // 2]), and it makes `center` and `mean` agree
        exactly on a 1x1 rectangle — which is how the classic Lepetit-Fua
        pixel-pair test falls out of either operator choice.
        """
        return self._px[((r + rh // 2) * self.w + (c + rw // 2)) * N_CHANNELS + ch]

    def __repr__(self) -> str:
        return f"PreparedPatch({self.h}x{self.w}x{N_CHANNELS})"


def as_prepared(patch) -> PreparedPatch:
    """PreparedPatch -> itself; raw ndarray -> freshly prepared.

    Idempotent, so it is free on the normal path. The prepare-on-the-fly branch
    exists only so a raw array arriving from TreeFlow or an evaluation helper cannot
    silently produce a WRONG answer — but it costs a full table build, so
    anything hot (training, evaluation, reference routing) must hand over
    already-prepared patches. rgb_forest and rgb_interface see to that.
    """
    return patch if type(patch) is PreparedPatch else PreparedPatch(patch)


def to_uint8_rgb(patch) -> np.ndarray:
    """Displayable (h, w, 3) uint8 view of a patch.

    Accepts a PreparedPatch, a uint8 RGB array, a grayscale array, or a float
    image in either [0, 1] or [0, 255]. This is the RGB replacement for
    polsar_interface.patch_to_rgb — here it is essentially the identity, which
    is the entire point of ImageRF.
    """
    if type(patch) is PreparedPatch:
        return patch.arr
    a = np.asarray(patch)
    if a.ndim == 2:
        a = np.repeat(a[:, :, None], N_CHANNELS, axis=2)
    if a.ndim != 3 or a.shape[2] not in (1, 3, 4):
        raise ValueError(f"cannot render an image of shape {a.shape}")
    if a.shape[2] == 1:
        a = np.repeat(a, N_CHANNELS, axis=2)
    a = a[:, :, :N_CHANNELS]
    if a.dtype != np.uint8:
        a = a.astype(np.float64)
        if a.size and a.max() <= 1.0:
            a = a * 255.0
        a = np.clip(a, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(a)
