"""
Rectangle-comparison node tests for RGB patches, and their random sampler.

This is the RGB counterpart of PolSAR's node_tests.py, and follows the same
structure (a fully-parameterised, serialisable NodeTest dataclass plus a
sampler that draws random ones) so the two forests feel like one codebase.

The test family is the Semantic Texton Forest / randomized-trees lineage:

    Shotton, Johnson & Cipolla, "Semantic Texton Forests for Image
    Categorization and Segmentation", CVPR 2008 — split functions on the value
    of a single pixel, and the sum / difference / absolute difference of a pair.

    Lepetit & Fua, "Randomized Trees for Real-Time Keypoint Recognition",
    CVPR 2005 / PAMI 2006 — the original pixel-intensity-difference tests.

Two deliberate design choices, both explained where they are implemented:
  * aggregates are taken over RECTANGLES rather than single pixels (noise
    robustness, and every test is drawable as two boxes on the patch), with a
    1x1 rectangle degenerating to the classic pixel test;
  * there is NO four-rectangle "difference of two differences" projection —
    Lepetit & Fua measured that family (their C4) as slightly WORSE than the
    simple pair tests, and it doubles the cost.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rgb_patch import N_CHANNELS, PreparedPatch, as_prepared

# ---------------------------------------------------------------------------
# Test family
# ---------------------------------------------------------------------------

# No min/max: they cannot be read from a summed-area table, so they would cost
# O(area) per evaluation and destroy the O(1) guarantee the whole design rests
# on. (In PolSAR, min_span/max_span were physically meaningful — the brightest
# and darkest scatterer. For RGB a small-rectangle mean covers the same ground.)
OPERATORS = ["mean", "center"]

# NOTE: no entry here may ever be the string "1pt". The reused PolSAR forest
# core (polsar_online_rf._sample_candidates) contains
#     if test.projection == "1pt" and test.reference is None: continue
# which is inert for any other value — that inertness, plus RectTest.reference
# being present and None, is exactly what lets polsar_online_rf.py be reused
# byte-for-byte unchanged. test_rgb_rf.py pins this.
PROJECTIONS = ["unary", "diff", "sum", "absdiff"]

N_RECTS = {"unary": 1, "diff": 2, "sum": 2, "absdiff": 2}


@dataclass
class RectTest:
    """One randomly sampled rectangle-comparison test.

    With phi_i = the aggregate of rectangle i read in channel channels[i]:

        unary   ->  phi_1                 absolute level ("is this area bright?")
        diff    ->  phi_1 - phi_2         contrast; invariant to a global
                                          illumination shift — the workhorse
        sum     ->  phi_1 + phi_2         joint level over two areas
        absdiff -> |phi_1 - phi_2|        contrast magnitude regardless of sign

    Fields
    ------
    projection : str
        One of PROJECTIONS. Never "1pt" (see the note on PROJECTIONS).
    rects : list of (row, col, rect_h, rect_w)
        Rectangles inside the patch. One entry for "unary", two otherwise.
    operators : list of str
        Aggregate applied to each rectangle: "mean" or "center".
    channels : list of int
        RGB channel index read for each rectangle.
    reference : None
        Always None. Present only to satisfy the duck-typed NodeTest contract
        that the reused PolSAR forest probes; RGB tests have no reference pool.
    threshold : float
        Split threshold, set to the MEDIAN of the projected values by
        OnlinePolSARDecisionTree._fix_split during training.
    """

    projection: str
    rects:      list
    operators:  list
    channels:   list
    reference:  Optional[None] = field(default=None, repr=False)
    threshold:  float = 0.0

    def project(self, patch) -> float:
        """Scalar projection value for one patch.

        The hottest function in the system — one call per candidate, per
        sample, per tree — so it is written flat: no helper calls, no numpy, no
        tuple unpacking in a loop, and `abs` inlined.
        """
        p = patch if type(patch) is PreparedPatch else as_prepared(patch)

        rects, ops, chs = self.rects, self.operators, self.channels
        r, c, rh, rw = rects[0]
        v0 = (p.rect_mean(r, c, rh, rw, chs[0]) if ops[0] == "mean"
              else p.rect_center(r, c, rh, rw, chs[0]))

        proj = self.projection
        if proj == "unary":
            return v0

        r, c, rh, rw = rects[1]
        v1 = (p.rect_mean(r, c, rh, rw, chs[1]) if ops[1] == "mean"
              else p.rect_center(r, c, rh, rw, chs[1]))

        if proj == "diff":
            return v0 - v1
        if proj == "sum":
            return v0 + v1
        if proj == "absdiff":
            return v0 - v1 if v0 >= v1 else v1 - v0
        raise ValueError(f"Unknown projection: {proj}")

    def evaluate(self, patch) -> bool:
        """Return True -> left child, False -> right child."""
        return self.project(patch) < self.threshold

    def fit_threshold(self, patches: list) -> None:
        """Set the threshold to the median of projections over these patches.

        Only used for offline/manual fitting; the online forest sets .threshold
        directly in _fix_split (with the same median rule).
        """
        self.threshold = float(np.median([self.project(p) for p in patches]))


# ---------------------------------------------------------------------------
# Random test generator
# ---------------------------------------------------------------------------

class RectTestSampler:
    """Samples random RectTests over a fixed-size square RGB patch.

    Parameters
    ----------
    patch_size : int
        Side length of the patch. Also read by the forest's
        predict_proba_image, hence the public attribute.
    min_rect_size, max_rect_size : int
        Bounds on a rectangle side. min 1 makes the classic single-pixel test
        reachable; max defaults to the full patch.
    projections, operators, channels : list or None
        Restrict the draw. None -> use all of them.
    cross_channel : bool
        False (default, faithful to Shotton): both rectangles of a pair read
        the same channel. True: channels are drawn independently, which adds
        colour-opponency tests such as "red here vs. blue there".
    """

    def __init__(
        self,
        patch_size: int = 64,
        min_rect_size: int = 1,
        max_rect_size: Optional[int] = None,
        projections: Optional[list] = None,
        operators: Optional[list] = None,
        channels: Optional[list] = None,
        cross_channel: bool = False,
    ):
        self.patch_size = int(patch_size)
        self.min_rect_size = max(1, int(min_rect_size))
        self.max_rect_size = int(max_rect_size or patch_size)
        if self.max_rect_size > self.patch_size:
            raise ValueError(
                f"max_rect_size {self.max_rect_size} exceeds patch_size {self.patch_size}"
            )
        if self.min_rect_size > self.max_rect_size:
            raise ValueError("min_rect_size must not exceed max_rect_size")
        self.projections = list(projections or PROJECTIONS)
        self.operators = list(operators or OPERATORS)
        self.channels = list(channels if channels is not None else range(N_CHANNELS))
        self.cross_channel = bool(cross_channel)

        if "1pt" in self.projections:                    # see PROJECTIONS
            raise ValueError('"1pt" is reserved by the PolSAR forest core and must not be used')

        self._log_lo = float(np.log(self.min_rect_size))
        self._log_hi = float(np.log(self.max_rect_size + 1))

    def _sample_rect(self, rng: np.random.Generator) -> tuple:
        """Sample one (row, col, rect_h, rect_w) inside the patch.

        Height and width are drawn INDEPENDENTLY (non-square rectangles pick up
        oriented structure — runways, field furrows, building edges) and
        LOG-UNIFORMLY: that gives every octave of scale equal probability and
        actually reaches 1x1. Uniform sampling over [1, 64] would put ~87% of
        the mass above side 8 and essentially never draw the fine tests.

        Position is uniform over all valid placements in the whole patch. This
        is the deliberate divergence from PolSAR's NodeTestSampler, which keeps
        regions near the patch centre because there the label belongs to the
        centre pixel; here the patch IS the object being classified, so no
        location is privileged.
        """
        lo, hi = self._log_lo, self._log_hi
        rh = int(np.exp(rng.uniform(lo, hi)))
        rw = int(np.exp(rng.uniform(lo, hi)))
        rh = min(max(rh, self.min_rect_size), self.max_rect_size)
        rw = min(max(rw, self.min_rect_size), self.max_rect_size)
        r = int(rng.integers(0, self.patch_size - rh + 1))
        c = int(rng.integers(0, self.patch_size - rw + 1))
        return (r, c, rh, rw)

    def sample(self, rng: Optional[np.random.Generator] = None) -> RectTest:
        """Draw one random RectTest."""
        rng = rng if rng is not None else np.random.default_rng()

        projection = str(rng.choice(self.projections))
        n = N_RECTS[projection]

        rects = [self._sample_rect(rng) for _ in range(n)]
        operators = [str(rng.choice(self.operators)) for _ in range(n)]

        # str()/int() casts keep numpy scalars (rng.choice yields np.str_) out
        # of the dataclass, so tests stay cleanly repr-able and picklable.
        if n == 1 or self.cross_channel:
            channels = [int(rng.choice(self.channels)) for _ in range(n)]
        else:
            ch = int(rng.choice(self.channels))
            channels = [ch, ch]

        return RectTest(
            projection=projection,
            rects=rects,
            operators=operators,
            channels=channels,
        )
