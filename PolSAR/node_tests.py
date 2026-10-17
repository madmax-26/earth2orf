"""
Patch operators (phi) and node test projections (Eqs. 16-19).

A PolSAR image patch is represented as a numpy array of shape
(w, w, k, k) where w is the spatial patch size and k is the
number of polarimetric channels.

Each node test is fully described by a NodeTest dataclass which
stores all randomly sampled parameters. This makes tests serialisable
and inspectable — the RF is not a black box.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from typing import Callable, Optional
from distances import ALL_DISTANCES


# ---------------------------------------------------------------------------
# Region operator (phi) — Eq. 16
# ---------------------------------------------------------------------------

def apply_operator(region: np.ndarray, op: str) -> np.ndarray:
    """
    Apply one of the four operators to a spatial region of covariance matrices.

    Parameters
    ----------
    region : ndarray, shape (rw, rw, k, k)
        Sub-patch of covariance matrices.
    op : str
        One of 'center', 'mean', 'min_span', 'max_span'.

    Returns
    -------
    ndarray, shape (k, k)
        A single covariance matrix.
    """
    rw = region.shape[0]
    mid = rw // 2

    if op == "center":
        return region[mid, mid]
    elif op == "mean":
        return region.reshape(-1, *region.shape[2:]).mean(axis=0)
    elif op == "min_span":
        # One vectorized trace over the whole region (a Python-level _span loop
        # per cell was a hot spot: this runs at every node evaluation).
        spans = np.trace(region, axis1=2, axis2=3).real
        idx = np.unravel_index(np.argmin(spans), spans.shape)
        return region[idx]
    elif op == "max_span":
        spans = np.trace(region, axis1=2, axis2=3).real
        idx = np.unravel_index(np.argmax(spans), spans.shape)
        return region[idx]
    else:
        raise ValueError(f"Unknown operator: {op}")


# ---------------------------------------------------------------------------
# NodeTest dataclass
# ---------------------------------------------------------------------------

OPERATORS = ["center", "mean", "min_span", "max_span"]
PROJECTIONS = ["1pt", "2pt", "4pt"]


@dataclass
class NodeTest:
    """
    One randomly sampled node test. Stores all parameters needed to
    evaluate the test on a new image patch, and to inspect afterwards
    which tests the forest preferred.

    Fields
    ------
    projection : str
        '1pt', '2pt', or '4pt' (Eqs. 17-19).
    distance_name : str
        Key into distances.ALL_DISTANCES.
    regions : list of (row_offset, col_offset, size) tuples
        Positions and sizes of the 1, 2, or 4 regions inside the patch.
    operators : list of str
        Which operator to apply to each region.
    reference : ndarray or None
        For 1-pt projection: a covariance matrix sampled from the image.
    threshold : float
        Split threshold h. Set to the median of projected values during
        training (call fit_threshold).
    """
    projection:     str
    distance_name:  str
    regions:        list
    operators:      list
    reference:      Optional[np.ndarray] = field(default=None, repr=False)
    threshold:      float = 0.0

    def _get_region(self, patch: np.ndarray, r_idx: int) -> np.ndarray:
        ro, co, sz = self.regions[r_idx]
        return patch[ro:ro + sz, co:co + sz]

    def project(self, patch: np.ndarray) -> float:
        """
        Compute the scalar projection value for a single patch.

        Parameters
        ----------
        patch : ndarray, shape (w, w, k, k)

        Returns
        -------
        float
        """
        dist_fn = ALL_DISTANCES[self.distance_name]

        C = [apply_operator(self._get_region(patch, i), self.operators[i])
             for i in range(len(self.regions))]

        if self.projection == "1pt":
            return dist_fn(C[0], self.reference)
        elif self.projection == "2pt":
            return dist_fn(C[0], C[1])
        elif self.projection == "4pt":
            return dist_fn(C[0], C[1]) - dist_fn(C[2], C[3])
        else:
            raise ValueError(f"Unknown projection: {self.projection}")

    def evaluate(self, patch: np.ndarray) -> bool:
        """Return True → left child, False → right child."""
        return self.project(patch) < self.threshold

    def fit_threshold(self, patches: list) -> None:
        """Set threshold to the median of projected values over training patches."""
        values = np.array([self.project(p) for p in patches])
        self.threshold = float(np.median(values))


# ---------------------------------------------------------------------------
# Random test generator
# ---------------------------------------------------------------------------

class NodeTestSampler:
    """
    Samples random NodeTest instances according to the paper's scheme.

    Parameters
    ----------
    patch_size : int
        Spatial size of the image patch (w).
    region_size_min, region_size_max : int
        Range [w_min, w_max] for the sub-region side length.
    max_offset : int
        Maximum distance from patch centre (a_max).
    reference_pool : list of ndarray, shape (k, k)
        Pool of covariance matrices sampled from the training image(s).
        Used for 1-pt projections. Build it with example.build_reference_pool,
        which accepts one image or a list of images (different scenes /
        parts of a city or landscape).
    distance_names : list of str or None
        Which distances to sample from. None → use all.
    """

    def __init__(
        self,
        patch_size: int = 21,
        region_size_min: int = 3,
        region_size_max: int = 10,
        max_offset: int = 10,
        reference_pool: Optional[list] = None,
        distance_names: Optional[list] = None,
    ):
        self.patch_size = patch_size
        self.rsz_min = region_size_min
        self.rsz_max = region_size_max
        self.max_offset = max_offset
        self.reference_pool = reference_pool or []
        self.distance_names = distance_names or list(ALL_DISTANCES.keys())

    def _sample_region(self, rng: np.random.Generator) -> tuple:
        """Sample (row_offset, col_offset, size) for one region inside the patch."""
        sz = int(rng.integers(self.rsz_min, self.rsz_max + 1))
        center = self.patch_size // 2
        max_shift = self.max_offset
        ro = int(rng.integers(max(0, center - max_shift),
                               min(self.patch_size - sz, center + max_shift) + 1))
        co = int(rng.integers(max(0, center - max_shift),
                               min(self.patch_size - sz, center + max_shift) + 1))
        return (ro, co, sz)

    def sample(self, rng: Optional[np.random.Generator] = None) -> NodeTest:
        """Draw one random NodeTest."""
        rng = rng or np.random.default_rng()
        proj = rng.choice(PROJECTIONS)
        dist_name = rng.choice(self.distance_names)
        n_regions = {"1pt": 1, "2pt": 2, "4pt": 4}[proj]
        regions = [self._sample_region(rng) for _ in range(n_regions)]
        operators = [rng.choice(OPERATORS) for _ in range(n_regions)]
        reference = None
        if proj == "1pt" and self.reference_pool:
            reference = self.reference_pool[int(rng.integers(len(self.reference_pool)))]
        return NodeTest(
            projection=proj,
            distance_name=dist_name,
            regions=regions,
            operators=operators,
            reference=reference,
        )
