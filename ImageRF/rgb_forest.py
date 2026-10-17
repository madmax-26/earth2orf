"""
Online random forest over RGB image patches.

Everything that matters — online bagging (Oza 2005), growing leaves that
accumulate split statistics until tau samples, the depth-dependent tau schedule
(tau_decay / tau_min), the median split rule, split_retries, and the per-node
class histograms the TreeFlow frontend visualizes — is INHERITED VERBATIM from
PolSAR/polsar_online_rf.py, which is reused byte-for-byte unchanged.

This module adds exactly two things:
  * patches are PreparedPatch objects (integral images built once per patch and
    reused across every tree, node and candidate), and
  * the node tests are rectangle comparisons on RGB instead of matrix distances
    on polarimetric covariances.

The forest core can be reused untouched because it never inspects a patch: it
passes it straight to test.project / test.evaluate. Its one PolSAR-specific
line, the `test.projection == "1pt" and test.reference is None` guard in
_sample_candidates, is inert for RectTest (see rgb_node_tests.PROJECTIONS).
"""

from __future__ import annotations

import os
import sys

# PolSAR is consumed as a sys.path directory, not a package — the same pattern
# polsar_interface.py uses for Library. ImageRF's own modules are all prefixed
# rgb_/example_rgb_ so they can never shadow node_tests.py, distances.py,
# example.py or polsar_rf.py on the shared flat path.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PolSAR"))

from polsar_online_rf import (  # noqa: E402
    OnlinePolSARDecisionTree,
    OnlinePolSARRandomForest,
)

from rgb_patch import as_prepared  # noqa: E402
from rgb_node_tests import RectTestSampler  # noqa: E402


class ImageOnlineDecisionTree(OnlinePolSARDecisionTree):
    """Online decision tree over RGB patches.

    Identical to its PolSAR base except that the prediction entry point accepts
    a raw ndarray. That guard matters for performance, not correctness:
    polsar_interface.evaluate_polsar_rf calls tree.predict_proba(patch)
    directly, and without preparing once here a raw array would be rebuilt into
    a PreparedPatch inside EVERY RectTest.project along the root-to-leaf path.
    """

    def predict_proba(self, patch):
        return super().predict_proba(as_prepared(patch))


class ImageOnlineRandomForest(OnlinePolSARRandomForest):
    """Random Forest over RGB patches, trained incrementally in batches.

    Usage
    -----
    >>> sampler = RectTestSampler(patch_size=64)
    >>> rf = ImageOnlineRandomForest(sampler, n_trees=30, n_classes=30, tau=200)
    >>> for patches, labels in stream:          # PreparedPatch objects
    ...     rf.partial_fit(patches, labels)
    >>> proba = rf.predict_proba(patch)

    Parameters are the base class's; see polsar_online_rf for tau / tau_decay /
    tau_min / split_retries / bagging_lambda. Two RGB-specific notes:

      * n_candidates can be far higher than PolSAR's (100+ rather than 15),
        because a rectangle projection is four integer lookups instead of two
        matrix logarithms. A large candidate set is also what compensates for
        the median-only split rule and for uint8 ties on small rectangles.
      * with median splits the reachable depth is about
        log(total samples streamed / tau) / log(2 * tau_decay), so the SIZE OF
        THE STREAM — not max_depth — is what bounds tree growth. Stream more
        epochs to grow deeper trees.
    """

    def __init__(self, sampler: RectTestSampler, n_trees: int = 30,
                 n_candidates: int = 100, max_depth: int = 24, tau: int = 200,
                 tau_decay: float = 0.6, tau_min: int = 20, n_classes: int = 5,
                 bagging_lambda: float = 1.0, split_retries: int = 2,
                 random_state=None):
        super().__init__(
            sampler=sampler, n_trees=n_trees, n_candidates=n_candidates,
            max_depth=max_depth, tau=tau, tau_decay=tau_decay, tau_min=tau_min,
            n_classes=n_classes, bagging_lambda=bagging_lambda,
            split_retries=split_retries, random_state=random_state,
        )
        # The base __init__ hardcodes OnlinePolSARDecisionTree, so rebuild the
        # members with the Image subclass. Seeds still come from self.rng, so
        # construction stays deterministic for a given random_state.
        self.trees = [
            ImageOnlineDecisionTree(
                sampler=sampler, n_candidates=n_candidates, max_depth=max_depth,
                tau=tau, tau_decay=tau_decay, tau_min=tau_min,
                n_classes=n_classes, split_retries=split_retries,
                random_state=int(self.rng.integers(0, 2 ** 31)),
            )
            for _ in range(n_trees)
        ]

    def partial_fit(self, patches: list, labels):
        """Learn from one batch. as_prepared is idempotent, so handing over
        already-prepared patches (the normal path, see rgb_datasets) is free."""
        return super().partial_fit([as_prepared(p) for p in patches], labels)

    def predict_proba(self, patch):
        return super().predict_proba(as_prepared(patch))

    # predict / predict_image need no override — they call predict_proba.
    # predict_proba_image needs none either: it reads sampler.patch_size and
    # image.shape[:2] and slices two axes, so (H, W, 3) already works, and each
    # slice flows through the overridden predict_proba above.
