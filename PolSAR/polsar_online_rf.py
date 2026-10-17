"""
Online / batch-trained Random Forest for PolSAR patches.

Implements the batch-processing scheme of

    Hänsch & Hellwich, "Online Random Forests for Large-Scale Land-Use
    Classification from Polarimetric SAR Images", IGARSS 2019

on top of the node tests from node_tests.py (Hänsch & Hellwich 2017).
The key ideas, mirrored in the code:

  * Tree creation and tree training are decoupled (Sec. 2.2).
    - Tree training (leaf predictors = class histograms) is inherently
      online: every node updates its class_counts with every sample
      that passes through, across all batches.
    - Tree creation is made online by letting each fringe node act as a
      *growing leaf*: it samples its candidate tests once, then only
      accumulates the projected 1-D values and labels of incoming
      samples — never the patches themselves.
  * Once a growing leaf has observed tau samples, its split is fixed:
    threshold = median of the accumulated projections, best candidate
    by drop of impurity (Eqs. 14/15). The accumulated statistics are
    then deleted to free memory and two new growing leaves are created.
  * Bagging is replaced by online bagging (Oza 2005): each tree sees
    each sample k ~ Poisson(lambda) times.

Train with repeated partial_fit(patches, labels) calls — one call per
batch. Prediction works at any point in between (nodes that have not
split yet simply act as leaves).
"""

from __future__ import annotations

import numpy as np
from typing import Optional

from node_tests import NodeTestSampler, NodeTest
from polsar_rf import impurity_drop


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

class _OnlineNode:
    """
    A node of an online tree. Starts as a growing leaf that accumulates
    split statistics; becomes either a split node (after tau samples) or
    a permanent leaf (max depth reached / no valid split found).

    class_counts is updated by *every* sample routed through the node,
    so internal nodes keep meaningful statistics for visualization and
    leaves keep learning from all batches even after the topology is fixed.
    """

    __slots__ = ("depth", "is_leaf", "test", "left", "right",
                 "class_counts", "can_split", "n_seen",
                 "candidates", "cand_values", "acc_labels",
                 "n_split_attempts")

    def __init__(self, depth: int, n_classes: int, can_split: bool):
        self.depth = depth
        self.is_leaf = True
        self.test: Optional[NodeTest] = None
        self.left: Optional["_OnlineNode"] = None
        self.right: Optional["_OnlineNode"] = None
        self.class_counts = np.zeros(n_classes, dtype=np.int64)

        self.can_split = can_split
        self.n_seen = 0                     # samples accumulated towards tau
        self.candidates: Optional[list[NodeTest]] = None
        self.cand_values: Optional[list[list[float]]] = None
        self.acc_labels: Optional[list[int]] = None
        self.n_split_attempts = 0           # failed split rounds so far

    # -- interface shared with polsar_rf._Node (used by polsar_interface) --

    @property
    def n_samples(self) -> int:
        return int(self.class_counts.sum())

    @property
    def posterior(self) -> np.ndarray:
        total = self.class_counts.sum()
        if total == 0:
            return np.full(len(self.class_counts), 1.0 / len(self.class_counts))
        return self.class_counts / total


# ---------------------------------------------------------------------------
# Online decision tree
# ---------------------------------------------------------------------------

class OnlinePolSARDecisionTree:
    """
    Decision tree grown incrementally over batches of PolSAR patches.

    Parameters
    ----------
    sampler : NodeTestSampler
        Generates candidate node tests for each growing leaf.
    n_candidates : int
        Candidate tests per node.
    max_depth : int
        Maximum tree depth (paper uses 50).
    tau : int
        Number of samples a node accumulates before its split is fixed
        (paper Sec. 2.2). Larger tau = more reliable splits, slower growth.
        NOTE the depth this implies: median thresholds make every split
        ~50/50, so the samples reaching depth d shrink like N/2^d and a node
        only splits once it has seen tau of them — the achievable depth is
        roughly log2(N_total / tau), NOT max_depth. 3000 samples with tau=60
        top out around depth 6. Deeper trees need more stream (e.g. several
        shuffled epochs over a finite dataset), a smaller tau, or tau_decay.
    tau_decay : float
        Per-level multiplier on tau: a node at depth d splits after
        max(tau_min, tau * tau_decay**d) samples. Root splits stay
        conservative (they shape the whole tree) while deep nodes — which
        see exponentially fewer samples — split on less evidence. 1.0
        (default) keeps the classic constant-tau behavior. With decay γ the
        depth bound becomes log(N/tau) / log(2·γ): e.g. γ=0.5 removes the
        stream-imposed cap entirely (every level sees the same fraction of
        its incoming stream) until tau_min takes over.
    tau_min : int
        Floor for the decayed tau — below ~10 samples a median threshold and
        an impurity estimate are mostly noise, so growth stops there.
    n_classes : int
    split_retries : int
        How many times a growing leaf whose candidate round failed (every
        candidate degenerate or erroring on real data) redraws a fresh
        candidate set and accumulates another tau samples before giving up
        and becoming a permanent leaf. 0 restores the old behavior of dying
        on the first failed round.
    random_state : int or None
    """

    def __init__(
        self,
        sampler: NodeTestSampler,
        n_candidates: int = 50,
        max_depth: int = 50,
        tau: int = 100,
        tau_decay: float = 1.0,
        tau_min: int = 10,
        n_classes: int = 5,
        split_retries: int = 2,
        random_state: Optional[int] = None,
    ):
        if not 0.0 < tau_decay:
            raise ValueError(f"tau_decay must be positive, got {tau_decay}")
        if tau_min < 2:
            raise ValueError(f"tau_min must be at least 2, got {tau_min}")
        self.sampler = sampler
        self.n_candidates = n_candidates
        self.max_depth = max_depth
        self.tau = tau
        self.tau_decay = tau_decay
        self.tau_min = tau_min
        self.n_classes = n_classes
        self.split_retries = split_retries
        self.rng = np.random.default_rng(random_state)
        self.root = self._new_leaf(depth=0)

    def tau_at(self, depth: int) -> int:
        """Samples a node at this depth accumulates before its split is fixed.

        The tau_min floor only ever applies to the DECAYED value — a base tau
        below tau_min is honored as-is (back-compat with constant-tau usage)."""
        floor = min(self.tau, self.tau_min)
        return max(floor, int(round(self.tau * self.tau_decay ** depth)))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def partial_fit(
        self,
        patches: list,
        labels: np.ndarray,
        weights: Optional[np.ndarray] = None,
    ) -> "OnlinePolSARDecisionTree":
        """
        Learn from one batch.

        Parameters
        ----------
        patches : list of ndarray, each shape (w, w, k, k)
        labels  : ndarray, shape (N,), dtype int
        weights : ndarray, shape (N,), dtype int or None
            Per-sample multiplicities (online bagging). None → all 1.
            Samples with weight 0 are skipped.
        """
        labels = np.asarray(labels, dtype=int)
        if weights is None:
            weights = np.ones(len(patches), dtype=int)

        for patch, label, w in zip(patches, labels, weights):
            if w > 0:
                self._route(patch, int(label), int(w))
        return self

    def _route(self, patch: np.ndarray, label: int, weight: int) -> None:
        """Propagate one (weighted) sample down the tree, learning on the way."""
        node = self.root
        while True:
            node.class_counts[label] += weight
            if node.is_leaf:
                break
            node = node.left if node.test.evaluate(patch) else node.right

        if not node.can_split:
            return

        self._accumulate(node, patch, label, weight)
        if node.n_seen >= self.tau_at(node.depth):
            self._fix_split(node)

    def _new_leaf(self, depth: int) -> _OnlineNode:
        node = _OnlineNode(depth, self.n_classes,
                           can_split=depth < self.max_depth)
        return node

    def _accumulate(self, node: _OnlineNode, patch: np.ndarray,
                    label: int, weight: int) -> None:
        """Add one sample's projections to the node's split statistics."""
        if node.candidates is None:
            node.candidates = self._sample_candidates()
            node.cand_values = [[] for _ in node.candidates]
            node.acc_labels = []

        # Project once per candidate; a candidate that fails on real data
        # (e.g. singular matrix in a distance) is discarded entirely so the
        # remaining value lists stay aligned with acc_labels.
        for i in reversed(range(len(node.candidates))):
            try:
                value = node.candidates[i].project(patch)
            except Exception:
                del node.candidates[i]
                del node.cand_values[i]
                continue
            node.cand_values[i].extend([value] * weight)

        node.acc_labels.extend([label] * weight)
        node.n_seen += weight

    def _sample_candidates(self) -> list[NodeTest]:
        candidates = []
        for _ in range(self.n_candidates * 2):
            test = self.sampler.sample(self.rng)
            if test.projection == "1pt" and test.reference is None:
                continue
            candidates.append(test)
            if len(candidates) == self.n_candidates:
                break
        return candidates

    def _fix_split(self, node: _OnlineNode) -> None:
        """
        Define the node's split from the accumulated statistics (or turn it
        into a permanent leaf), then delete the statistics.
        """
        acc_labels = np.array(node.acc_labels)

        best_drop = -np.inf
        best_test = None
        best_mask = None

        for test, values in zip(node.candidates or [], node.cand_values or []):
            values = np.array(values)
            threshold = float(np.median(values))
            left_mask = values < threshold
            if left_mask.sum() == 0 or left_mask.sum() == len(values):
                continue
            drop = impurity_drop(acc_labels[left_mask], acc_labels[~left_mask],
                                 acc_labels, self.n_classes)
            if drop > best_drop:
                best_drop = drop
                best_test = test
                best_test.threshold = threshold
                best_mask = left_mask

        # Free the accumulated statistics either way (paper Sec. 2.2)
        node.candidates = node.cand_values = node.acc_labels = None

        if best_test is None:
            # Every candidate was degenerate (constant projections) or errored
            # on real data. Rather than dying on one unlucky draw, redraw a
            # fresh candidate set and accumulate another tau samples — up to
            # split_retries times.
            node.n_split_attempts += 1
            if node.n_split_attempts > self.split_retries:
                node.can_split = False  # permanent leaf
            else:
                node.n_seen = 0         # fresh round: candidates resampled on
            return                      # the next _accumulate call

        node.test = best_test
        node.is_leaf = False
        node.can_split = False
        node.left = self._new_leaf(node.depth + 1)
        node.right = self._new_leaf(node.depth + 1)

        # Seed the children's histograms with the split of the accumulated
        # labels — that information is available even though the patches are
        # not. Note that a child's own seed can never be propagated further
        # down (its patches were discarded), so across the tree the counts
        # satisfy parent >= left + right rather than strict equality.
        node.left.class_counts = np.bincount(
            acc_labels[best_mask], minlength=self.n_classes).astype(np.int64)
        node.right.class_counts = np.bincount(
            acc_labels[~best_mask], minlength=self.n_classes).astype(np.int64)

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, patch: np.ndarray) -> np.ndarray:
        """Class posterior for a single patch, shape (n_classes,)."""
        node = self.root
        while not node.is_leaf:
            node = node.left if node.test.evaluate(patch) else node.right
        return node.posterior

    def predict(self, patch: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(patch)))


# ---------------------------------------------------------------------------
# Online Random Forest
# ---------------------------------------------------------------------------

class OnlinePolSARRandomForest:
    """
    Random Forest over PolSAR patches, trained incrementally in batches.

    Usage
    -----
    >>> rf = OnlinePolSARRandomForest(sampler, n_trees=30, n_classes=5, tau=200)
    >>> for batch_patches, batch_labels in batches:
    ...     rf.partial_fit(batch_patches, batch_labels)
    >>> proba = rf.predict_proba(patch)

    Parameters
    ----------
    sampler : NodeTestSampler
    n_trees : int
    n_candidates : int
        Candidate tests per node.
    max_depth : int
    tau : int
        Samples a node accumulates before its split is fixed. Together
        with the batch size this controls how fast the trees grow
        (paper Sec. 2.2: batches should cover the data before the first
        splits are made, i.e. tau >= batch size is a sensible default).
        Median thresholds give ~balanced splits, so achievable depth is
        about log2(total samples streamed / tau) — stream several shuffled
        epochs of a finite dataset, or use tau_decay, to grow deeper trees.
    tau_decay : float
        Per-level multiplier on tau: a node at depth d splits after
        max(tau_min, tau * tau_decay**d) samples — conservative near the
        root, cheaper further down where samples arrive exponentially more
        rarely. 1.0 (default) = classic constant tau.
    tau_min : int
        Floor for the decayed tau (median splits on fewer than ~10 samples
        are mostly noise).
    n_classes : int
    bagging_lambda : float
        Lambda of the Poisson distribution for online bagging (Oza 2005).
        Each tree sees each sample k ~ Poisson(lambda) times; 1.0
        emulates the classic bootstrap.
    split_retries : int
        Fresh candidate rounds a node gets when every candidate of a round
        failed, before it becomes a permanent leaf (see the tree class).
    random_state : int or None
    """

    def __init__(
        self,
        sampler: NodeTestSampler,
        n_trees: int = 30,
        n_candidates: int = 50,
        max_depth: int = 50,
        tau: int = 100,
        tau_decay: float = 1.0,
        tau_min: int = 10,
        n_classes: int = 5,
        bagging_lambda: float = 1.0,
        split_retries: int = 2,
        random_state: Optional[int] = None,
    ):
        self.sampler = sampler
        self.n_trees = n_trees
        self.n_classes = n_classes
        self.bagging_lambda = bagging_lambda
        self.rng = np.random.default_rng(random_state)
        self.n_samples_seen = 0
        self.n_batches_seen = 0

        self.trees = [
            OnlinePolSARDecisionTree(
                sampler=sampler,
                n_candidates=n_candidates,
                max_depth=max_depth,
                tau=tau,
                tau_decay=tau_decay,
                tau_min=tau_min,
                n_classes=n_classes,
                split_retries=split_retries,
                random_state=int(self.rng.integers(0, 2**31)),
            )
            for _ in range(n_trees)
        ]

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def partial_fit(self, patches: list, labels: np.ndarray) -> "OnlinePolSARRandomForest":
        """
        Learn from one batch of samples. Call repeatedly with new batches;
        the forest can be used for prediction at any point in between.

        Parameters
        ----------
        patches : list of ndarray, each shape (w, w, k, k)
        labels  : ndarray, shape (N,), dtype int
        """
        labels = np.asarray(labels, dtype=int)
        for t, tree in enumerate(self.trees):
            weights = self.rng.poisson(self.bagging_lambda, size=len(patches))
            print(f"  Batch {self.n_batches_seen + 1}: tree {t + 1}/{self.n_trees} …", end="\r")
            tree.partial_fit(patches, labels, weights)

        self.n_samples_seen += len(patches)
        self.n_batches_seen += 1
        print(f"  Batch {self.n_batches_seen} done "
              f"({self.n_samples_seen} samples seen total).          ")
        return self

    # ------------------------------------------------------------------
    # Prediction (identical to the offline forest, Eq. 20)
    # ------------------------------------------------------------------

    def predict_proba(self, patch: np.ndarray) -> np.ndarray:
        """Average class posterior across all trees, shape (n_classes,)."""
        posteriors = np.stack([t.predict_proba(patch) for t in self.trees])
        return posteriors.mean(axis=0)

    def predict(self, patch: np.ndarray) -> int:
        return int(np.argmax(self.predict_proba(patch)))

    def predict_proba_image(
        self,
        image: np.ndarray,
        patch_size: Optional[int] = None,
    ) -> np.ndarray:
        """
        Classify every pixel of a (H, W, k, k) PolSAR image.
        Returns ndarray of shape (H, W, n_classes).
        """
        w = patch_size or self.sampler.patch_size
        half = w // 2
        H, W = image.shape[:2]
        out = np.zeros((H, W, self.n_classes), dtype=np.float32)

        for r in range(half, H - half):
            for c in range(half, W - half):
                patch = image[r - half:r + half + 1, c - half:c + half + 1]
                out[r, c] = self.predict_proba(patch)
            print(f"  Row {r - half + 1}/{H - 2 * half}", end="\r")

        print()
        return out

    def predict_image(self, image: np.ndarray, patch_size: Optional[int] = None) -> np.ndarray:
        """Hard class labels for every pixel, shape (H, W), dtype int."""
        proba = self.predict_proba_image(image, patch_size)
        return np.argmax(proba, axis=-1)
