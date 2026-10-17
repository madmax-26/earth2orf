"""
Decision tree and Random Forest that operate directly on PolSAR patches.

The tree-growing loop follows Criminisi & Shotton (2013), with the node
test interface replaced by NodeTest / NodeTestSampler from node_tests.py.
Everything that touches PolSAR data lives in distances.py and node_tests.py;
the tree/forest code here is data-agnostic and could be reused for any
custom split function by swapping in a different sampler.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from node_tests import NodeTestSampler, NodeTest


# ---------------------------------------------------------------------------
# Gini impurity helpers
# ---------------------------------------------------------------------------

def gini(labels: np.ndarray, n_classes: int) -> float:
    """Gini impurity of a label set (Eq. 15)."""
    if len(labels) == 0:
        return 0.0
    counts = np.bincount(labels, minlength=n_classes)
    probs = counts / len(labels)
    return float(1.0 - np.sum(probs ** 2))


def impurity_drop(
    labels_left: np.ndarray,
    labels_right: np.ndarray,
    labels_parent: np.ndarray,
    n_classes: int,
) -> float:
    """Delta impurity (Eq. 14)."""
    n = len(labels_parent)
    nl, nr = len(labels_left), len(labels_right)
    drop = (gini(labels_parent, n_classes)
            - (nl / n) * gini(labels_left, n_classes)
            - (nr / n) * gini(labels_right, n_classes))
    return drop


# ---------------------------------------------------------------------------
# Tree node
# ---------------------------------------------------------------------------

@dataclass
class _Node:
    depth:      int
    is_leaf:    bool = False
    test:       Optional[NodeTest] = None
    left:       Optional["_Node"] = None
    right:      Optional["_Node"] = None
    # Leaf: class posterior P(y | node)
    posterior:  Optional[np.ndarray] = None
    # Training statistics, recorded at every node (visualization / inspection)
    n_samples:  int = 0
    class_counts: Optional[np.ndarray] = None


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------

class PolSARDecisionTree:
    """
    A single decision tree whose node tests operate directly on PolSAR patches
    (covariance/coherency matrix image patches).

    Parameters
    ----------
    sampler : NodeTestSampler
        Generates candidate node tests at each split.
    n_candidates : int
        Number of random tests evaluated per node (the paper uses thousands;
        start small and increase).
    max_depth : int
        Maximum tree depth (paper uses 50).
    min_samples_split : int
        Minimum samples at a node to attempt a split.
    n_classes : int
        Number of target classes.
    random_state : int or None
    """

    def __init__(
        self,
        sampler: NodeTestSampler,
        n_candidates: int = 50,
        max_depth: int = 50,
        min_samples_split: int = 5,
        n_classes: int = 5,
        random_state: Optional[int] = None,
    ):
        self.sampler = sampler
        self.n_candidates = n_candidates
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_classes = n_classes
        self.rng = np.random.default_rng(random_state)
        self.root: Optional[_Node] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, patches: list, labels: np.ndarray) -> "PolSARDecisionTree":
        """
        Grow the tree.

        Parameters
        ----------
        patches : list of ndarray, each shape (w, w, k, k)
            One patch per training pixel, centred on that pixel.
        labels : ndarray, shape (N,), dtype int
            Class labels in [0, n_classes).
        """
        labels = np.asarray(labels)
        indices = np.arange(len(patches))
        self.root = self._grow(patches, labels, indices, depth=0)
        return self

    def _grow(
        self,
        patches: list,
        labels: np.ndarray,
        indices: np.ndarray,
        depth: int,
    ) -> _Node:
        node = _Node(depth=depth)

        local_labels = labels[indices]
        unique = np.unique(local_labels)
        class_counts = np.bincount(local_labels, minlength=self.n_classes)
        posterior = class_counts / len(local_labels)
        node.n_samples = int(len(indices))
        node.class_counts = class_counts

        # Stopping criteria
        stop = (
            depth >= self.max_depth
            or len(indices) < self.min_samples_split
            or len(unique) == 1
        )

        if stop:
            node.is_leaf = True
            node.posterior = posterior
            return node

        # Find the best split among n_candidates random tests
        best_drop = -np.inf
        best_test = None
        best_left = None
        best_right = None

        local_patches = [patches[i] for i in indices]

        for _ in range(self.n_candidates):
            test = self.sampler.sample(self.rng)

            # Skip 1-pt tests if no reference pool is available
            if test.projection == "1pt" and test.reference is None:
                continue

            # Project all patches and find the median threshold
            try:
                values = np.array([test.project(p) for p in local_patches])
            except Exception:
                continue

            test.threshold = float(np.median(values))
            left_mask = values < test.threshold
            right_mask = ~left_mask

            if left_mask.sum() == 0 or right_mask.sum() == 0:
                continue

            drop = impurity_drop(
                local_labels[left_mask],
                local_labels[right_mask],
                local_labels,
                self.n_classes,
            )

            if drop > best_drop:
                best_drop = drop
                best_test = test
                best_left = indices[left_mask]
                best_right = indices[right_mask]

        # If no valid split found, make a leaf
        if best_test is None:
            node.is_leaf = True
            node.posterior = posterior
            return node

        node.test = best_test
        node.left = self._grow(patches, labels, best_left, depth + 1)
        node.right = self._grow(patches, labels, best_right, depth + 1)
        return node

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, patch: np.ndarray) -> np.ndarray:
        """
        Return class posterior for a single patch.

        Parameters
        ----------
        patch : ndarray, shape (w, w, k, k)

        Returns
        -------
        ndarray, shape (n_classes,)
        """
        node = self.root
        while not node.is_leaf:
            if node.test.evaluate(patch):
                node = node.left
            else:
                node = node.right
        return node.posterior

    def predict(self, patch: np.ndarray) -> int:
        """Return the most likely class label for a single patch."""
        return int(np.argmax(self.predict_proba(patch)))


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------

class PolSARRandomForest:
    """
    Random Forest over PolSAR image patches (Section 3 of the paper).

    Usage
    -----
    >>> from distances import ALL_DISTANCES
    >>> from node_tests import NodeTestSampler
    >>> from polsar_rf import PolSARRandomForest
    >>>
    >>> sampler = NodeTestSampler(patch_size=21, reference_pool=ref_pool)
    >>> rf = PolSARRandomForest(sampler, n_trees=30, n_classes=5)
    >>> rf.fit(train_patches, train_labels)
    >>> proba = rf.predict_proba_image(image)   # shape (H, W, n_classes)

    Parameters
    ----------
    sampler : NodeTestSampler
    n_trees : int
        Number of trees (paper uses 30).
    n_candidates : int
        Candidate tests per node.
    max_depth : int
    min_samples_split : int
    n_classes : int
    bagging_fraction : float
        Fraction of training samples per tree (1.0 = sample with replacement
        to the same size, i.e. bootstrap).
    random_state : int or None
    """

    def __init__(
        self,
        sampler: NodeTestSampler,
        n_trees: int = 30,
        n_candidates: int = 50,
        max_depth: int = 50,
        min_samples_split: int = 5,
        n_classes: int = 5,
        bagging_fraction: float = 1.0,
        random_state: Optional[int] = None,
    ):
        self.sampler = sampler
        self.n_trees = n_trees
        self.n_candidates = n_candidates
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.n_classes = n_classes
        self.bagging_fraction = bagging_fraction
        self.rng = np.random.default_rng(random_state)
        self.trees: list[PolSARDecisionTree] = []

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, patches: list, labels: np.ndarray) -> "PolSARRandomForest":
        """
        Train the forest.

        Parameters
        ----------
        patches : list of ndarray, each shape (w, w, k, k)
        labels  : ndarray, shape (N,), dtype int
        """
        labels = np.asarray(labels, dtype=int)
        N = len(patches)
        bag_size = max(1, int(N * self.bagging_fraction))

        self.trees = []
        for t in range(self.n_trees):
            seed = int(self.rng.integers(0, 2**31))
            # Bootstrap sample (with replacement)
            bag_idx = self.rng.integers(0, N, size=bag_size)
            bag_patches = [patches[i] for i in bag_idx]
            bag_labels = labels[bag_idx]

            tree = PolSARDecisionTree(
                sampler=self.sampler,
                n_candidates=self.n_candidates,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                n_classes=self.n_classes,
                random_state=seed,
            )
            print(f"  Tree {t + 1}/{self.n_trees} …", end="\r")
            tree.fit(bag_patches, bag_labels)
            self.trees.append(tree)

        print(f"  Forest of {self.n_trees} trees trained.          ")
        return self

    # ------------------------------------------------------------------
    # Prediction
    # ------------------------------------------------------------------

    def predict_proba(self, patch: np.ndarray) -> np.ndarray:
        """
        Eq. 20: average class posteriors across all trees.

        Parameters
        ----------
        patch : ndarray, shape (w, w, k, k)

        Returns
        -------
        ndarray, shape (n_classes,)
        """
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
        Classify every pixel of a PolSAR image.

        Parameters
        ----------
        image : ndarray, shape (H, W, k, k)
            Full PolSAR covariance matrix image.
        patch_size : int or None
            If None, uses sampler.patch_size.

        Returns
        -------
        ndarray, shape (H, W, n_classes)
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
        """
        Return hard class labels for every pixel.

        Returns
        -------
        ndarray, shape (H, W), dtype int
        """
        proba = self.predict_proba_image(image, patch_size)
        return np.argmax(proba, axis=-1)
