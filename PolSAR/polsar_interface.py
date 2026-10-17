"""
TreeFlow PolSAR interface — converts a PolSAR Random Forest to the TreeFlow Forest format.

Works with both forest variants:
  * OnlinePolSARRandomForest (polsar_online_rf.py) — trained in batches.
  * PolSARRandomForest (polsar_rf.py) — trained offline in one fit() call.

The trees record per-node statistics (sample count and per-class counts)
during training, which map directly onto the TreeFlow Node fields. Nodes are
emitted in DFS preorder (parent, left subtree, right subtree) as required
by the frontend's buildHierarchy.

A TreeFlow iteration shows how the forest improved as it learned new data, so
push one snapshot after every batch of the online forest:

    from node_tests import NodeTestSampler
    from polsar_online_rf import OnlinePolSARRandomForest
    from polsar_interface import convert_polsar_rf
    from treeflow import TreeFlow

    rf = OnlinePolSARRandomForest(sampler, n_trees=30, n_classes=5, tau=200)
    treeflow = TreeFlow(port=8000)

    for batch_patches, batch_labels in batches:
        rf.partial_fit(batch_patches, batch_labels)
        treeflow.push(convert_polsar_rf(rf, eval_patches=test_patches,
                                   eval_labels=test_labels))

For the offline forest there is no learning over time — push the single
fitted forest as one iteration:

    rf = PolSARRandomForest(sampler, n_trees=30, n_classes=5)
    rf.fit(train_patches, train_labels)
    treeflow.push(convert_polsar_rf(rf))
"""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING, Optional

import numpy as np

# lib.py / treeflow.py live in the Library directory (same pattern as Backend/backend.py)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Library"))

from lib import Forest, Node, Tree  # noqa: E402

if TYPE_CHECKING:
    from typing import Union
    from polsar_rf import PolSARDecisionTree, PolSARRandomForest
    from polsar_online_rf import OnlinePolSARDecisionTree, OnlinePolSARRandomForest

    AnyForest = Union[PolSARRandomForest, OnlinePolSARRandomForest]
    AnyTree = Union[PolSARDecisionTree, OnlinePolSARDecisionTree]


# ── Public converter ──────────────────────────────────────────────────────────

def convert_polsar_rf(
    rf: "AnyForest",
    class_labels: Optional[list] = None,
    eval_patches: Optional[list] = None,
    eval_labels: Optional[np.ndarray] = None,
    n_trees: Optional[int] = None,
    tree_metrics: Optional[list] = None,
) -> Forest:
    """Convert a (partially) trained PolSAR Random Forest to a TreeFlow Forest.

    Accepts both OnlinePolSARRandomForest and the offline PolSARRandomForest.

    class_labels: display names for the classes. Defaults to "0".."n_classes-1".
    eval_patches / eval_labels: optional held-out set. When both are provided,
                  each tree's accuracy on it is added to that tree's metrics.
                  Routing the eval set is the expensive part of a push — if you
                  also want the forest-level accuracy, DON'T route it a second
                  time via rf.predict: use evaluate_polsar_rf once and pass its
                  "tree_metrics" here instead.
    n_trees:      only convert the first n trees (None → all).
    tree_metrics: precomputed per-tree metrics dicts (one per converted tree,
                  e.g. from evaluate_polsar_rf). Takes precedence over
                  eval_patches/eval_labels.
    """
    members = rf.trees if n_trees is None else rf.trees[:n_trees]
    labels = [str(c) for c in (class_labels or range(rf.n_classes))]
    if tree_metrics is None and eval_patches is not None and eval_labels is not None:
        tree_metrics = evaluate_polsar_rf(rf, eval_patches, eval_labels, n_trees=n_trees)["tree_metrics"]

    trees: dict[int, Tree] = {}
    for i, ptree in enumerate(members):
        metrics = dict(tree_metrics[i]) if tree_metrics is not None and i < len(tree_metrics) else {}
        trees[i] = _tree_from_polsar_tree(ptree, metrics)

    return Forest(trees=trees, class_labels=labels)


def evaluate_polsar_rf(
    rf: "AnyForest",
    patches: list,
    labels: np.ndarray,
    n_trees: Optional[int] = None,
) -> dict:
    """Per-tree AND forest accuracy from ONE routing pass over the eval set.

    Each patch is routed once per tree and its class posterior kept; per-tree
    accuracy comes from each tree's own argmax, forest accuracy from the argmax
    of the averaged posteriors (exactly rf.predict's rule). Compared to calling
    convert_polsar_rf(eval_patches=...) and then rf.predict per patch, this
    halves the eval routing — by far the dominant cost of a push.

        ev = evaluate_polsar_rf(rf, eval_patches, eval_labels)
        treeflow.push(convert_polsar_rf(rf, tree_metrics=ev["tree_metrics"]))
        print(f"forest accuracy {ev['forest_accuracy']:.2%}")

    Returns {"tree_metrics": [{"accuracy": float} per tree],
             "forest_accuracy": float}.

    The two are NOT the same scale and should not be compared as if one were a
    sanity check on the other. "tree_metrics" is each tree judged alone, while
    "forest_accuracy" averages the posteriors first — so the forest normally
    scores well above the mean tree (e.g. 22% forest vs ~10% per tree on 30-class
    AID, against 3.3% chance). That gap is the ensemble working, not an error.
    Only "forest_accuracy" is comparable to a published model score; the per-tree
    values go into Tree.metrics and are labelled per-tree in the frontend.
    """
    members = rf.trees if n_trees is None else rf.trees[:n_trees]
    labels = np.asarray(labels, dtype=int)
    posteriors = np.zeros((len(members), len(patches), rf.n_classes))
    for t, ptree in enumerate(members):
        for i, patch in enumerate(patches):
            posteriors[t, i] = ptree.predict_proba(patch)

    tree_preds = posteriors.argmax(axis=2)                      # (T, N)
    forest_preds = posteriors.mean(axis=0).argmax(axis=1)       # (N,)
    return {
        "tree_metrics": [
            {"accuracy": float((tree_preds[t] == labels).mean())}
            for t in range(len(members))
        ],
        "forest_accuracy": float((forest_preds == labels).mean()),
    }


# ── Shared helpers ────────────────────────────────────────────────────────────

def _preorder_nodes(ptree: "AnyTree") -> list:
    """The tree's nodes in DFS preorder (parent, left subtree, right subtree).

    Single source of truth for node ordering: both the converter (which emits
    id = preorder position) and the reference router (which records those same
    ids) consume this walk, so they cannot diverge.
    """
    if ptree.root is None:
        return []
    order = []
    stack = [ptree.root]
    while stack:
        n = stack.pop()
        order.append(n)
        if not n.is_leaf:
            stack.append(n.right)   # push right first → left popped first (preorder)
            stack.append(n.left)
    return order


def _tree_from_polsar_tree(ptree: "AnyTree", metrics: dict) -> Tree:
    """Build a TreeFlow Tree from a fitted PolSAR decision tree (DFS preorder)."""
    nodes: list[Node] = []
    height = 0
    for n in _preorder_nodes(ptree):
        counts = getattr(n, "class_counts", None)
        if counts is None:
            raise ValueError(
                "Tree node has no stored class statistics — the forest was trained "
                "with an older polsar_rf.py. Retrain with the current version, which "
                "records n_samples / class_counts at every node during fit()."
            )
        nodes.append(Node(
            id=len(nodes),
            depth=int(n.depth),
            n_node_samples=int(n.n_samples),
            impurity=_gini_from_counts(counts),
            value=[int(c) for c in counts],
        ))
        height = max(height, int(n.depth))

    return Tree(nodes=nodes, height=height, n_nodes=len(nodes), metrics=metrics)


def _gini_from_counts(counts: np.ndarray) -> float:
    total = counts.sum()
    if total == 0:
        return 0.0
    probs = counts / total
    return float(1.0 - np.sum(probs ** 2))


# ── Reference routing (TreeFlow visual references) ─────────────────────────────────
#
# TreeFlow's reference feature (treeflow.add_reference / treeflow.set_reference_router) needs
# a model-specific router that maps samples to per-tree node paths. This is the
# PolSAR implementation; the node ids it records are the DFS-preorder positions
# _tree_from_polsar_tree emits, guaranteed by sharing _preorder_nodes.

def _walk_patch(ptree: "AnyTree", patch: np.ndarray, pos: dict) -> list:
    """Root→leaf preorder-id path of one patch. A NodeTest that cannot be
    evaluated on this patch (e.g. singular matrix) truncates the path at the
    current node."""
    node = ptree.root
    if node is None:
        return []
    path = [pos[id(node)]]
    while not node.is_leaf:
        try:
            go_left = node.test.evaluate(patch)
        except Exception:
            break
        node = node.left if go_left else node.right
        path.append(pos[id(node)])
    return path


def route_patch(ptree: "AnyTree", patch: np.ndarray) -> list:
    """Route a single patch through one tree → list of preorder node ids."""
    pos = {id(n): i for i, n in enumerate(_preorder_nodes(ptree))}
    return _walk_patch(ptree, patch, pos)


def make_polsar_router(rf: "AnyForest", n_trees: Optional[int] = None):
    """Router for treeflow.set_reference_router, closing over the live forest.

    n_trees must mirror the value passed to convert_polsar_rf — the router only
    routes through (and the snapshot only contains) those trees.

        treeflow.set_reference_router(make_polsar_router(rf))
        treeflow.add_reference(patch, image=patch_to_rgb(patch), label="water")

    Returns router(samples) -> [{tree_id: [node_id, ...]}, ...] (input order).
    """
    def router(samples: list) -> list:
        members = rf.trees if n_trees is None else rf.trees[:n_trees]
        # One preorder map per tree per push — trees mutate between pushes.
        tree_pos = [
            {id(n): i for i, n in enumerate(_preorder_nodes(t))}
            for t in members
        ]
        return [
            {tid: _walk_patch(t, s, pos)
             for tid, (t, pos) in enumerate(zip(members, tree_pos))}
            for s in samples
        ]
    return router


def patch_to_rgb(patch: np.ndarray, low_pct: float = 2.0, high_pct: float = 98.0) -> np.ndarray:
    """Displayable uint8 RGB rendition of a PolSAR covariance patch.

    patch: (h, w, k, k) complex covariance/coherency matrices. The first three
    diagonal backscatter intensities become R, G, B (replicated when k < 3),
    each log-scaled and percentile-normalized independently. This is the
    default `image=` argument for treeflow.add_reference — other domains supply
    their own rendition.
    """
    p = np.asarray(patch)
    if p.ndim != 4:
        raise ValueError(f"expected patch of shape (h, w, k, k), got {p.shape}")
    h, w, k = p.shape[0], p.shape[1], p.shape[2]
    chans = []
    for i in range(3):
        d = min(i, k - 1)
        v = np.log10(np.abs(p[:, :, d, d]).astype(np.float64) + 1e-12)
        lo, hi = np.percentile(v, [low_pct, high_pct])
        if hi - lo < 1e-12:
            chans.append(np.zeros((h, w), dtype=np.uint8))
        else:
            chans.append((np.clip((v - lo) / (hi - lo), 0.0, 1.0) * 255).astype(np.uint8))
    return np.stack(chans, axis=-1)
