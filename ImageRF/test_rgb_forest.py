"""
Sections 8-12 of the ImageRF assertion suite: forest behaviour on real AID
data, the TreeFlow round-trip, and whole-image prediction.

Imported and run by test_rgb_rf.py; not meant to be run standalone (it shares
that module's check()/failure counter).
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _check():
    """The shared check()/counter from the main suite."""
    import test_rgb_rf
    return test_rgb_rf.check


def _count_nodes(node) -> int:
    return 1 + (_count_nodes(node.left) + _count_nodes(node.right)
                if not node.is_leaf else 0)


def _max_depth(node) -> int:
    best, stack = 0, [node]
    while stack:
        n = stack.pop()
        best = max(best, n.depth)
        if not n.is_leaf:
            stack.extend([n.left, n.right])
    return best


def _walk(node):
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        if not n.is_leaf:
            stack.extend([n.left, n.right])


def run_forest_sections(rng) -> None:
    check = _check()

    from rgb_datasets import (load_scene_dataset, train_test_split_scenes,
                              sample_patch_set, prepare_patch_stream)
    from rgb_node_tests import RectTestSampler
    from rgb_forest import ImageOnlineRandomForest
    from rgb_interface import evaluate_image_rf

    PATCH = 48          # smaller than the demo's 64 to keep the suite quick
    N_CLASSES = 10

    # ── 10. Image-level split (checked first: everything else depends on it) ──
    print("Dataset + image-level split:")
    full = load_scene_dataset("AID")
    keep = set(full.class_names[:N_CLASSES])
    idx = [i for i, l in enumerate(full.labels) if full.class_names[l] in keep]
    from rgb_datasets import SceneDataset
    ds = SceneDataset("AID-10", [full.paths[i] for i in idx],
                      np.array([full.labels[i] for i in idx], dtype=int),
                      full.class_names[:N_CLASSES])
    check(ds.n_classes == N_CLASSES and len(ds) > 0,
          f"loaded {len(ds)} images over {ds.n_classes} classes")

    train_ds, test_ds = train_test_split_scenes(ds, test_frac=0.2, rng=rng)
    overlap = set(train_ds.paths) & set(test_ds.paths)
    check(not overlap, f"no image appears in both splits ({len(overlap)} overlaps)")
    check(set(np.unique(train_ds.labels)) == set(range(N_CLASSES))
          and set(np.unique(test_ds.labels)) == set(range(N_CLASSES)),
          "both splits contain every class")
    check(len(train_ds) + len(test_ds) == len(ds), "split is a partition")

    # ── 8. The forest actually splits ────────────────────────────────────────
    print("Forest growth on real AID patches:")
    sampler = RectTestSampler(patch_size=PATCH)
    rf = ImageOnlineRandomForest(sampler=sampler, n_trees=8, n_candidates=100,
                                 max_depth=24, tau=150, tau_decay=0.6, tau_min=20,
                                 n_classes=N_CLASSES, random_state=42)

    n_seen = 0
    for patches, labels in prepare_patch_stream(train_ds, patch_size=PATCH,
                                                per_image=3, batch_size=1500,
                                                epochs=2, rng=rng):
        rf.partial_fit(patches, labels)
        n_seen += len(patches)
    print(f"  (trained on {n_seen} patches)")

    sizes = [_count_nodes(t.root) for t in rf.trees]
    depths = [_max_depth(t.root) for t in rf.trees]
    print(f"  tree sizes {sizes}")
    print(f"  tree depths {depths}")
    check(all(s > 1 for s in sizes), "every tree split at least once")
    check(max(depths) >= 4, f"deepest tree reaches depth {max(depths)} (>= 4)")

    finite = all(np.isfinite(n.test.threshold)
                 for t in rf.trees for n in _walk(t.root) if not n.is_leaf)
    check(finite, "every split node has a finite threshold")

    conserved = True
    for t in rf.trees:
        for n in _walk(t.root):
            if not n.is_leaf:
                if n.left.class_counts.sum() + n.right.class_counts.sum() > n.class_counts.sum():
                    conserved = False
    check(conserved, "child counts never exceed the parent (documented seeding invariant)")

    # ── 9. Beats chance ──────────────────────────────────────────────────────
    print("Accuracy on held-out AID images:")
    eval_patches, eval_labels = sample_patch_set(test_ds, n=1000, patch_size=PATCH, rng=rng)
    ev = evaluate_image_rf(rf, eval_patches, eval_labels)
    acc = ev["forest_accuracy"]
    per_tree = float(np.mean([m["accuracy"] for m in ev["tree_metrics"]]))
    chance = 1.0 / N_CLASSES
    print(f"  forest {acc:.2%} | mean tree {per_tree:.2%} | chance {chance:.2%}")
    check(acc > 0.35, f"forest accuracy {acc:.2%} beats the 35% smoke-test bar (chance {chance:.0%})")
    check(acc > per_tree, "the ensemble beats its average member")

    # ── 12. Whole-image prediction ───────────────────────────────────────────
    print("predict_proba_image:")
    from rgb_datasets import load_image
    tile = load_image(test_ds.paths[0])
    small = tile[:96, :96]                       # keep the O(pixels) loop short
    proba = rf.predict_proba_image(small)
    check(proba.shape == (96, 96, N_CLASSES), f"output shape {proba.shape}")
    half = PATCH // 2
    inner = proba[half:-half, half:-half]
    check(inner.size > 0 and float(inner.std()) > 0,
          "predictions vary inside the valid border")

    return rf, ds, eval_patches, eval_labels
