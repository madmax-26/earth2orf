"""
End-to-end demo: train an online PolSAR RF batch by batch and watch it
learn in the TreeFlow frontend.

Run the backend:
    python example_treeflow.py                 # serves on http://localhost:8000

Then start the frontend:
    cd ../Frontend && npm run dev         # http://localhost:5173

One snapshot is pushed after every partial_fit batch, so each TreeFlow
iteration shows the forest after learning more data: trees grow deeper
and the per-tree accuracy improves across the iteration slider.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distances import ALL_DISTANCES
from node_tests import NodeTestSampler
from polsar_online_rf import OnlinePolSARRandomForest
from polsar_interface import convert_polsar_rf, evaluate_polsar_rf, make_polsar_router, patch_to_rgb
from example import make_synthetic_image, build_reference_pool, sample_training_patches
from treeflow import TreeFlow  # Library path is added by polsar_interface
from air_polsar import *

def main():
    rng = np.random.default_rng(0)

    # ---- 1. Synthetic data: several scenes sharing one class set -----------
    # (swap with your real loader — e.g. air_polsar.load_air_tiles for the
    # AIR-PolarSAR-Seg tiles; each tile is a different part of the scene)
    # k, n_classes = 3, 5
    # print("Generating 3 synthetic PolSAR scenes ...")
    # A = rng.standard_normal((n_classes, k, k)) + 1j * rng.standard_normal((n_classes, k, k))
    # class_means = [(a @ a.conj().T) / k + np.eye(k) * 0.5 for a in A]
    # scene_sizes = [(48, 48), (32, 48), (40, 32)]   # different parts, different extents
    # scenes = [make_synthetic_image(h, w, k, n_classes, rng, class_means=class_means)
    #           for h, w in scene_sizes]
    # images = [img for img, _ in scenes]
    # label_masks = [lbl for _, lbl in scenes]
    rng = np.random.default_rng(0)
    patch_size = 32
    train_n = 20000
    batch_size = 10000
    # Push-time eval cost is linear in this: the subset is routed through every
    # tree once per push. patches are shuffled, so a prefix is a fair sample.
    eval_n = 2000
    images, labels = load_air_tiles("test/train_set/", indices=[1, 55, 100, 299])
    pool = build_reference_pool(images, n=5000, rng=rng)
    patches, patch_labels = sample_training_patches(
    images, labels, n=train_n, patch_size=patch_size, rng=rng, ignore_label=AIR_IGNORE, balanced=True)
    # # Training stream and held-out evaluation set, drawn across ALL scenes
    # # (shuffled, so every batch mixes scenes)
    # n_train, n_eval = 1200, 200
    # all_patches, all_labels = sample_training_patches(
    #     images, label_masks, n=n_train + n_eval, patch_size=patch_size, rng=rng)
    # train_patches, train_labels = all_patches[:n_train], all_labels[:n_train]
    # eval_patches, eval_labels = all_patches[n_train:], all_labels[n_train:]
    n_classes = 6
    # ---- 2. Build the online forest ----------------------------------------
    # The reference pool for 1-pt projections is sampled across all scenes.
    sampler = NodeTestSampler(
        patch_size=patch_size,
        region_size_min=2,
        region_size_max=12,
        max_offset=20,
        reference_pool=pool,
        distance_names=list(ALL_DISTANCES.keys()),
    )
    
    rf = OnlinePolSARRandomForest(
        sampler=sampler,
        n_trees=30,           # demo settings — paper uses 50 trees, depth 50
        n_candidates=80,
        max_depth=32,
        tau=100,              # samples a node accumulates before it splits
        tau_min=5,
        tau_decay=0.5,
        n_classes=n_classes,
        random_state=42,
    )

    # ---- 3. Learn batch by batch, push one TreeFlow iteration per batch ---------
    treeflow = TreeFlow(port=8000, work_dir='vis-test/')

    # Map raw class indices ("0".."5") to the AIR-PolarSAR-Seg legend names —
    # shown everywhere the frontend displays labels (legend, node widget, ...).
    label_map = dict(enumerate(AIR_CLASS_NAMES))
    # Ground-truth palette as hex — tree pies / legend match the segmentation maps.
    class_colors = ["#%02x%02x%02x" % rgb for _, rgb in AIR_CLASSES]

    # Visual references: pooled patches are routed through the live forest at
    # every push, so each iteration records their root→leaf paths (NodeView in
    # the frontend shows them per node). Register the router once, then a
    # couple of eval patches per class as references.
    treeflow.set_reference_router(make_polsar_router(rf))
    per_class = {c: 0 for c in range(n_classes)}
    for patch, lbl in zip(patches, patch_labels):
        c = int(lbl)
        if per_class[c] >= 2:
            continue
        per_class[c] += 1
        treeflow.add_reference(patch, image=patch_to_rgb(patch),
                          label=AIR_CLASS_NAMES[c], label_index=c, step=0,
                          name=f"{AIR_CLASS_NAMES[c]}-{per_class[c]}")

    print(f"Training on {train_n} samples in batches of {batch_size} ...")
    for start in range(0, train_n, batch_size):
        rf.partial_fit(patches[start:start + batch_size],
                       patch_labels[start:start + batch_size])

        # Mid-training pool growth: later references appear from the iteration
        # they were added onward (earlier iterations don't know them).
        print('reference stage')
        if start == batch_size * 2:
            p, c = patches[-1], int(patch_labels[-1])
            treeflow.add_reference(p, image=patch_to_rgb(p), label=AIR_CLASS_NAMES[c],
                              label_index=c, step=start, name=f"late-{AIR_CLASS_NAMES[c]}")
        print('eval stage')
        # One routing pass gives per-tree AND forest accuracy. (Previously the
        # eval set was routed twice per push: once inside convert_polsar_rf for
        # per-tree metrics, once through rf.predict for the printout.)
        ev = evaluate_polsar_rf(rf, patches[:eval_n], patch_labels[:eval_n])
        print('convert stage')
        forest = convert_polsar_rf(rf, tree_metrics=ev["tree_metrics"])
        forest.class_colors = class_colors
        idx = treeflow.push(forest, label_map=label_map)
        # Print both scopes together: the frontend shows PER-TREE accuracy, so
        # seeing it next to the ensemble figure makes the (expected) gap obvious.
        tree_acc = sum(m["accuracy"] for m in ev["tree_metrics"]) / len(ev["tree_metrics"])
        print(f"Pushed iteration {idx} — forest {ev['forest_accuracy']:.2%} "
              f"vs ⌀ single tree {tree_acc:.2%} (chance {1 / n_classes:.2%})")

    print("\nBackend running at http://localhost:8000")
    print("Start the frontend with:  cd ../Frontend && npm run dev")
    print("Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down ...")
        treeflow.shutdown()


if __name__ == "__main__":
    main()
