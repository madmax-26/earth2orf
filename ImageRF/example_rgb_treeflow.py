"""
End-to-end demo: train an online RGB scene forest batch by batch and watch it
learn in the TreeFlow frontend.

Run the backend:
    python example_rgb_treeflow.py             # serves on http://localhost:8000

Then start the frontend:
    cd ../Frontend && npm run dev         # http://localhost:5173

One snapshot is pushed after every batch, so each TreeFlow iteration shows the
forest after learning more data. The point of ImageRF over the PolSAR forest is
what you see in NodeView: because the patches ARE pictures, every reference
thumbnail is a readable aerial scene and MeanImage's per-node average shows
what that node has actually collected.

NOTE: the UAE-RS files are the `_adv` (adversarially perturbed) copies of
AID/UC-Merced, so absolute accuracy here is not comparable to published AID
numbers. Point load_scene_dataset at a clean directory with root=... if you
have one.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rgb_datasets import (load_scene_dataset, prepare_patch_stream,  # noqa: E402
                          sample_patch_set, train_test_split_scenes)
from rgb_node_tests import RectTestSampler  # noqa: E402
from rgb_forest import ImageOnlineRandomForest  # noqa: E402
from rgb_interface import (convert_image_rf, evaluate_image_rf,  # noqa: E402
                           make_image_router, patch_to_rgb,
                           scene_class_colors, scene_label_map)
from treeflow import TreeFlow  # noqa: E402  (Library path is added by rgb_interface)


def main():
    rng = np.random.default_rng(0)

    # ---- 1. Data ------------------------------------------------------------
    patch_size = 64
    per_image = 4
    batch_size = 2000
    eval_n = 1500
    epochs = 4

    print("Loading AID ...")
    ds = load_scene_dataset("AID")
    train_ds, test_ds = train_test_split_scenes(ds, test_frac=0.2, rng=rng)
    print(f"  {len(ds)} tiles, {ds.n_classes} classes "
          f"({len(train_ds)} train / {len(test_ds)} test images)")

    # The eval set is routed again at EVERY push, so prepare it once here
    # rather than rebuilding integral images per iteration.
    print(f"Preparing {eval_n} evaluation patches ...")
    eval_patches, eval_labels = sample_patch_set(test_ds, n=eval_n, patch_size=patch_size, rng=rng)

    # ---- 2. Forest ----------------------------------------------------------
    sampler = RectTestSampler(patch_size=patch_size, min_rect_size=1,
                              max_rect_size=patch_size, cross_channel=False)
    rf = ImageOnlineRandomForest(
        sampler=sampler,
        n_trees=50,
        n_candidates=100,    # affordable: a projection is 4 integer lookups
        max_depth=24,
        tau=1000,             # samples a node accumulates before it splits
        tau_decay=0.7,       # deeper nodes split on less evidence
        tau_min=10,
        n_classes=ds.n_classes,
        random_state=42,
    )

    # ---- 3. TreeFlow -------------------------------------------------------------
    treeflow = TreeFlow(port=8000, work_dir="vis-imagerf/")
    treeflow.set_reference_router(make_image_router(rf))

    # One reference per class, taken from the eval pool. These are the readable
    # pictures that motivated ImageRF — patch_to_rgb here is the identity.
    for c in range(ds.n_classes):
        hits = np.where(eval_labels == c)[0]
        if not len(hits):
            continue
        for i in range(0, 6):#How many samples per class
            p = eval_patches[int(hits[i])]
            treeflow.add_reference(p, image=patch_to_rgb(p), label=ds.class_names[c],
                          label_index=c, step=0, name=f"{ds.class_names[c]}-1")

    label_map = scene_label_map(ds.class_names)
    class_colors = scene_class_colors(ds.n_classes)

    # ---- 4. Learn batch by batch, push one TreeFlow iteration per batch ----------
    print(f"Training: {epochs} epoch(s) x {len(train_ds)} tiles x {per_image} subwindows ...")
    t0 = time.perf_counter()
    for patches, labels in prepare_patch_stream(train_ds, patch_size=patch_size,
                                                per_image=per_image,
                                                batch_size=batch_size,
                                                epochs=epochs, rng=rng):
        rf.partial_fit(patches, labels)

        # One routing pass gives per-tree AND forest accuracy.
        ev = evaluate_image_rf(rf, eval_patches, eval_labels)
        forest = convert_image_rf(rf, class_labels=ds.class_names,
                                  tree_metrics=ev["tree_metrics"])
        forest.class_colors = class_colors
        idx = treeflow.push(forest, label_map=label_map)
        # Print both scopes together: the frontend shows PER-TREE accuracy, so
        # seeing it next to the ensemble figure makes the (expected) gap obvious.
        tree_acc = sum(m["accuracy"] for m in ev["tree_metrics"]) / len(ev["tree_metrics"])
        print(f"Pushed iteration {idx} — forest {ev['forest_accuracy']:.2%} "
              f"vs ⌀ single tree {tree_acc:.2%} "
              f"(chance {1 / ds.n_classes:.2%}, {time.perf_counter() - t0:.0f}s elapsed)")

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
