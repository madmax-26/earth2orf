"""
End-to-end demo: train an online forest on Five-Billion-Pixels and watch it
learn in the TreeFlow frontend.

Run the backend:
    python example_fbp_treeflow.py             # serves on http://localhost:8000

Then start the frontend:
    cd ../Frontend && npm run dev         # http://localhost:5173

Same shape as example_rgb_treeflow.py, on a different KIND of data. AID labels a
whole tile; Five-Billion-Pixels labels every pixel, so the label of a patch is
read out of the mask crop (see fbp_datasets). Three consequences worth knowing
before reading the accuracy figure:

  * 24 land-cover classes at 4 m GSD are far more confusable than AID's scene
    categories — paddy field vs irrigated field vs dry cropland differ by
    texture and season more than by anything visible in a 64 px crop. Expect
    numbers well below the AID demo's.

  * The classes are heavily imbalanced, so overall accuracy flatters a forest
    that has mostly learned "cropland or residential". The per-class histogram
    printed at startup is the context for the accuracy that follows.

  * Startup is slow in a way the AID demo is not: each tile is a ~150 MB
    decode. Building the eval set alone reads several tiles. MAX_TILES exists
    to keep a first run short.

BITS/COMPOSITE is the one choice worth making deliberately. bits=16 with
composite='rgb' is the default here because it is the only way to get TRUE
COLOUR — the 8-bit tiles carry no blue band at all (see fbp_datasets) — and
readable thumbnails are the whole reason to run ImageRF rather than the PolSAR
forest. Switch to bits=8, composite='cir' to train on the exact re-quantized
tiles the paper used; that is the setting to compare against published numbers,
at the cost of false-colour thumbnails.
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from fbp_datasets import (class_histogram, fbp_class_colors,  # noqa: E402
                          fbp_label_map, load_fbp_dataset,
                          prepare_patch_stream, sample_patch_set,
                          train_test_split_fbp)
from rgb_node_tests import RectTestSampler  # noqa: E402
from rgb_forest import ImageOnlineRandomForest  # noqa: E402
from rgb_interface import (convert_image_rf, evaluate_image_rf,  # noqa: E402
                           make_image_router, patch_to_rgb)
from treeflow import TreeFlow  # noqa: E402  (Library path is added by rgb_interface)

# None = all 150 tiles. Set to e.g. 20 for a quick first run; the official test
# split is applied AFTER this, so keep it generous enough to retain test tiles.
MAX_TILES = None

# See the module docstring: (16, 'rgb') is true colour, (8, 'cir') is the
# paper's tiles. (16, 'cir') also works; (8, 'rgb') is impossible and raises.
BITS, COMPOSITE = 16, "rgb"


def main():
    rng = np.random.default_rng(0)

    # ---- 1. Data ------------------------------------------------------------
    patch_size = 64
    per_tile = 400
    batch_size = 2000
    eval_n = 1500
    epochs = 2

    print("Loading Five-Billion-Pixels ...")
    ds = load_fbp_dataset(bits=BITS, composite=COMPOSITE,
                          max_tiles=MAX_TILES, rng=rng)
    train_ds, test_ds = train_test_split_fbp(ds)
    print(f"  {ds.bits}-bit {ds.composite}: {len(ds)} tiles, {ds.n_classes} classes "
          f"({len(train_ds)} train / {len(test_ds)} test tiles, official split)")

    print(f"Preparing {eval_n} evaluation patches (a few ~150 MB tile reads) ...")
    eval_patches, eval_labels = sample_patch_set(test_ds, n=eval_n,
                                                 patch_size=patch_size, rng=rng)
    hist = {k: v for k, v in class_histogram(eval_labels).items() if v}
    print("  eval class mix: "
          + ", ".join(f"{k} {v}" for k, v in sorted(hist.items(), key=lambda kv: -kv[1])))

    # ---- 2. Forest ----------------------------------------------------------
    sampler = RectTestSampler(patch_size=patch_size, min_rect_size=1,
                              max_rect_size=patch_size, cross_channel=False)
    rf = ImageOnlineRandomForest(
        sampler=sampler,
        n_trees=20,
        n_candidates=100,
        max_depth=24,
        tau=1000,
        tau_decay=0.7,
        tau_min=10,
        n_classes=ds.n_classes,
        random_state=42,
    )

    # ---- 3. TreeFlow -------------------------------------------------------------
    treeflow = TreeFlow(port=8000, work_dir="vis-fbp/")
    treeflow.set_reference_router(make_image_router(rf))

    # References come from the eval pool, as in the AID demo. Classes missing
    # from that pool are skipped rather than forced — with 24 imbalanced
    # classes several rare ones may simply not appear in 1500 patches.
    refs_per_class = 6
    for c in range(ds.n_classes):
        hits = np.where(eval_labels == c)[0]
        for i in range(min(refs_per_class, len(hits))):
            p = eval_patches[int(hits[i])]
            treeflow.add_reference(p, image=patch_to_rgb(p), label=ds.class_names[c],
                              label_index=c, step=0,
                              name=f"{ds.class_names[c]}-{i + 1}")

    label_map = fbp_label_map(ds.class_names)
    class_colors = fbp_class_colors()          # the dataset's own palette

    # ---- 4. Learn batch by batch, push one TreeFlow iteration per batch ----------
    print(f"Training: {epochs} epoch(s) x {len(train_ds)} tiles x {per_tile} patches ...")
    t0 = time.perf_counter()
    for patches, labels in prepare_patch_stream(train_ds, patch_size=patch_size,
                                                per_tile=per_tile,
                                                batch_size=batch_size,
                                                epochs=epochs, rng=rng):
        rf.partial_fit(patches, labels)

        ev = evaluate_image_rf(rf, eval_patches, eval_labels)
        forest = convert_image_rf(rf, class_labels=ds.class_names,
                                  tree_metrics=ev["tree_metrics"])
        forest.class_colors = class_colors
        idx = treeflow.push(forest, label_map=label_map)
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
