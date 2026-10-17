"""
Toy online random forest — a dataset-free way to see TreeFlow in action.

Trains River's Adaptive Random Forest on a synthetic, slowly drifting
multi-class stream and records a snapshot every few hundred samples, so the
timeline shows what makes a forest *online*: trees growing as data arrives,
and trees being replaced when the drifting concept makes them obsolete.

Every tree is scored test-then-train (each sample is predicted before it is
learned), so the per-tree accuracy in the frontend is honest and tracks drift.

Usage (from the repository root):

    pip install river fastapi uvicorn
    python examples/toy_online_forest.py              # trains, then serves on :8000
    python examples/toy_online_forest.py --no-serve   # only writes the run folder

Then start the frontend (cd Frontend && npm run dev) and open
http://localhost:5173. A written run folder can be served again later with:

    python Library/treeflow.py --dir vis-toy
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from collections import deque

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO, "Library"))

try:
    from river.datasets import synth
    from river.forest import ARFClassifier
except ImportError:
    sys.exit("This example needs River:  pip install river")

from treeflow import TreeFlow                      # noqa: E402
from lib import apply_label_map, export_forest      # noqa: E402
from river_interface import convert_arf             # noqa: E402

CLASS_NAMES = ["Class A", "Class B", "Class C", "Class D", "Class E",
               "Class F", "Class G", "Class H", "Class I", "Class J"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Train a toy online random forest and visualize it with TreeFlow.")
    p.add_argument("--samples", type=int, default=6000, help="stream length (default 6000)")
    p.add_argument("--every", type=int, default=400,
                   help="samples between snapshots, i.e. per timeline step (default 400)")
    p.add_argument("--trees", type=int, default=10, help="trees in the forest (default 10)")
    p.add_argument("--classes", type=int, default=4, choices=range(2, 11), metavar="2-10",
                   help="number of classes (default 4)")
    p.add_argument("--features", type=int, default=8, help="input features (default 8)")
    p.add_argument("--drift", type=float, default=0.001,
                   help="concept drift speed, 0 = stationary stream (default 0.001)")
    p.add_argument("--seed", type=int, default=42, help="random seed (default 42)")
    p.add_argument("--out", default=os.path.join(REPO, "vis-toy"),
                   help="run folder to write (default <repo>/vis-toy)")
    p.add_argument("--port", type=int, default=8000, help="backend port (default 8000)")
    p.add_argument("--no-serve", action="store_true",
                   help="only write the run folder, don't start the backend")
    return p.parse_args()


def clear_run_folder(path: str) -> None:
    """Remove a previous run's snapshots. The backend appends to whatever it
    finds in its folder, so a rerun would otherwise continue the old timeline.
    Only the iteration-N/ and references/ folders it writes are deleted."""
    if not os.path.isdir(path):
        return
    removed = 0
    for name in os.listdir(path):
        if name.startswith("iteration-") or name == "references":
            shutil.rmtree(os.path.join(path, name))
            removed += 1
    if removed:
        print(f"Cleared {removed} folder(s) from the previous run in {path}")


def main() -> None:
    a = parse_args()
    clear_run_folder(a.out)
    os.makedirs(a.out, exist_ok=True)

    stream = synth.RandomRBFDrift(
        seed_model=a.seed, seed_sample=a.seed + 1,
        n_classes=a.classes, n_features=a.features,
        n_centroids=5 * a.classes, change_speed=a.drift,
        n_drift_centroids=2 * a.classes,
    )
    model = ARFClassifier(n_models=a.trees, seed=a.seed)
    label_map = {i: CLASS_NAMES[i] for i in range(a.classes)}

    viz = None if a.no_serve else TreeFlow(port=a.port, work_dir=a.out)

    recent: deque = deque(maxlen=200)       # eval samples for the tree-similarity map
    tree_hits = [0] * a.trees               # test-then-train correct, this window
    forest_hits = window = 0
    step = 0

    print(f"Training {a.trees} trees on {a.samples} samples "
          f"({a.classes} classes, drift speed {a.drift}) ...")
    for i, (x, y) in enumerate(stream.take(a.samples), 1):
        # Test-then-train: predict with the current trees before learning x.
        forest_hits += model.predict_one(x) == y
        for t, tree in enumerate(model.models):
            tree_hits[t] += tree.predict_one(x) == y
        window += 1

        model.learn_one(x, y)
        recent.append((x, y))

        if i % a.every == 0 or i == a.samples:
            forest = convert_arf(model, n_samples=i, eval_data=list(recent))
            for t, tree in forest.trees.items():
                tree.metrics["accuracy"] = round(tree_hits[t] / window, 4)

            if viz is not None:
                step = viz.push(forest, label_map=label_map)
            else:
                folder = os.path.join(a.out, f"iteration-{step}")
                os.makedirs(folder, exist_ok=True)
                export_forest(apply_label_map(forest, label_map), folder)

            print(f"  step {step:>2}: {i:>6} samples | forest accuracy "
                  f"{forest_hits / window:.1%} (last {window})")
            if viz is None:
                step += 1
            tree_hits = [0] * a.trees
            forest_hits = window = 0

    if viz is None:
        print(f"\nWrote {step} step(s) to {a.out}")
        rel = os.path.relpath(a.out)
        shown = a.out if rel.startswith("..") else rel   # no ../../ for folders elsewhere
        print(f"Serve them with:  python Library/treeflow.py --dir {shown}")
        return

    print(f"\nBackend running at http://localhost:{a.port} - start the frontend with")
    print("  cd Frontend && npm run dev     (then open http://localhost:5173)")
    print("Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        viz.shutdown()


if __name__ == "__main__":
    main()
