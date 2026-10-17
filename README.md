# TreeFlow

Watch online random forests learn. TreeFlow visualizes a random forest
**while it trains**: your training loop pushes a snapshot of the forest every
few steps, and a web frontend lets you scrub through those snapshots and see
how every tree grows, splits and separates the classes.

It was built for remote-sensing classification (PolSAR and RGB aerial scenes),
but works with any tree ensemble that can be converted to its forest format —
including [River](https://riverml.xyz) models out of the box.

## What you can do

- **Browse the forest over time** — one snapshot per training step; trees are
  drawn in a botanical layout with leaf size proportional to sample count.
- **Inspect nodes** — hover or click any node for its class distribution,
  impurity and sample count.
- **Analyse the forest** — class similarity heatmaps and MDS maps, and
  *Decisive Node Analysis*, which highlights the splits that separate two
  chosen classes.
- **Follow real samples through the trees** (image forests) — register a few
  image patches as *references*; each snapshot records the path every patch
  takes. *NodeView* shows which patches went left and right at a split, and can
  highlight a patch's path in the tree.
- **Track per-tree metrics** such as accuracy, with mean and spread across trees.

## Requirements

- Python 3.10+
- Node.js 20+ (for the frontend)

Python packages:

| Needed for | Packages |
|---|---|
| The backend (always) | `fastapi`, `uvicorn` |
| River models and the toy example | `river` |
| PolSAR / ImageRF forests and examples | `numpy`, `scipy`, `scikit-image`, `Pillow` |

```bash
pip install fastapi uvicorn          # add the others as needed
```

## Quick start (no download needed)

`examples/toy_online_forest.py` trains a small online random forest (River's
Adaptive Random Forest) on a synthetic, slowly drifting stream, so you can see
the interface without downloading a dataset:

```bash
# 1. Train the toy forest and serve it (backend on http://localhost:8000)
pip install fastapi uvicorn river
python examples/toy_online_forest.py

# 2. In a second terminal, start the frontend
cd Frontend
npm install
npm run dev
```

Open <http://localhost:5173> and drag the iteration slider: each step is the
forest after 400 more samples. Watch the trees grow, the per-tree accuracy
rise, and occasionally a tree shrink back to a stump — that is the forest
replacing a tree the drifting data made obsolete.

`--help` lists the knobs (stream length, trees, classes, drift speed, …).
`--no-serve` only writes the run folder (`vis-toy/`), which you can serve again
later with `python Library/treeflow.py --dir vis-toy`.

## Using it with your own model

Create one `TreeFlow` instance, then push a converted snapshot whenever you
want a new step on the timeline. The backend starts on the first push.

```python
import sys
sys.path.insert(0, "path/to/TreeFlow/Library")

from river.forest import ARFClassifier
from treeflow import TreeFlow
from river_interface import convert_arf

model = ARFClassifier()
viz = TreeFlow(port=8000, work_dir="runs/my-run")   # work_dir is optional

for i, (x, y) in enumerate(stream):
    model.learn_one(x, y)
    if i % 500 == 0:
        viz.push(convert_arf(model, n_samples=i + 1))

viz.shutdown()
```

- **`work_dir`** saves every snapshot to disk. Serve a saved run again later
  with `python Library/treeflow.py --dir runs/my-run`.
- **Converters:** `Library/river_interface.py` covers River's ARF, SRP and AMF
  classifiers, regressors, multi-label and mixed forests.
  `PolSAR/polsar_interface.py` and `ImageRF/rgb_interface.py` cover the two
  image forests in this repository. For another library, build a `Forest` from
  `Library/lib.py` yourself.

### Following samples through the trees (image forests)

```python
from rgb_interface import make_image_router, patch_to_rgb

viz.set_reference_router(make_image_router(rf))     # once, next to the model
viz.add_reference(patch, image=patch_to_rgb(patch), label="Forest")
# ... keep training and pushing; every push re-routes all references
```

Then **Alt+click** a node in the frontend to open NodeView.

## Controls

| Action | Gesture |
|---|---|
| Show a node's details | Hover |
| Keep the details open | Click (Esc or click elsewhere to close) |
| Collapse / expand a branch | Double-click |
| Open NodeView (image forests) | Alt+click |
| Show only the branches | Hold Ctrl |
| Zoom / pan | Mouse wheel / drag |

## Using it from the command line

You don't need to write any Python to use TreeFlow. The backend can serve a
**run folder** from disk — one saved by `work_dir`, or one you produce with any
tool or language:

```bash
python Library/treeflow.py --dir my-run --port 8000
```

| Option | Default | Meaning |
|---|---|---|
| `--dir` | `.` | Run folder to serve |
| `--port` | `8000` | Port the backend listens on |

Then start the frontend (`cd Frontend && npm run dev`) and open
<http://localhost:5173>.

### Run folder format

```
my-run/
  iteration-0/        one folder per timeline step, numbered from 0
    forest.json       class names and colours, one entry per tree
    tree-0.csv        one CSV per tree
    tree-1.csv
  iteration-1/
    ...
```

This shell session builds a complete, servable run by hand — a single tree
whose root splits *Water* from *Forest*:

```bash
mkdir -p my-run/iteration-0

cat > my-run/iteration-0/forest.json <<'EOF'
{
  "class_labels": ["Water", "Forest"],
  "class_colors": ["#4C78C8", "#2E7D32"],
  "trees": { "0": { "height": 1, "n_nodes": 3, "metrics": { "accuracy": 0.9 } } }
}
EOF

cat > my-run/iteration-0/tree-0.csv <<'EOF'
id,depth,n_node_samples,impurity,value
0,0,100,0.5,"[50, 50]"
1,1,55,0.165,"[50, 5]"
2,1,45,0.0,"[0, 45]"
EOF

python Library/treeflow.py --dir my-run
```

**`forest.json`**
- `class_labels` — class names; `class_colors` is optional.
- `trees` — one entry per `tree-N.csv`, keyed by `N`, with `height` (deepest
  node depth) and `n_nodes`. `metrics` holds any per-tree numbers you want
  shown (accuracy, …) and may be `{}`.

**`tree-N.csv`** — one row per node:
- Rows must be in **depth-first order**: a node, then its whole left subtree,
  then its whole right subtree. The tree structure is read from this order and
  the `depth` column — there are no parent columns.
- `n_node_samples` — training samples that reached the node.
- `impurity` — Gini impurity, 0 (pure) to 1; it colours the branches.
- `value` — samples per class, in `class_labels` order.

**Good to know**
- The folder is read **once at startup**. To see newly written iterations,
  restart the command.
- Iterations must be numbered without gaps: loading stops at the first
  missing `iteration-N`.
- The command-line server only displays data; to add iterations while
  training, use the Python API above with `work_dir`.

## Examples

| Script | Forest | Data |
|---|---|---|
| `examples/toy_online_forest.py` | River Adaptive Random Forest | None (synthetic stream) |
| `PolSAR/example_treeflow.py` | Online PolSAR forest | AIR-PolarSAR-Seg tiles in `test/train_set/` |
| `ImageRF/example_rgb_treeflow.py` | Online RGB scene forest | UAE-RS (AID / UCM) |
| `ImageRF/example_fbp_treeflow.py` | Online RGB scene forest | Five-Billion-Pixels |

Datasets are not included. Download them separately and point the scripts at
them:

- **UAE-RS:** a `UAE-RS/` folder next to this repository, or set the
  `UAE_RS_ROOT` environment variable.
- **Five-Billion-Pixels:** `big_datasets/Five-Billion-Pixels/` inside the
  repository, or set `FBP_ROOT`.

## Connecting to another backend

The frontend talks to `localhost:8000` by default. To use a backend on another
machine or port, click the gear icon in the top-right corner and enter its
address. The setting is remembered in the browser.

## Project layout

```
Library/    Backend (treeflow.py), forest format (lib.py), River converters
PolSAR/     Online random forest for polarimetric SAR patches
ImageRF/    Online random forest for RGB aerial image patches
Frontend/   React + TypeScript web interface (Vite)
examples/   Dataset-free toy online forest
```

## License

TreeFlow is **dual-licensed**:

- **Open source — [GNU AGPL v3](LICENSE).** You may use, modify and share
  TreeFlow for free, provided that you release the complete source of any work
  based on it under the same license. The AGPL also covers network use: if you
  run a modified version as a service that others access (for example a hosted
  dashboard), you must offer those users its source code.
- **Commercial license.** If you want to use TreeFlow in a product or service
  without the AGPL's obligations — for example in closed-source software — a
  separate commercial license is available from Thoren Hirsch
  (t.hirsch@campus.tu-berlin.de).

Copyright © 2026 Thoren Hirsch. Contributions are accepted only if the contributor
agrees that they may be distributed under both licenses.
