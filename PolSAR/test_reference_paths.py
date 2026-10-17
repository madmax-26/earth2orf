"""
Assertion script for the visual-reference feature (run manually):

    python test_reference_paths.py

Checks, on a small synthetic online PolSAR forest:
  1. Path/id alignment — every recorded reference path starts at the root id,
     every consecutive pair is parent→child in the tree the converter emitted
     (children rebuilt from (id, depth) preorder exactly like the frontend),
     and the terminal node is a leaf (unless the path was truncated by an
     unevaluable NodeTest, which is reported).
  2. Endpoints — /references schema, /reference/{id}/image PNG signature,
     404 for a stale id.
  3. Persistence round-trip — a second TreeFlow on the same work_dir restores
     identical snapshots and serves images from disk.
  4. Backward compat — a pre-feature work_dir loads cleanly with empty
     reference lists.
"""

import json
import os
import shutil
import sys
import tempfile
import urllib.error
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from distances import ALL_DISTANCES
from node_tests import NodeTestSampler
from polsar_online_rf import OnlinePolSARRandomForest
from polsar_interface import convert_polsar_rf, make_polsar_router, patch_to_rgb
from example import make_synthetic_image, build_reference_pool
from treeflow import TreeFlow

PASS, FAIL = "PASS", "FAIL"
failures = 0


def check(ok: bool, msg: str) -> None:
    global failures
    print(f"  {PASS if ok else FAIL}  {msg}")
    if not ok:
        failures += 1


def children_map(nodes):
    """id -> [child ids] rebuilt from (id, depth) preorder — the same depth-stack
    walk the frontend's buildHierarchy uses."""
    children = {n.id: [] for n in nodes}
    stack = []  # indices into nodes
    for i, n in enumerate(nodes):
        while stack and nodes[stack[-1]].depth >= n.depth:
            stack.pop()
        if stack:
            children[nodes[stack[-1]].id].append(n.id)
        stack.append(i)
    return children


def fetch_json(url):
    with urllib.request.urlopen(url) as r:
        return json.loads(r.read())


def main():
    rng = np.random.default_rng(1)
    H, W, k, n_classes = 32, 32, 3, 4
    image, labels = make_synthetic_image(H, W, k, n_classes, rng)
    patch_size = 7
    half = patch_size // 2
    padded = np.pad(image, ((half, half), (half, half), (0, 0), (0, 0)), mode="edge")
    rows = rng.integers(0, H, size=500)
    cols = rng.integers(0, W, size=500)
    patches = [padded[r:r + patch_size, c:c + patch_size] for r, c in zip(rows, cols)]
    pix_labels = labels[rows, cols]

    sampler = NodeTestSampler(
        patch_size=patch_size, region_size_min=2, region_size_max=3, max_offset=2,
        reference_pool=build_reference_pool(image, n=50, rng=rng),
        distance_names=list(ALL_DISTANCES.keys()),
    )
    rf = OnlinePolSARRandomForest(sampler=sampler, n_trees=3, n_candidates=8,
                                  max_depth=6, tau=40, n_classes=n_classes,
                                  random_state=7)

    work_dir = tempfile.mkdtemp(prefix="treeflow-ref-test-")
    port = 8123
    treeflow = TreeFlow(port=port, work_dir=work_dir)
    treeflow.set_reference_router(make_polsar_router(rf))

    # 2 references per class from held-back patches
    for c in range(n_classes):
        idxs = np.where(pix_labels == c)[0][:2]
        for j, i in enumerate(idxs):
            treeflow.add_reference(patches[i], image=patch_to_rgb(patches[i]),
                              label=str(c), label_index=c, step=0, name=f"c{c}-{j}")

    # Two iterations: train, push, train more, push
    rf.partial_fit(patches[:250], pix_labels[:250])
    treeflow.push(convert_polsar_rf(rf))
    rf.partial_fit(patches[250:], pix_labels[250:])
    treeflow.push(convert_polsar_rf(rf))

    # ── 1. Path/id alignment ────────────────────────────────────────────────
    print("Path/id alignment:")
    truncated = 0
    checked = 0
    for it, (forest, snaps) in enumerate(zip(treeflow._iterations, treeflow._iteration_refs)):
        check(len(snaps) == 2 * n_classes, f"iteration {it}: {len(snaps)} snapshots")
        for s in snaps:
            check(set(s.paths.keys()) == set(forest.trees.keys()),
                  f"it{it} {s.id}: paths cover all trees") if s is snaps[0] else None
            for tid, path in s.paths.items():
                nodes = forest.trees[tid].nodes
                ids = {n.id for n in nodes}
                kids = children_map(nodes)
                ok = bool(path) and path[0] == nodes[0].id and all(p in ids for p in path)
                ok = ok and all(path[i + 1] in kids[path[i]] for i in range(len(path) - 1))
                if ok and kids[path[-1]]:
                    truncated += 1
                if not ok:
                    check(False, f"it{it} {s.id} tree{tid}: invalid path {path}")
                checked += 1
    check(checked > 0, f"validated {checked} paths ({truncated} truncated)")
    print(f"  ({checked} paths checked, {truncated} truncated by unevaluable tests)")

    # ── 2. Endpoints ────────────────────────────────────────────────────────
    print("Endpoints:")
    base = f"http://127.0.0.1:{port}"
    refs = fetch_json(f"{base}/iteration/0/references")
    check(len(refs) == 2 * n_classes, f"/references returns {len(refs)} entries")
    r0 = refs[0]
    check(all(key in r0 for key in
              ("id", "name", "label", "label_index", "step", "paths", "image_url")),
          "reference schema complete")
    check(isinstance(next(iter(r0["paths"].values())), list), "paths are id lists")
    with urllib.request.urlopen(base + r0["image_url"]) as resp:
        png = resp.read()
        check(resp.headers["Content-Type"] == "image/png", "image content-type")
    check(png.startswith(b"\x89PNG\r\n\x1a\n"), "image is a valid PNG")
    try:
        urllib.request.urlopen(f"{base}/iteration/0/reference/ref-999/image")
        check(False, "stale ref id -> 404")
    except urllib.error.HTTPError as e:
        check(e.code == 404, "stale ref id -> 404")

    # ── 3. Persistence round-trip ───────────────────────────────────────────
    print("Persistence round-trip:")
    treeflow2 = TreeFlow(port=port + 1, work_dir=work_dir)  # no server started (no push)
    check(len(treeflow2._iteration_refs) == len(treeflow._iteration_refs), "iteration_refs parallel")
    same = all(
        [(a.id, a.label, a.label_index, a.step, a.name, a.paths)
         for a in s1] ==
        [(b.id, b.label, b.label_index, b.step, b.name, b.paths)
         for b in s2]
        for s1, s2 in zip(treeflow._iteration_refs, treeflow2._iteration_refs)
    )
    check(same, "snapshots identical after reload")
    check(treeflow2._get_ref_png(refs[0]["id"]) == png, "PNG bytes identical from disk")
    check(treeflow2._ref_counter == treeflow._ref_counter, "ref id counter restored (no collisions)")

    # ── 4. Backward compat: pre-feature work_dir ────────────────────────────
    print("Backward compat:")
    legacy = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test-multilabel")
    if os.path.isdir(os.path.join(legacy, "iteration-0")):
        treeflow3 = TreeFlow(port=port + 2, work_dir=legacy)
        check(len(treeflow3._iterations) > 0, f"legacy dir loads ({len(treeflow3._iterations)} iterations)")
        check(all(s == [] for s in treeflow3._iteration_refs), "legacy iterations have no references")
    else:
        print("  SKIP  ../test-multilabel not found")

    treeflow.shutdown()
    shutil.rmtree(work_dir, ignore_errors=True)

    print(f"\n{'ALL PASS' if failures == 0 else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
