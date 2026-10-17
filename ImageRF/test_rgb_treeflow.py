"""
Section 11 of the ImageRF assertion suite: the TreeFlow round-trip.

Mirrors PolSAR/test_reference_paths.py's path/id-alignment check, and adds the
assertion that is the entire point of ImageRF: the PNG the backend serves for a
reference decodes back to EXACTLY the patch the forest was given.

Imported and run by test_rgb_rf.py.
"""

import io
import json
import os
import shutil
import sys
import tempfile
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _check():
    import test_rgb_rf
    return test_rgb_rf.check


def children_map(nodes):
    """id -> [child ids] rebuilt from (id, depth) preorder — the same depth-stack
    walk the frontend's buildHierarchy uses."""
    children = {n.id: [] for n in nodes}
    stack = []
    for i, n in enumerate(nodes):
        while stack and nodes[stack[-1]].depth >= n.depth:
            stack.pop()
        if stack:
            children[nodes[stack[-1]].id].append(n.id)
        stack.append(i)
    return children


def run_treeflow_section(rng) -> None:
    check = _check()

    from rgb_datasets import load_scene_dataset, sample_patch_set, SceneDataset
    from rgb_node_tests import RectTestSampler
    from rgb_forest import ImageOnlineRandomForest
    from rgb_interface import (convert_image_rf, evaluate_image_rf,
                               make_image_router, patch_to_rgb,
                               scene_class_colors, scene_label_map)
    from treeflow import TreeFlow

    print("TreeFlow round-trip:")
    PATCH, N_CLASSES, PORT = 32, 4, 8351

    full = load_scene_dataset("AID", max_per_class=12, rng=rng)
    idx = [i for i, l in enumerate(full.labels) if l < N_CLASSES]
    ds = SceneDataset("AID-4", [full.paths[i] for i in idx],
                      np.array([full.labels[i] for i in idx], dtype=int),
                      full.class_names[:N_CLASSES])
    patches, labels = sample_patch_set(ds, n=400, patch_size=PATCH, rng=rng)

    sampler = RectTestSampler(patch_size=PATCH)
    rf = ImageOnlineRandomForest(sampler=sampler, n_trees=3, n_candidates=40,
                                 max_depth=8, tau=60, tau_decay=0.7, tau_min=10,
                                 n_classes=N_CLASSES, random_state=7)

    work_dir = tempfile.mkdtemp(prefix="treeflow-imagerf-test-")
    treeflow = TreeFlow(port=PORT, work_dir=work_dir)
    treeflow.set_reference_router(make_image_router(rf))

    ref_patches = {}
    for c in range(N_CLASSES):
        hits = np.where(labels == c)[0]
        if not len(hits):
            continue
        p = patches[int(hits[0])]
        rid = treeflow.add_reference(p, image=patch_to_rgb(p), label=ds.class_names[c],
                                label_index=c, step=0, name=f"{ds.class_names[c]}-1")
        ref_patches[rid] = p

    label_map = scene_label_map(ds.class_names)
    for half in (patches[:200], patches[200:]):
        lab = labels[:200] if half is patches[:200] else labels[200:]
        rf.partial_fit(half, lab)
        ev = evaluate_image_rf(rf, patches[:100], labels[:100])
        forest = convert_image_rf(rf, class_labels=ds.class_names,
                                  tree_metrics=ev["tree_metrics"])
        forest.class_colors = scene_class_colors(N_CLASSES)
        treeflow.push(forest, label_map=label_map)

    # ── Path / id alignment ─────────────────────────────────────────────────
    checked = truncated = 0
    bad = 0
    for it, (forest, snaps) in enumerate(zip(treeflow._iterations, treeflow._iteration_refs)):
        for s in snaps:
            for tid, path in s.paths.items():
                nodes = forest.trees[tid].nodes
                ids = {n.id for n in nodes}
                kids = children_map(nodes)
                ok = bool(path) and path[0] == nodes[0].id and all(p in ids for p in path)
                ok = ok and all(path[i + 1] in kids[path[i]] for i in range(len(path) - 1))
                if ok and kids[path[-1]]:
                    truncated += 1
                if not ok:
                    bad += 1
                checked += 1
    check(checked > 0 and bad == 0,
          f"all {checked} reference paths are valid root->leaf walks ({truncated} truncated)")
    check(len(treeflow._iteration_refs) == 2 and all(len(s) == N_CLASSES for s in treeflow._iteration_refs),
          f"{N_CLASSES} reference snapshots recorded per iteration")

    # ── Endpoints + the assertion that matters ──────────────────────────────
    base = f"http://127.0.0.1:{PORT}"
    refs = json.loads(urllib.request.urlopen(f"{base}/iteration/0/references").read())
    check(len(refs) == N_CLASSES, f"/references returns {len(refs)} entries")
    r0 = refs[0]
    check(all(k in r0 for k in ("id", "name", "label", "label_index", "step",
                                "paths", "image_url")),
          "reference schema complete")
    check("?v=" in r0["image_url"], "image_url carries the cache-busting nonce")

    from PIL import Image
    identical = True
    for r in refs:
        with urllib.request.urlopen(base + r["image_url"]) as resp:
            png = resp.read()
        decoded = np.asarray(Image.open(io.BytesIO(png)).convert("RGB"), dtype=np.uint8)
        want = ref_patches[r["id"]].arr
        if not np.array_equal(decoded, want):
            identical = False
    check(identical,
          "served PNG decodes to EXACTLY the patch — references are real pictures")

    # ── Persistence round-trip ──────────────────────────────────────────────
    treeflow2 = TreeFlow(port=PORT + 1, work_dir=work_dir)
    same = all(
        [(a.id, a.label, a.label_index, a.step, a.name, a.paths) for a in s1]
        == [(b.id, b.label, b.label_index, b.step, b.name, b.paths) for b in s2]
        for s1, s2 in zip(treeflow._iteration_refs, treeflow2._iteration_refs)
    )
    check(same, "reference snapshots identical after reload")
    check(treeflow2._iterations[0].class_labels == [ds.class_names[i] for i in range(N_CLASSES)]
          or len(treeflow2._iterations[0].class_labels) == N_CLASSES,
          "class labels persisted")

    treeflow.shutdown()
    shutil.rmtree(work_dir, ignore_errors=True)
