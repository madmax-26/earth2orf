"""
TreeFlow — visualization backend for the TreeFlow random forest frontend.

Programmatic usage:
    from treeflow import TreeFlow
    from river_interface import convert_srp

    treeflow = TreeFlow(port=8000)
    for i, (x, y) in enumerate(stream):
        srp.learn_one(x, y)
        if i % 500 == 0:
            treeflow.push(convert_srp(srp, n_samples=i + 1))
    treeflow.shutdown()

Visual references (optional; forests on visual data):
    treeflow.set_reference_router(make_polsar_router(rf))     # once, next to the model
    treeflow.add_reference(patch, image=patch_to_rgb(patch), label="water")
    ...training...
    treeflow.push(convert_polsar_rf(rf))   # routes every pooled reference through
                                      # the current model and stores the paths

CLI usage (loads saved iterations from a directory and serves them):
    python treeflow.py --port 8000 --dir ./data
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import struct
import threading
import time
import warnings
import zlib
from dataclasses import dataclass
from typing import Any, Callable

import uvicorn
from fastapi import FastAPI, HTTPException, Response
from fastapi.middleware.cors import CORSMiddleware

from lib import Forest, Node, Tree, apply_label_map


# ── Visual references ─────────────────────────────────────────────────────────
#
# A "reference" is a sample (e.g. an image patch) the user registers during
# training via TreeFlow.add_reference. References live in a pool; at every push()
# the registered router routes ALL pooled references through the current model,
# so each iteration stores every reference's root→leaf node path per tree —
# always consistent with that snapshot, and re-routed fresh each iteration so
# a reference's migration through the growing trees is visible over time.
# (Not to be confused with PolSAR's NodeTest.reference comparison matrices.)

@dataclass
class VisualReference:
    """One pooled reference: the opaque model input plus a displayable PNG."""
    id: str                          # "ref-N", stable across iterations
    sample: Any                      # model input (patch, feature dict, ...); NOT persisted
    png: bytes                       # displayable thumbnail (PNG bytes)
    label: str | None = None
    label_index: int | None = None   # index into the forest's class labels (badge color)
    step: int | None = None          # user's training step at registration
    name: str | None = None


@dataclass
class ReferenceSnapshot:
    """A pool entry as routed at one push: metadata + per-tree node paths."""
    id: str
    name: str | None
    label: str | None
    label_index: int | None
    step: int | None
    paths: dict[int, list[int]]      # tree_id -> node ids, root..terminal


_PNG_SIG = b"\x89PNG\r\n\x1a\n"


def _write_png(pixels: bytes, w: int, h: int, channels: int) -> bytes:
    """Minimal stdlib PNG encoder (8-bit gray/RGB/RGBA, no filtering)."""
    color_type = {1: 0, 3: 2, 4: 6}[channels]
    stride = w * channels
    raw = b"".join(b"\x00" + pixels[y * stride:(y + 1) * stride] for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        body = tag + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, color_type, 0, 0, 0)
    return _PNG_SIG + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def _encode_png(image) -> bytes:
    """Normalize a user-supplied image to PNG bytes.

    Accepts ready-made PNG bytes (validated by signature) or a uint8 ndarray of
    shape HxW, HxWx3, or HxWx4. Uses PIL when available, else a minimal stdlib
    encoder — no hard dependency. Duck-typed so treeflow.py needs no numpy import.
    """
    if isinstance(image, (bytes, bytearray)):
        b = bytes(image)
        if not b.startswith(_PNG_SIG):
            raise ValueError("image bytes must be a PNG file (bad signature)")
        return b

    shape = getattr(image, "shape", None)
    if shape is None:
        raise TypeError("image must be PNG bytes or a uint8 ndarray (HxW, HxWx3, HxWx4)")
    if str(getattr(image, "dtype", "")) != "uint8":
        raise TypeError("image array must have dtype uint8")
    if len(shape) == 2:
        h, w, c = int(shape[0]), int(shape[1]), 1
    elif len(shape) == 3 and shape[2] in (1, 3, 4):
        h, w, c = int(shape[0]), int(shape[1]), int(shape[2])
    else:
        raise ValueError(f"unsupported image shape {shape}")

    try:
        import io
        from PIL import Image  # optional
        arr = image.reshape(h, w) if c == 1 else image
        buf = io.BytesIO()
        Image.fromarray(arr, {1: "L", 3: "RGB", 4: "RGBA"}[c]).save(buf, "PNG")
        return buf.getvalue()
    except ImportError:
        return _write_png(image.tobytes(), w, h, c)


def _group_specs(forest: Forest) -> list:
    """Label groups of a forest as (name, value_offset, size, kind, reg_index) tuples.

    kind is "classification" or "regression" per group; reg_index is the group's
    position in target_labels/target_range/std_range (-1 for classification).
    Classification: one tuple per entry when class_labels is a dict (multi-group),
    else a single tuple spanning the whole value vector. Regression: one tuple
    per target (each target occupies one slot in the value/impurity arrays).
    Mixed: the class_labels dict groups first, then one regression tuple per
    target_labels entry — the same global ordering the frontend derives, so the
    ?group= ordinal stays consistent across both.
    """
    if forest.task == "regression":
        names = forest.target_labels or ([forest.target_label] if forest.target_label else ["target"])
        return [(str(n), i, 1, "regression", i) for i, n in enumerate(names)]

    cl = forest.class_labels
    specs, off = [], 0
    if isinstance(cl, dict):
        for name, labels in cl.items():
            specs.append((str(name), off, len(labels), "classification", -1))
            off += len(labels)
    elif cl:  # defensive: flat list = one unnamed classification group
        specs.append(("", 0, len(cl), "classification", -1))
        off = len(cl)

    if forest.task == "mixed":
        for k, name in enumerate(forest.target_labels or []):
            specs.append((str(name), off + k, 1, "regression", k))
        return specs

    return specs or [("", 0, len(cl), "classification", -1)]


def _tree_group_stats(tree: Tree, specs: list) -> list:
    """Per-group aggregate stats for one tree, aligned with the group ordinals.

    Classification group: {"purity": 1 − sample-weighted leaf Gini} — 1 means
    every leaf is pure. Regression target: {"mean": root mean prediction,
    "std": sample-weighted leaf std} — the residual spread left after splits.
    Derived entirely from node data, so nothing needs persisting.
    """
    nodes = tree.nodes
    if not nodes:
        return [{} for _ in specs]
    n = len(nodes)
    leaf_idx = [i for i in range(n) if i == n - 1 or nodes[i + 1].depth <= nodes[i].depth]
    total = sum(nodes[i].n_node_samples for i in leaf_idx) or 1

    stats = []
    for gi, (name, off, size, kind, reg_index) in enumerate(specs):
        def imp(node: Node) -> float:
            return node.impurity[gi] if isinstance(node.impurity, list) else node.impurity
        w_leaf_imp = sum(nodes[i].n_node_samples * imp(nodes[i]) for i in leaf_idx) / total
        if kind == "regression":
            root_mean = nodes[0].value[off] if off < len(nodes[0].value) else 0.0
            stats.append({"mean": root_mean, "std": w_leaf_imp})
        else:
            stats.append({"purity": max(0.0, min(1.0, 1.0 - w_leaf_imp))})
    return stats


def _forest_group_stats(per_tree_stats: list) -> list:
    """Forest-level aggregates: plain average of each per-tree stat, per group."""
    if not per_tree_stats:
        return []
    n_groups = len(per_tree_stats[0])
    out = []
    for gi in range(n_groups):
        acc: dict[str, float] = {}
        count = 0
        for stats in per_tree_stats:
            if gi >= len(stats) or not stats[gi]:
                continue
            count += 1
            for key, v in stats[gi].items():
                acc[key] = acc.get(key, 0.0) + v
        out.append({k: v / count for k, v in acc.items()} if count else {})
    return out


def _slice_forest_group(forest: Forest, group: int) -> Forest:
    """Project a multi-group forest onto one label group / regression target.

    Returns a single-group Forest of the group's own PURE task (per-node value
    sliced to the group's span, impurity reduced to the group's scalar) so every
    existing single-group analytic — similarity, tree distances — works on it
    unchanged. Downstream code never sees task="mixed". Single-group non-mixed
    forests are returned as-is.
    """
    specs = _group_specs(forest)
    if len(specs) <= 1 and forest.task != "mixed":
        return forest
    group = max(0, min(group, len(specs) - 1))
    name, off, size, kind, reg_index = specs[group]

    def node_of(n: Node) -> Node:
        imp = n.impurity[group] if isinstance(n.impurity, list) else n.impurity
        return Node(
            id=n.id, depth=n.depth, n_node_samples=n.n_node_samples,
            impurity=imp, value=n.value[off:off + size],
        )

    trees = {
        tid: Tree(nodes=[node_of(n) for n in t.nodes],
                  height=t.height, n_nodes=t.n_nodes, metrics=t.metrics)
        for tid, t in forest.trees.items()
    }

    def pick_range(r, idx):
        # Per-target ranges are a list of [min, max] pairs indexed by reg target;
        # a single flat [min, max] passes through.
        if r and isinstance(r[0], (list, tuple)):
            return list(r[idx]) if 0 <= idx < len(r) else None
        return r

    if kind == "regression":
        return Forest(
            trees=trees, task="regression",
            target_range=pick_range(forest.target_range, reg_index),
            std_range=pick_range(forest.std_range, reg_index),
            target_label=name,
            correlation_matrix=forest.correlation_matrix,
        )
    labels = forest.class_labels[name] if isinstance(forest.class_labels, dict) else forest.class_labels
    return Forest(
        trees=trees, class_labels=list(labels),
        correlation_matrix=forest.correlation_matrix,
    )


def _compute_similarity(forest: Forest) -> dict:
    """Pairwise label OVERLAP matrices from two complementary RF proximity metrics.

    Both are plain fractions in [0, 1] — 1 means the two labels behave identically
    under that metric, 0 means nothing in common — and both are averaged over the
    trees. The diagonal is 1 because a label trivially overlaps itself.

    leaf_cooccurrence = SUM over leaves of min(P_i, P_j)
        where P_c is label c's leaf distribution: of all of label c's samples
        that reached a leaf, the share sitting in each one. It sums to 1, so the
        overlap reads directly as "the share of a label's samples that land in
        the same leaves as the other label's". Answers "did the forest FAIL to
        separate them?" — it only sees confusion the splits left behind, and
        fades to nothing once leaves are pure.
        (Adapted from Shi & Horvath 2006 / Zhu et al. 2014 Variant I.)

    path_overlap = SUM over nodes of min(q_i, q_j)  /  the pair's mean path length
        where q_c(v) is the fraction of label c's samples that reach node v, so
        summing q_c over the tree gives label c's expected root->leaf path length.
        The ratio reads as "the fraction of their average root->leaf journey the
        two labels travel together before the forest can tell them apart".
        Answers "how HARD did the forest work to separate them?" — it keeps
        working at full depth because its signal is in the internal nodes, which
        is why it is the default and carries the imbalance check.
        (Zhu, Loy & Gong 2014, Eq. 11. The root is excluded, as in the source:
        every sample passes through it, so it would only add a constant.)

    Normalising each profile up front is what keeps these readable. A forest whose
    child counts do not sum to the parent's — online forests seed children from
    accumulated statistics and discard the samples, so parent >= left + right —
    leaks mass on the way down. Dividing the leaf profile by its own total rather
    than by the root sidesteps that entirely, so neither metric needs a repair
    step afterwards.

    Also returned, in raw units, for the UI readout:
      path_shared[i][j] — expected number of nodes labels i and j share
      path_len[c]       — expected root->leaf path length of label c
    Both root-excluded and tree-averaged, and consistent with path_overlap by
    construction: path_shared / mean(path_len) IS the path_overlap value.

    CLASSIFICATION ONLY. Both metrics compare labels through a per-node class
    histogram; a regression node carries a mean and a spread instead, with no
    labels to pair up. A regression group returns the same empty payload as a
    forest with too few labels, and the frontend renders "classification
    forests only".
    """
    if forest.task == "regression":
        return {"labels": [], "metrics": {}, "per_tree": {}}

    C = len(forest.class_labels)
    if C < 2:
        return {"labels": forest.class_labels, "metrics": {}, "per_tree": {}}

    def zeros(): return [[0.0] * C for _ in range(C)]
    leaf_acc = zeros()          # summed per-tree leaf overlap
    path_num = zeros()          # summed shared path length
    path_den = zeros()          # summed mean path length of each pair
    len_acc = [0.0] * C         # summed path length per label
    class_totals = [0.0] * C    # forest-wide samples per label (imbalance check)
    valid = 0
    per_tree: dict[str, dict] = {}

    for tid, tree in forest.trees.items():
        nodes = tree.nodes
        if not nodes:
            continue
        n = len(nodes)

        leaf_idx = [
            i for i in range(n)
            if i == n - 1 or nodes[i + 1].depth <= nodes[i].depth
        ]
        L = len(leaf_idx)
        if L == 0:
            continue

        root_value = nodes[0].value
        for c in range(C):
            class_totals[c] += root_value[c] if c < len(root_value) else 0.0

        def at(node_i: int, c: int) -> float:
            v = nodes[node_i].value
            return v[c] if c < len(v) else 0.0

        # P[c] — label c's leaf distribution, normalised to sum to 1.
        P = []
        for c in range(C):
            vals = [at(i, c) for i in leaf_idx]
            s = sum(vals)
            P.append([v / s for v in vals] if s > 0 else [0.0] * L)

        # q[c][v] — fraction of label c's samples reaching node v. Summing it
        # over the tree (root excluded) gives that label's expected path length.
        q = []
        for c in range(C):
            root_c = max(at(0, c), 1.0)
            q.append([at(i, c) / root_c for i in range(n)])
        plen = [sum(q[c][i] for i in range(1, n)) for c in range(C)]
        for c in range(C):
            len_acc[c] += plen[c]

        # Sparse supports — an optimisation only. Both metrics use min(), which
        # is 0 wherever either label is 0, so a pair need only walk the SMALLER
        # of the two supports. Node 0 is the root, excluded from path_overlap.
        nz_P = [[li for li in range(L) if P[c][li] > 0.0] for c in range(C)]
        nz_q = [[i for i in range(1, n) if q[c][i] > 0.0] for c in range(C)]

        t_leaf = zeros(); t_path = zeros(); t_shared = zeros()
        for c in range(C):
            t_leaf[c][c] = 1.0
            t_path[c][c] = 1.0
            t_shared[c][c] = plen[c]

        for ci in range(C):
            P_i, q_i = P[ci], q[ci]
            nzP_i, nzq_i = nz_P[ci], nz_q[ci]
            for cj in range(ci + 1, C):
                P_j, q_j = P[cj], q[cj]

                nzP_j = nz_P[cj]
                leaf = 0.0
                for li in (nzP_i if len(nzP_i) <= len(nzP_j) else nzP_j):
                    a, b = P_i[li], P_j[li]
                    leaf += a if a < b else b

                nzq_j = nz_q[cj]
                shared = 0.0
                for i in (nzq_i if len(nzq_i) <= len(nzq_j) else nzq_j):
                    a, b = q_i[i], q_j[i]
                    shared += a if a < b else b

                mean_len = (plen[ci] + plen[cj]) / 2
                t_leaf[ci][cj] = t_leaf[cj][ci] = leaf
                t_shared[ci][cj] = t_shared[cj][ci] = shared
                ratio = shared / mean_len if mean_len > 0 else 0.0
                t_path[ci][cj] = t_path[cj][ci] = min(1.0, ratio)

                leaf_acc[ci][cj] += leaf; leaf_acc[cj][ci] += leaf
                path_num[ci][cj] += shared; path_num[cj][ci] += shared
                path_den[ci][cj] += mean_len; path_den[cj][ci] += mean_len

        for c in range(C):
            leaf_acc[c][c] += 1.0
            path_num[c][c] += plen[c]
            path_den[c][c] += plen[c]

        per_tree[str(tid)] = {
            "leaf_cooccurrence": t_leaf,
            "path_overlap":      t_path,
            "path_shared":       t_shared,
            "path_len":          list(plen),
        }

        valid += 1

    if valid == 0:
        zm = [[0.0] * C for _ in range(C)]
        return {"labels": forest.class_labels,
                "metrics": {m: zm for m in ["leaf_cooccurrence", "path_overlap"]},
                "per_tree": {},
                "path_shared": zm, "path_len": [0.0] * C,
                "diagnostics": {"class_totals": class_totals, "majority_class": None,
                                "imbalance_warning": False}}

    def imbalance_warning(sim: list) -> bool:
        """Flag the imbalance degeneracy: a genuinely dominant label showing up as
        the most-overlapping neighbour of most others regardless of semantics.
        Only fires when one label actually dominates — a balanced forest, where
        any "majority" is just a tie-break artefact, never triggers it."""
        if C < 3:
            return False
        maj = max(range(C), key=lambda c: class_totals[c])
        avg = sum(class_totals) / C
        if class_totals[maj] <= 1.5 * avg:   # not actually dominant
            return False
        others = [i for i in range(C) if i != maj]
        hits = sum(
            1 for i in others
            if max((j for j in range(C) if j != i), key=lambda j: sim[i][j]) == maj
        )
        return hits / len(others) > 0.5

    # leaf_cooccurrence: mean of the per-tree overlaps.
    leaf_forest = [
        [min(1.0, leaf_acc[i][j] / valid) for j in range(C)]
        for i in range(C)
    ]
    # path_overlap: total shared length over total path length, pooled across the
    # trees. Pooling (rather than averaging the per-tree ratios) is what makes
    # path_shared / mean(path_len) reproduce this number exactly.
    path_forest = [
        [min(1.0, path_num[i][j] / path_den[i][j]) if path_den[i][j] > 0 else 0.0
         for j in range(C)]
        for i in range(C)
    ]

    return {
        "labels": forest.class_labels,
        "metrics": {
            "leaf_cooccurrence": leaf_forest,
            "path_overlap":      path_forest,
        },
        "per_tree": per_tree,
        # Raw units behind path_overlap, for the "X% of path shared" readout.
        "path_shared": [[path_num[i][j] / valid for j in range(C)] for i in range(C)],
        "path_len": [len_acc[c] / valid for c in range(C)],
        "diagnostics": {
            "class_totals": class_totals,
            "majority_class": max(range(C), key=lambda c: class_totals[c]),
            "imbalance_warning": imbalance_warning(path_forest),
        },
    }


def _compute_tree_distances(forest: Forest) -> dict:
    """T×T pairwise distance matrix between trees, from Forest.correlation_matrix.

    d = 1 - corr, where corr is supplied by the converter (see the converters'
    eval_data / eval_patches arguments). Returns source="none" with no distances
    when the forest carries no matrix — the frontend then disables the map.

    There is deliberately NO fallback. A structural fingerprint built from node
    statistics (impurity/depth histograms, class-prediction fractions) was
    measured against real per-tree prediction agreement on three independent
    datasets and correlated with it at roughly zero every time:

        TreeFlow mixed forests (45 pairs)   spearman -0.14 / -0.00
        sklearn tabular  (190 pairs)   spearman +0.11 / +0.04
        PolSAR           (190 pairs)   spearman +0.06

    The reason is structural: tau-driven growth makes every tree the same size
    and shape, and best-of-N split selection makes them split the same way near
    the root, so the aggregate statistics the fingerprint reads are precisely
    the parts that do NOT vary. Two trees can route samples completely
    differently and still produce identical histograms. Showing a map built on
    that is worse than showing none, because it looks authoritative.
    """
    tree_ids = [str(tid) for tid in sorted(forest.trees.keys())]
    T = len(tree_ids)
    if T < 2 or forest.correlation_matrix is None:
        return {"tree_ids": tree_ids, "distances": [], "source": "none"}

    mat = forest.correlation_matrix
    n = len(mat)
    dist = []
    for i in range(T):
        row = []
        for j in range(T):
            if i == j:
                row.append(0.0)
            elif i < n and j < n:
                row.append(max(0.0, min(1.0, 1.0 - mat[i][j])))
            else:
                row.append(1.0)
        dist.append(row)
    return {"tree_ids": tree_ids, "distances": dist, "source": "correlation"}




class TreeFlow:
    def __init__(self, port: int = 8000, work_dir: str | None = None):
        """
        port:     port the FastAPI server listens on.
        work_dir: optional directory for persistence. Each pushed iteration is
                  written to work_dir/iteration-N.json; existing files are
                  loaded on construction so a CLI restart restores prior state.
        """
        self.port = port
        self.work_dir = work_dir
        self._iterations: list[Forest] = []
        self._lock = threading.Lock()
        # Visual references (see module comment): pooled samples, the router
        # that maps them to per-tree node paths at each push, and the routed
        # snapshots per iteration (parallel to _iterations). _ref_png caches
        # PNG bytes by reference id (filled on add, lazily from disk on load).
        self._references: list[VisualReference] = []
        self._reference_router: Callable[[list], list[dict[int, list[int]]]] | None = None
        self._iteration_refs: list[list[ReferenceSnapshot]] = []
        self._ref_png: dict[str, bytes] = {}
        self._ref_counter = 0
        self._router_warned = False
        # Cache-busting nonce baked into reference image URLs. Ref ids and
        # iteration numbers restart from zero on every run, so without this a
        # browser that cached ref-0's PNG (Cache-Control below) would keep
        # showing the PREVIOUS run's image for the same URL.
        self._instance_nonce = os.urandom(4).hex()
        self._server: uvicorn.Server | None = None
        self._thread: threading.Thread | None = None
        self._app = self._build_app()

        if work_dir:
            self._load_from_dir(work_dir)

    # ── Public API ────────────────────────────────────────────────────────────

    def push(self, forest: Forest, label_map: dict | None = None) -> int:
        """Add a new iteration. Starts the server on the first call. Returns the iteration index.

        label_map: optional mapping applied to the forest's class labels /
                   group names / target names before serving and persisting,
                   e.g. {0: "Car", 1: "Truck"} — see lib.apply_label_map for
                   the full semantics (nested per-group maps, stripped
                   matching). The display names then appear everywhere the
                   frontend shows labels: legend, node widget, similarity
                   axes, references tab, tree statistics.

        If references were registered (add_reference) and a router is set
        (set_reference_router), every pooled reference is routed through the
        current model here, so the recorded paths always match this snapshot.
        """
        if label_map:
            apply_label_map(forest, label_map)

        with self._lock:
            idx = len(self._iterations)

            snaps: list[ReferenceSnapshot] = []
            if self._references:
                if self._reference_router is None:
                    if not self._router_warned:
                        warnings.warn(
                            "References were added but no router is registered — "
                            "call set_reference_router(...) to record their paths.",
                            stacklevel=2,
                        )
                        self._router_warned = True
                else:
                    all_paths = self._reference_router([r.sample for r in self._references])
                    snaps = [
                        ReferenceSnapshot(
                            id=r.id, name=r.name, label=r.label,
                            label_index=r.label_index, step=r.step,
                            paths=all_paths[i],
                        )
                        for i, r in enumerate(self._references)
                    ]

            self._iterations.append(forest)
            self._iteration_refs.append(snaps)
            if self.work_dir:
                self._persist(idx, forest, snaps)

        if self._server is None:
            self._start_server()

        return idx

    def add_reference(self, sample, image, *, label: str | None = None,
                      label_index: int | None = None, step: int | None = None,
                      name: str | None = None) -> str:
        """Register a reference sample to be routed through the model at every push.

        sample: the model input (e.g. an image patch) — kept in memory only,
                handed verbatim to the router at each push (not persisted).
        image:  a displayable rendition — PNG bytes, or a uint8 ndarray of
                shape HxW / HxWx3 / HxWx4 (interface modules provide helpers,
                e.g. polsar_interface.patch_to_rgb).
        Returns the reference id ("ref-N").

        Intended scale is tens of references, not thousands — every push routes
        the whole pool and each iteration stores every path.
        """
        png = _encode_png(image)
        with self._lock:
            rid = f"ref-{self._ref_counter}"
            self._ref_counter += 1
            self._references.append(VisualReference(
                id=rid, sample=sample, png=png,
                label=label, label_index=label_index, step=step, name=name,
            ))
            self._ref_png[rid] = png
            if len(self._references) == 501:
                warnings.warn("Over 500 references pooled — routing/persistence cost "
                              "grows with every push.", stacklevel=2)
        return rid

    def set_reference_router(self, router: Callable[[list], list[dict[int, list[int]]]]) -> None:
        """Register the routing function used by push() for pooled references.

        router(samples) must return one dict per sample (input order preserved):
        {tree_id: [node_id, ...]} — the root→leaf path of that sample in each
        tree. Node ids must match the ids the forest CONVERTER emits for the
        same trees (PolSAR: DFS-preorder positions, see make_polsar_router;
        a River router must map to River's BFS-assigned node ids).

        Routers typically close over the live model object — re-register if the
        model is replaced (e.g. retrained from scratch) rather than mutated.
        """
        with self._lock:
            self._reference_router = router

    def shutdown(self) -> None:
        """Stop the backend server."""
        if self._server is not None:
            self._server.should_exit = True
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
            self._server = None

    # ── Server lifecycle ──────────────────────────────────────────────────────

    def _start_server(self) -> None:
        config = uvicorn.Config(
            self._app,
            host="0.0.0.0",
            port=self.port,
            log_level="warning",
        )
        self._server = uvicorn.Server(config)
        self._thread = threading.Thread(target=self._server.run, daemon=True)
        self._thread.start()
        for _ in range(100):
            if self._server.started:
                return
            if not self._thread.is_alive():
                break
            time.sleep(0.1)
        # Failing silently here is poisonous: the script keeps training while
        # the frontend talks to whatever process actually owns the port.
        self._thread = None
        self._server = None
        raise RuntimeError(
            f"TreeFlow backend failed to start on port {self.port}. Most likely the "
            "port is already in use — e.g. a previous run or a notebook kernel "
            "still holding its TreeFlow server. Stop that process (or pass a "
            "different port=) and retry."
        )

    # ── FastAPI app ───────────────────────────────────────────────────────────

    def _build_app(self) -> FastAPI:
        app = FastAPI()
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["http://localhost:5173"],
            allow_credentials=False,
            allow_methods=["*"],
            allow_headers=["*"],
        )

        @app.get("/")
        def status():
            with self._lock:
                n = len(self._iterations)
            return {"status": "online", "iterations": n}

        @app.get("/iterations")
        def get_iterations():
            with self._lock:
                return list(range(len(self._iterations)))

        @app.get("/iteration/{iter_id}/meta")
        def get_meta(iter_id: int):
            with self._lock:
                if not 0 <= iter_id < len(self._iterations):
                    raise HTTPException(status_code=404, detail="Iteration not found")
                forest = self._iterations[iter_id]
            specs = _group_specs(forest)
            tree_ids = list(forest.trees.keys())
            per_tree_stats = [_tree_group_stats(forest.trees[tid], specs) for tid in tree_ids]
            return {
                "class_labels": forest.class_labels,
                "class_colors": forest.class_colors,
                "task": forest.task,
                "target_range": forest.target_range,
                "std_range": forest.std_range,
                "target_label": forest.target_label,
                "target_labels": forest.target_labels,
                # Forest-level per-group aggregates (avg over trees), group ordinals
                "forest_stats": _forest_group_stats(per_tree_stats),
                "trees": {
                    str(tid): {
                        "height": forest.trees[tid].height,
                        "n_nodes": forest.trees[tid].n_nodes,
                        "metrics": forest.trees[tid].metrics,
                        "group_metrics": forest.trees[tid].group_metrics or {},
                        # Per-tree per-group aggregates, aligned with group ordinals
                        "stats": per_tree_stats[i],
                    }
                    for i, tid in enumerate(tree_ids)
                },
            }

        @app.get("/iteration/{iter_id}/tree/{tree_id}")
        def get_tree(iter_id: int, tree_id: int):
            with self._lock:
                if not 0 <= iter_id < len(self._iterations):
                    raise HTTPException(status_code=404, detail="Iteration not found")
                forest = self._iterations[iter_id]
            if tree_id not in forest.trees:
                raise HTTPException(status_code=404, detail="Tree not found")
            return [
                {
                    "id": n.id,
                    "depth": n.depth,
                    "n_node_samples": n.n_node_samples,
                    "impurity": n.impurity,
                    "value": n.value,
                }
                for n in forest.trees[tree_id].nodes
            ]

        @app.get("/iteration/{iter_id}/references")
        def get_references(iter_id: int):
            with self._lock:
                if not 0 <= iter_id < len(self._iterations):
                    raise HTTPException(status_code=404, detail="Iteration not found")
                # Parallel list — but stay defensive for pre-feature state.
                snaps = self._iteration_refs[iter_id] if iter_id < len(self._iteration_refs) else []
            return [
                {
                    "id": s.id,
                    "name": s.name,
                    "label": s.label,
                    "label_index": s.label_index,
                    "step": s.step,
                    "paths": {str(tid): path for tid, path in s.paths.items()},
                    # ?v= keys the browser cache to THIS server instance — ids
                    # restart every run, the images behind them do not stay
                    # identical across runs.
                    "image_url": f"/iteration/{iter_id}/reference/{s.id}/image?v={self._instance_nonce}",
                }
                for s in snaps
            ]

        @app.get("/iteration/{iter_id}/reference/{ref_id}/image")
        def get_reference_image(iter_id: int, ref_id: str):
            with self._lock:
                if not 0 <= iter_id < len(self._iterations):
                    raise HTTPException(status_code=404, detail="Iteration not found")
            png = self._get_ref_png(ref_id)
            if png is None:
                raise HTTPException(status_code=404, detail="Reference image not found")
            # PNGs are immutable per id WITHIN one server instance — let the
            # browser cache them across iteration scrubbing. Staleness across
            # restarts is prevented by the ?v= instance nonce in image_url.
            return Response(content=png, media_type="image/png",
                            headers={"Cache-Control": "public, max-age=86400"})

        @app.get("/iteration/{iter_id}/similarity")
        def get_similarity(iter_id: int, group: int = 0):
            with self._lock:
                if not 0 <= iter_id < len(self._iterations):
                    raise HTTPException(status_code=404, detail="Iteration not found")
                forest = self._iterations[iter_id]

            t0 = time.perf_counter()
            result = _compute_similarity(_slice_forest_group(forest, group))
            result["computed_ms"] = (time.perf_counter() - t0) * 1000.0
            return result

        @app.get("/iteration/{iter_id}/tree-map")
        def get_tree_map(iter_id: int):
            # Deliberately no ?group=. A tree serves every task at once, so tree
            # similarity is a whole-tree property: the map reads the forest-level
            # correlation_matrix, which the converters already build across ALL
            # tasks (task-balanced — see _correlation_matrix_reg_multi). Slicing
            # by group never changed it; it only made switching groups refetch
            # an identical map and implied a per-task view that did not exist.
            # (Unknown query params are ignored, so an old ?group= is harmless.)
            with self._lock:
                if not 0 <= iter_id < len(self._iterations):
                    raise HTTPException(status_code=404, detail="Iteration not found")
                forest = self._iterations[iter_id]
            return _compute_tree_distances(forest)

        return app

    # ── Persistence ───────────────────────────────────────────────────────────
    #
    # Layout:  work_dir/
    #            references/         reference PNGs, shared across iterations
    #              ref-0.png         (written once; a single copied iteration-N
    #              ...                folder therefore has no images)
    #            iteration-0/
    #              forest.json   {"0": {height, n_nodes, metrics}, ...}
    #              tree-0.csv    id,depth,n_node_samples,impurity,value
    #              tree-1.csv
    #              references.json   pool snapshot + per-tree paths (absent when
    #                                no references were registered)
    #            iteration-1/
    #              ...

    def _persist(self, idx: int, forest: Forest, snaps: list[ReferenceSnapshot] | None = None) -> None:
        folder = os.path.join(self.work_dir, f"iteration-{idx}")
        os.makedirs(folder, exist_ok=True)

        # references.json + shared PNGs (only when references were routed —
        # absent file means "no references", which _load_from_dir maps to []).
        if snaps:
            refs_dir = os.path.join(self.work_dir, "references")
            os.makedirs(refs_dir, exist_ok=True)
            for s in snaps:
                png_path = os.path.join(refs_dir, f"{s.id}.png")
                if not os.path.exists(png_path):
                    png = self._ref_png.get(s.id)
                    if png is not None:
                        with open(png_path, "wb") as f:
                            f.write(png)
            with open(os.path.join(folder, "references.json"), "w") as f:
                json.dump([
                    {
                        "id": s.id,
                        "name": s.name,
                        "label": s.label,
                        "label_index": s.label_index,
                        "step": s.step,
                        "image": f"references/{s.id}.png",
                        "paths": {str(tid): path for tid, path in s.paths.items()},
                    }
                    for s in snaps
                ], f, indent=2)

        # forest.json
        def tree_json(tree: Tree) -> dict:
            d = {"height": tree.height, "n_nodes": tree.n_nodes, "metrics": tree.metrics}
            if tree.group_metrics:
                d["group_metrics"] = tree.group_metrics
            return d

        meta = {
            "class_labels": forest.class_labels,
            "class_colors": forest.class_colors,
            "task": forest.task,
            "trees": {str(tid): tree_json(tree) for tid, tree in forest.trees.items()},
        }
        if forest.correlation_matrix is not None:
            meta["correlation_matrix"] = forest.correlation_matrix
        if forest.task != "classification" or forest.target_labels is not None:
            meta["target_range"] = forest.target_range
            meta["std_range"] = forest.std_range
            meta["target_label"] = forest.target_label
            if forest.target_labels is not None:
                meta["target_labels"] = forest.target_labels
        with open(os.path.join(folder, "forest.json"), "w") as f:
            json.dump(meta, f, indent=4)

        # tree-N.csv
        for tid, tree in forest.trees.items():
            with open(os.path.join(folder, f"tree-{tid}.csv"), "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["id", "depth", "n_node_samples", "impurity", "value"])
                writer.writeheader()
                for n in tree.nodes:
                    writer.writerow({
                        "id": n.id,
                        "depth": n.depth,
                        "n_node_samples": n.n_node_samples,
                        # Multi-group impurity is a JSON array; scalar stays a bare float
                        "impurity": json.dumps(n.impurity) if isinstance(n.impurity, list) else n.impurity,
                        "value": json.dumps(n.value),
                    })

    def _load_from_dir(self, work_dir: str) -> None:
        if not os.path.isdir(work_dir):
            return
        idx = 0
        while True:
            folder = os.path.join(work_dir, f"iteration-{idx}")
            if not os.path.isdir(folder):
                break
            with open(os.path.join(folder, "forest.json")) as f:
                raw = json.load(f)
            class_labels = raw.get("class_labels", [])   # flat list or {group: [labels]}
            class_colors = raw.get("class_colors", [])
            tree_meta: dict = raw.get("trees", raw)  # fall back to root for old format
            trees: dict[int, Tree] = {}
            for tid_str, m in tree_meta.items():
                tid = int(tid_str)
                nodes = []
                with open(os.path.join(folder, f"tree-{tid}.csv"), newline="") as f:
                    for row in csv.DictReader(f):
                        raw_imp = row["impurity"]
                        nodes.append(Node(
                            id=int(row["id"]),
                            depth=int(row["depth"]),
                            n_node_samples=int(row["n_node_samples"]),
                            # JSON array = per-group impurities; bare float = single group
                            impurity=json.loads(raw_imp) if raw_imp.lstrip().startswith("[") else float(raw_imp),
                            value=json.loads(row["value"]),
                        ))
                raw_metrics = m["metrics"]
                # old format was a list; convert to empty dict for compat
                metrics_dict = raw_metrics if isinstance(raw_metrics, dict) else {}
                trees[tid] = Tree(
                    nodes=nodes, height=m["height"], n_nodes=m["n_nodes"],
                    metrics=metrics_dict,
                    group_metrics=m.get("group_metrics", None),
                )
            correlation_matrix = raw.get("correlation_matrix", None)
            self._iterations.append(Forest(
                trees=trees,
                class_labels=class_labels,
                class_colors=class_colors,
                correlation_matrix=correlation_matrix,
                task=raw.get("task", "classification"),
                target_range=raw.get("target_range", None),
                std_range=raw.get("std_range", None),
                target_label=raw.get("target_label", ""),
                target_labels=raw.get("target_labels", None),
            ))

            # references.json (absent on pre-feature or reference-free
            # iterations → empty list, keeping _iteration_refs parallel).
            # PNG bytes stay on disk until first requested (_get_ref_png).
            snaps: list[ReferenceSnapshot] = []
            ref_file = os.path.join(folder, "references.json")
            if os.path.isfile(ref_file):
                with open(ref_file) as f:
                    for r in json.load(f):
                        snaps.append(ReferenceSnapshot(
                            id=r["id"], name=r.get("name"), label=r.get("label"),
                            label_index=r.get("label_index"), step=r.get("step"),
                            paths={int(tid): path for tid, path in r.get("paths", {}).items()},
                        ))
                        # Never mint an id that collides with a persisted PNG.
                        tail = r["id"].rsplit("-", 1)[-1]
                        if tail.isdigit():
                            self._ref_counter = max(self._ref_counter, int(tail) + 1)
            self._iteration_refs.append(snaps)

            idx += 1

    def _get_ref_png(self, ref_id: str) -> bytes | None:
        """PNG bytes for a reference id — from memory, else lazily from disk."""
        with self._lock:
            png = self._ref_png.get(ref_id)
        if png is not None:
            return png
        if self.work_dir:
            path = os.path.join(self.work_dir, "references", f"{ref_id}.png")
            if os.path.isfile(path):
                with open(path, "rb") as f:
                    png = f.read()
                with self._lock:
                    self._ref_png[ref_id] = png
                return png
        return None


# ── CLI entry point ───────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="TreeFlow visualization backend")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on (default: 8000)")
    parser.add_argument("--dir", type=str, default=".", metavar="DIR",
                        help="Run folder containing iteration-0/, iteration-1/, ... (default: .)")
    args = parser.parse_args()

    treeflow = TreeFlow(port=args.port, work_dir=args.dir)

    n = len(treeflow._iterations)
    if n:
        print(f"Loaded {n} iteration{'s' if n != 1 else ''} from '{args.dir}'.")
    else:
        print(f"No iteration-0/ folder found in '{args.dir}' - nothing to show. "
              "Folders are read once at startup; restart after adding data.")

    treeflow._start_server()
    print(f"Backend running at http://localhost:{args.port}  (Ctrl+C to stop)")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down...")
        treeflow.shutdown()


if __name__ == "__main__":
    main()
