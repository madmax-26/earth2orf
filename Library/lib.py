"""
TreeFlow Library — Common Random Forest interface for the TreeFlow visualizer.

Users implement one conversion function:

    def convert(my_rf) -> Forest:
        trees = {}
        for i, ... in enumerate(my_rf.estimators_):
            nodes = [
                Node(
                    id=j,
                    depth=depth_of_j,
                    n_node_samples=samples_at_j,
                    impurity=impurity_at_j,
                    value=[class_0_count, class_1_count, ...]  # or regression value
                )
                for j in range(n_nodes)
            ]
            trees[i] = Tree(nodes=nodes, height=max_depth, n_nodes=len(nodes), metrics={"accuracy": acc})
        return Forest(trees=trees)

Then call export_forest(forest, output_dir) to write files the backend can serve.
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from typing import Dict, List, Union


@dataclass
class Node:
    id: int
    depth: int
    n_node_samples: int
    # Single label group: one Gini (classification) / one std (regression).
    # Multi-group: a list with one entry per label group / regression target.
    # Mixed task: [gini per class group..., std per regression target...] in
    # global group order (class groups first, then targets).
    impurity: Union[float, List[float]]
    # Single group: class histogram, or [mean] for regression.
    # Multi-group classification: the CONCATENATION of every group's histogram,
    # in the insertion order of Forest.class_labels (offsets derive from label
    # counts). Multi-target regression: [mean_target_0, mean_target_1, ...].
    # Mixed task: [class-group histograms..., then one mean per target].
    value: List[Union[int, float]]


@dataclass
class Tree:
    nodes: List[Node]
    height: int
    n_nodes: int
    # Tree-global custom metrics (e.g. drifts, tree_age) — shown for every group.
    #
    # SCOPE: these describe THIS TREE ALONE, never the ensemble. An "accuracy"
    # here is what this single tree scores on its own; the forest predicts by
    # averaging all trees' posteriors and normally scores well above the mean of
    # its individual trees (that gap IS the ensemble effect, not a bug). Keep any
    # forest-level figure out of here — the frontend labels everything in this
    # dict as per-tree.
    metrics: Dict[str, float] = field(default_factory=dict)
    # Group-scoped custom metrics, keyed by label-group / target NAME ("" for the
    # unnamed single group): e.g. {"animal": {"accuracy": 0.91}, "temperature":
    # {"mae": 1.2}}. Lets accuracy attach to a classification group and mae to a
    # specific regression target in multi-output / mixed forests.
    group_metrics: Union[Dict[str, Dict[str, float]], None] = None


@dataclass
class Forest:
    trees: Dict[int, Tree]
    # Single label group: a flat list of labels (legacy, stays fully supported).
    # Multi-group classification: a dict {group_name: [labels...]}; dict insertion
    # order defines the group order and the value-vector offsets.
    class_labels: Union[List[str], Dict[str, List[str]]] = field(default_factory=list)
    # Parallel colors: flat list for the single-group case, or {group_name: [colors...]}.
    class_colors: Union[List[str], Dict[str, List[str]]] = field(default_factory=list)
    # Optional T×T correlation matrix (trees ordered by sorted key).
    # Values in [-1, 1]. Converted to distance as 1 - corr when used.
    correlation_matrix: Union[List[List[float]], None] = field(default=None)
    # Task type: "classification" (default), "regression", or "mixed".
    # For regression, each Node.value is [mean] and Node.impurity is the target std.
    # For mixed, class_labels MUST be a dict (its groups come first in the value
    # vector) and target_labels lists the regression targets appended after them;
    # group names must be non-numeric strings (JS object-key ordering).
    task: str = "classification"
    # Regression color domains (forest-wide). Single target: [min, max].
    # Multi-target: a list of [min, max] per target (aligned with target_labels).
    target_range: Union[List[float], List[List[float]], None] = field(default=None)
    std_range: Union[List[float], List[List[float]], None] = field(default=None)
    target_label: str = ""  # single-target name (legacy; still supported)
    # Multi-target regression: names of the targets, aligned with the per-node
    # value/impurity arrays. When set, takes precedence over target_label.
    target_labels: Union[List[str], None] = field(default=None)


def apply_label_map(forest: Forest, label_map: dict) -> Forest:
    """Rename class labels, group names, and regression target names for display.

    label_map maps existing names (raw values or their str() form) to the
    display strings shown in the legend, node widget, similarity axes, etc.:

        {0: "Car", 1: "Truck"}                       # flat: applies everywhere
        {"Class": {"0": "Car", "1": "Truck"}}        # nested: only within group "Class"
        {"temp": "Temperature [°C]"}                 # renames a group / target name

    Matching is exact first, then with surrounding whitespace stripped (River
    labels sometimes carry stray spaces, e.g. " Approved"). Unmatched names
    stay unchanged. Group-keyed structures (class_colors, group_metrics) are
    renamed in sync. Mutates the forest and returns it.
    """
    flat: dict[str, str] = {}
    nested: dict[str, dict[str, str]] = {}
    for k, v in label_map.items():
        if isinstance(v, dict):
            nested[str(k)] = {str(a): str(b) for a, b in v.items()}
        else:
            flat[str(k)] = str(v)

    def rename(name: str, group_table: Union[dict, None] = None) -> str:
        for table in (group_table or {}, flat):
            if name in table:
                return table[name]
            if name.strip() in table:
                return table[name.strip()]
        return name

    if isinstance(forest.class_labels, dict):
        forest.class_labels = {
            rename(g): [rename(lbl, nested.get(g)) for lbl in labels]
            for g, labels in forest.class_labels.items()
        }
    else:
        forest.class_labels = [rename(lbl) for lbl in forest.class_labels]

    # class_colors dict is keyed by group name — keep keys aligned
    if isinstance(forest.class_colors, dict):
        forest.class_colors = {rename(g): c for g, c in forest.class_colors.items()}

    if forest.target_label:
        forest.target_label = rename(forest.target_label)
    if forest.target_labels:
        forest.target_labels = [rename(t) for t in forest.target_labels]

    # group-scoped tree metrics are keyed by group / target name
    for tree in forest.trees.values():
        if tree.group_metrics:
            tree.group_metrics = {rename(g): m for g, m in tree.group_metrics.items()}

    return forest


def export_forest(forest: Forest, output_dir: str = ".") -> None:
    """Write forest.json and tree-N.csv files for the TreeFlow backend."""
    _write_forest_json(forest, output_dir)
    _write_tree_csvs(forest, output_dir)


def _tree_json(tree: Tree) -> dict:
    d = {"height": tree.height, "n_nodes": tree.n_nodes, "metrics": tree.metrics}
    if tree.group_metrics:
        d["group_metrics"] = tree.group_metrics
    return d


def _write_forest_json(forest: Forest, output_dir: str) -> None:
    data = {
        "class_labels": forest.class_labels,
        "class_colors": forest.class_colors,
        "task": forest.task,
        "trees": {str(tree_id): _tree_json(tree) for tree_id, tree in forest.trees.items()},
    }
    if forest.correlation_matrix is not None:
        data["correlation_matrix"] = forest.correlation_matrix
    if forest.task != "classification" or forest.target_labels is not None:
        data["target_range"] = forest.target_range
        data["std_range"] = forest.std_range
        data["target_label"] = forest.target_label
        if forest.target_labels is not None:
            data["target_labels"] = forest.target_labels
    path = os.path.join(output_dir, "forest.json")
    with open(path, "w") as f:
        json.dump(data, f, indent=4)


def _write_tree_csvs(forest: Forest, output_dir: str) -> None:
    fieldnames = ["id", "depth", "n_node_samples", "impurity", "value"]
    for tree_id, tree in forest.trees.items():
        path = os.path.join(output_dir, f"tree-{tree_id}.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for node in tree.nodes:
                writer.writerow({
                    "id": node.id,
                    "depth": node.depth,
                    "n_node_samples": node.n_node_samples,
                    # Per-group impurity is a JSON array; scalar stays a bare float
                    "impurity": json.dumps(node.impurity) if isinstance(node.impurity, list) else node.impurity,
                    "value": json.dumps(node.value),
                })
