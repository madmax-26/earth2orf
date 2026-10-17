"""
TreeFlow River interface — converts River classifiers to the TreeFlow Forest format.

Supported classifiers
---------------------
SRPClassifier  (river.ensemble)   — Streaming Random Patches
ARFClassifier  (river.forest)     — Adaptive Random Forest
AMFClassifier  (river.forest)     — Aggregated Mondrian Forest

Multi-output (multi-label / multi-target)
-----------------------------------------
convert_arf_ml   — MultiClassEncoder(ARFClassifier) from river.multioutput.
                   The encoder maps each unique label combination to one
                   multiclass code, so every tree carries ALL label groups in
                   its node stats; the converter decodes the codes back and
                   marginalizes per-group histograms onto the shared trees.
convert_isoup_reg — iSOUPTreeRegressor (river.tree), River's native
                   multi-target regression tree. Accepts one tree or a
                   hand-bagged list of trees.
MixedISOUPForest + convert_isoup_mixed — mixed classification AND regression
                   targets on the SAME trees (task="mixed"). Class groups are
                   one-hot-expanded into 0/1 indicator targets inside an
                   ARF-style Poisson-bagged ensemble of iSOUP trees; a node's
                   indicator mean × its sample count recovers the class
                   histogram, while raw targets keep their mean/std.

ClassifierChain / RegressorChain are NOT supported: a chain trains a separate
forest per output, and the TreeFlow multi-group format requires all groups to live
on the same tree nodes (Node.value is the concatenation of per-group stats).

SRP and ARF wrap Hoeffding trees and expose them differently:
  SRP: model.models[i].model  → HoeffdingTreeClassifier  (wrapper layer)
  ARF: model.models[i]        → BaseTreeClassifier (IS the tree already)
Both are handled by _tree_from_htree (uses to_dataframe + bottom-up
correction, because Hoeffding branch stats freeze at split time).

AMF is structurally different: its members are MondrianTreeClassifier
objects. Node size is read from `node.n_samples` (true samples routed
through, which never inverts parent < child), while the class histogram is
read from the node's own `counts`. Counts are NOT summed bottom-up from
leaves: a Mondrian split resets a branch's counts (River's `replant` copies
n_samples but not counts), so counts hold the node's local post-split class
proportions. Empty sibling leaves created by splits are pruned. Handled by
_tree_from_mondrian.

Typical usage:

    from river.forest import ARFClassifier
    from river.datasets import Elec2
    from river_interface import convert_arf
    from treeflow import TreeFlow

    arf = ARFClassifier(n_models=10, seed=42)
    treeflow = TreeFlow(port=8000)

    for i, (x, y) in enumerate(Elec2()):
        arf.learn_one(x, y)
        if i % 500 == 0:
            treeflow.push(convert_arf(arf, n_samples=i + 1))

    treeflow.shutdown()
"""

from __future__ import annotations

import math
import random
import warnings
from typing import TYPE_CHECKING

from lib import Forest, Node, Tree

if TYPE_CHECKING:
    from river.ensemble import SRPClassifier
    from river.forest import AMFClassifier, ARFClassifier


# ── Public converters ─────────────────────────────────────────────────────────

def convert_srp(srp: "SRPClassifier", n_samples: int | None = None, eval_data: list | None = None) -> Forest:
    """Convert a River SRPClassifier to a TreeFlow Forest.

    n_samples: number of real samples seen so far. When provided, each tree's
               counts are normalized so its root equals n_samples, regardless
               of Poisson resampling or drift-triggered resets. Recommended.
               When omitted, raw Poisson-weighted counts are returned.
    eval_data: optional list of (x, y) samples. When given, a T×T Pearson
               correlation matrix is computed from per-tree predictions and
               attached to the Forest, enabling the tree-map view.
    """
    if srp.models and not hasattr(srp.models[0].model, "classes"):
        raise TypeError(
            "convert_srp requires an SRPClassifier; got a regressor tree. "
            "Use convert_srp_reg() for SRPRegressor."
        )
    trees: dict[int, Tree] = {}
    class_labels: list[str] = []
    for i, base_learner in enumerate(srp.models):
        htree = base_learner.model
        if not class_labels and getattr(htree, "classes", None):
            class_labels = [str(c) for c in sorted(htree.classes, key=str)]
        metrics: dict[str, float] = {}
        if base_learner.metric:
            metrics["accuracy"] = base_learner.metric.get()
            metrics["drifts"] = float(base_learner.n_drifts_detected)
            metrics["warnings"] = float(base_learner.n_warnings_detected)
        trees[i] = _tree_from_htree(htree, metrics, n_samples=n_samples, lambda_value=srp.lam)
    corr = _correlation_matrix([bl.model for bl in srp.models], eval_data) if eval_data else None
    return Forest(trees=trees, class_labels=class_labels, correlation_matrix=corr)


def convert_arf(arf: "ARFClassifier", n_samples: int | None = None, eval_data: list | None = None) -> Forest:
    """Convert a River ARFClassifier to a TreeFlow Forest.

    n_samples: number of real samples seen so far. When provided, each tree's
               counts are normalized so its root equals n_samples, regardless
               of Poisson resampling or drift-triggered resets. Recommended.
               When omitted, raw Poisson-weighted counts are returned.
    eval_data: optional list of (x, y) samples. When given, a T×T Pearson
               correlation matrix is computed from per-tree predictions and
               attached to the Forest, enabling the tree-map view.
    """
    if arf.models and not hasattr(arf.models[0], "classes"):
        raise TypeError(
            "convert_arf requires an ARFClassifier; got a regressor tree. "
            "Use convert_arf_reg() for ARFRegressor."
        )
    trees: dict[int, Tree] = {}
    class_labels: list[str] = []
    for i, htree in enumerate(arf.models):
        if not class_labels and getattr(htree, "classes", None):
            class_labels = [str(c) for c in sorted(htree.classes, key=str)]
        # ARF member IS the Hoeffding tree — no .model wrapper, no .metric
        trees[i] = _tree_from_htree(htree, metrics={}, n_samples=n_samples, lambda_value=arf.lambda_value)
    corr = _correlation_matrix(list(arf.models), eval_data) if eval_data else None
    return Forest(trees=trees, class_labels=class_labels, correlation_matrix=corr)


def convert_amf(amf: "AMFClassifier", eval_data: list | None = None) -> Forest:
    """Convert a River AMFClassifier (Aggregated Mondrian Forest) to a TreeFlow Forest.

    Unlike SRP/ARF, AMF does no bagging — every tree sees every sample exactly
    once — so node counts are already in real-sample units and need no Poisson
    correction. Each Mondrian node keeps accumulating its own class histogram
    (the `counts` it predicts from), so node values are read directly instead
    of being recomputed bottom-up from the leaves.

    Call once per snapshot while streaming; each pushed iteration then shows
    the trees after they have learned more data.

    eval_data: optional list of (x, y) samples. When given, each tree's
               accuracy is added to its metrics and a T×T Pearson correlation
               matrix is computed from per-tree predictions for the tree-map view.
    """
    sorted_classes = sorted(amf._classes, key=str)
    class_labels = [str(c) for c in sorted_classes]
    class_pos = {c: j for j, c in enumerate(sorted_classes)}

    mtrees = list(amf)
    trees: dict[int, Tree] = {}
    for i, mtree in enumerate(mtrees):
        metrics: dict[str, float] = {}
        if eval_data:
            correct = sum(_predict_label(mtree, x) == y for x, y in eval_data)
            metrics["accuracy"] = correct / len(eval_data)
        trees[i] = _tree_from_mondrian(mtree, class_pos, len(sorted_classes), metrics)
    corr = _correlation_matrix(mtrees, eval_data) if eval_data else None
    return Forest(trees=trees, class_labels=class_labels, correlation_matrix=corr)


# ── Regression converters ─────────────────────────────────────────────────────

def convert_arf_reg(
    arf, n_samples: int | None = None,
    eval_data: list | None = None, target_label: str = "",
) -> Forest:
    """Convert a River ARFRegressor to a regression TreeFlow Forest.

    Each node carries value=[mean target] and impurity=target std dev. n_samples
    (real samples seen) rescales node counts as in the classifier path. eval_data,
    when given, adds per-tree continuous-prediction correlation for the Forest Map.
    target_label names the regression target (legend caption).
    """
    trees: dict[int, Tree] = {}
    for i, htree in enumerate(arf.models):
        trees[i] = _tree_from_htree_reg(
            htree, metrics={}, n_samples=n_samples,
            lambda_value=getattr(arf, "lambda_value", None),
        )
    forest = Forest(trees=trees, task="regression", target_label=target_label)
    _finalize_regression(forest, list(arf.models), eval_data)
    return forest


def convert_srp_reg(
    srp, n_samples: int | None = None,
    eval_data: list | None = None, target_label: str = "",
) -> Forest:
    """Convert a River SRPRegressor to a regression TreeFlow Forest. See convert_arf_reg."""
    trees: dict[int, Tree] = {}
    for i, base_learner in enumerate(srp.models):
        htree = base_learner.model
        metrics: dict[str, float] = {}
        if getattr(base_learner, "metric", None):
            metrics["mae"] = base_learner.metric.get()
        if getattr(base_learner, "n_drifts_detected", None) is not None:
            metrics["drifts"] = float(base_learner.n_drifts_detected)
            metrics["warnings"] = float(base_learner.n_warnings_detected)
        trees[i] = _tree_from_htree_reg(
            htree, metrics, n_samples=n_samples,
            lambda_value=getattr(srp, "lam", None),
        )
    forest = Forest(trees=trees, task="regression", target_label=target_label)
    _finalize_regression(forest, [bl.model for bl in srp.models], eval_data)
    return forest


def _finalize_regression(forest: Forest, tree_models: list, eval_data: list | None) -> None:
    """Fill forest-wide target_range / std_range from node stats and attach the
    continuous-prediction correlation matrix when eval_data is provided."""
    means = [n.value[0] for t in forest.trees.values() for n in t.nodes if n.value]
    stds = [n.impurity for t in forest.trees.values() for n in t.nodes]
    if means:
        forest.target_range = [min(means), max(means)]
    if stds:
        forest.std_range = [0.0, max(stds)]
    if eval_data:
        forest.correlation_matrix = _correlation_matrix_reg(tree_models, eval_data)


# ── Multi-output converters ───────────────────────────────────────────────────

def convert_arf_ml(encoder, n_samples: int | None = None, eval_data: list | None = None) -> Forest:
    """Convert a river.multioutput.MultiClassEncoder wrapping an ARFClassifier
    to a multi-group TreeFlow Forest.

    The encoder assigns one multiclass code per unique label combination, so the
    wrapped ARF's tree nodes hold combo-code histograms. Each code is decoded back
    into its per-group labels via the encoder's reverse map and its count is
    credited to one slot per group — yielding, on the SAME trees, the concatenated
    per-group histograms and per-group Gini list the multi-group format expects.

    n_samples: real samples seen so far; rescales node counts as in convert_arf.
    eval_data: optional list of (x, y) samples for the tree correlation matrix
               (per-tree predicted combo codes, Pearson-correlated).
    """
    if not hasattr(encoder, "_r_label_map"):
        raise TypeError(
            "convert_arf_ml expects a river.multioutput.MultiClassEncoder "
            f"wrapping an ARFClassifier; got {type(encoder).__name__}. "
            "For single-output models use convert_arf / convert_arf_reg."
        )
    arf = encoder.model

    # Observed values per label group, from the encoder's combo registry.
    seen: dict[str, set] = {}
    for combo in encoder._r_label_map.values():
        for label_id, val in combo:
            seen.setdefault(str(label_id), set()).add(val)
    group_names = sorted(seen)
    sorted_vals = {g: sorted(seen[g], key=str) for g in group_names}
    class_labels = {g: [str(v) for v in sorted_vals[g]] for g in group_names}

    # A continuous target smuggled into the label dict registers (almost) every
    # sample as a new combo code; the Hoeffding bound then never justifies a
    # split and every tree stays a stump. Catch it early and loudly.
    for g in group_names:
        if len(sorted_vals[g]) > 25:
            warnings.warn(
                f"label group '{g}' has {len(sorted_vals[g])} distinct values — "
                "it looks continuous. MultiClassEncoder only handles discrete "
                "labels; bin the value before learn_one (e.g. deciles), or model "
                "it as a regression target instead. With this many combo classes "
                "the Hoeffding trees will barely split (stump forests).",
                stacklevel=2,
            )
    if len(encoder._r_label_map) > 100:
        warnings.warn(
            f"{len(encoder._r_label_map)} label combinations registered — the "
            "joint label space is too fine for 10³-sample streams; expect very "
            "shallow trees. Use coarser bins or fewer label groups.",
            stacklevel=2,
        )

    # combo code → one flat index per group into the concatenated value vector
    offsets: dict[str, int] = {}
    off = 0
    for g in group_names:
        offsets[g] = off
        off += len(sorted_vals[g])
    value_len = off
    val_idx = {g: {v: k for k, v in enumerate(sorted_vals[g])} for g in group_names}
    code_slots = {
        code: [offsets[str(lid)] + val_idx[str(lid)][v] for lid, v in combo]
        for code, combo in encoder._r_label_map.items()
    }
    group_spans = [(offsets[g], len(sorted_vals[g])) for g in group_names]

    trees: dict[int, Tree] = {}
    for i, htree in enumerate(arf.models):
        trees[i] = _tree_from_htree_ml(
            htree, code_slots, group_spans, value_len,
            n_samples=n_samples, lambda_value=getattr(arf, "lambda_value", None),
        )
    corr = _correlation_matrix(list(arf.models), eval_data) if eval_data else None
    return Forest(trees=trees, class_labels=class_labels, correlation_matrix=corr)


def convert_isoup_reg(
    models, n_samples: int | None = None,
    eval_data: list | None = None, target_labels: list | None = None,
) -> Forest:
    """Convert iSOUPTreeRegressor(s) to a multi-target regression TreeFlow Forest.

    iSOUP is River's native multi-target tree: every node tracks a Var per
    target, so all targets live on the same tree structure. ARFRegressor cannot
    do this (RegressorChain trains one separate forest per target); to get a
    forest, train several iSOUP trees on Poisson-resampled copies of the stream
    and pass them as a list.

    models:        a single iSOUPTreeRegressor or a list of them.
    n_samples:     real samples seen; rescales node counts as in convert_arf_reg.
    eval_data:     optional (x, y) list; per-target prediction correlations are
                   averaged into the tree correlation matrix.
    target_labels: explicit target order; defaults to the sorted union of the
                   trees' targets.
    """
    if not isinstance(models, (list, tuple)):
        models = [models]
    targets = list(target_labels) if target_labels else sorted(
        {t for m in models for t in getattr(m, "targets", set())}, key=str
    )

    trees = {
        i: _tree_from_isoup(m, targets, n_samples=n_samples)
        for i, m in enumerate(models)
    }
    forest = Forest(
        trees=trees, task="regression",
        target_labels=[str(t) for t in targets],
    )

    # Per-target color domains (list of [min, max] aligned with target_labels).
    K = len(targets)
    means: list[list[float]] = [[] for _ in range(K)]
    stds: list[list[float]] = [[] for _ in range(K)]
    for t in forest.trees.values():
        for nd in t.nodes:
            for k in range(K):
                if k < len(nd.value):
                    means[k].append(nd.value[k])
                if isinstance(nd.impurity, list) and k < len(nd.impurity):
                    stds[k].append(nd.impurity[k])
    if any(means):
        forest.target_range = [[min(m), max(m)] if m else [0.0, 0.0] for m in means]
        forest.std_range = [[0.0, max(s)] if s else [0.0, 0.0] for s in stds]
    if eval_data:
        forest.correlation_matrix = _correlation_matrix_reg_multi(models, eval_data, targets)
    return forest


# ── Mixed-task ensemble + converter ───────────────────────────────────────────

def _indicator_name(group, value) -> str:
    """iSOUP target name of one class indicator. Shared by MixedISOUPForest and
    convert_isoup_mixed so the trainer and the converter can never drift."""
    return f"{group}={value}"


class MixedISOUPForest:
    """ARF-style bagged iSOUP forest for MIXED classification + regression targets.

    Each member is an iSOUPTreeRegressor (River's multi-target tree), trained by
    online bagging: every sample is weighted Poisson(lambda_value) per member,
    exactly like ARF's resampling. Unlike ARFRegressor there is NO ADWIN drift
    reset and NO random feature subsetting — on non-stationary streams the trees
    go stale until retrained.

    Class groups are one-hot-expanded: a declared group {"Class": [0, 1, 2]}
    becomes indicator targets "Class=0"/"Class=1"/"Class=2" fed 1.0/0.0 every
    step, so at any tree node the indicator's mean is the class fraction — that
    is what convert_isoup_mixed turns back into per-node class histograms.

    class_groups: {group_name: [all label values]} — the label list is a hard
                  contract; learn_one raises on an undeclared value (silent
                  auto-extension would desynchronize indicator sets across
                  already-grown nodes). Group names must be non-numeric strings.
    reg_targets:  names of continuous targets read from y unchanged.
    reg_scale:    {target: factor} multiplied onto a target before learning.
                  LOAD-BEARING for split quality: iSOUP's split merit averages
                  raw per-target variances, so an O(100) target drowns the 0/1
                  indicators (variance <= 0.25). Scale continuous targets to
                  roughly unit range (e.g. {"Popularity": 0.01} for a 0-100
                  target); convert_isoup_mixed and predict_one divide back.
    model:        optional iSOUP prototype to clone per member (isoup_params are
                  ignored then); a fresh iSOUPTreeRegressor(**isoup_params)
                  otherwise.
    """

    def __init__(
        self,
        class_groups: dict,
        reg_targets: list | None = None,
        n_models: int = 10,
        lambda_value: float = 6.0,
        seed: int | None = None,
        reg_scale: dict | None = None,
        model=None,
        **isoup_params,
    ):
        from river.tree import iSOUPTreeRegressor
        from river.utils.random import poisson

        if not class_groups and not reg_targets:
            raise ValueError("MixedISOUPForest needs at least one class group or regression target")
        self.class_groups = {g: list(vals) for g, vals in (class_groups or {}).items()}
        for g, vals in self.class_groups.items():
            if not vals:
                raise ValueError(f"class group '{g}' has an empty label list")
        self.reg_targets = list(reg_targets or [])
        self.reg_scale = dict(reg_scale or {})
        self.n_models = n_models
        self.lambda_value = lambda_value
        self.seed = seed
        self._poisson = poisson

        # {group: {declared_value: indicator target name}} — original values as
        # keys (hash(5.0) == hash(5), so float/int label mismatches still hit).
        self._indicators = {
            g: {v: _indicator_name(g, v) for v in vals}
            for g, vals in self.class_groups.items()
        }

        proto = model if model is not None else iSOUPTreeRegressor(**isoup_params)
        self.models = [proto.clone() for _ in range(n_models)]
        self._rngs = [
            random.Random(seed + i) if seed is not None else random.Random()
            for i in range(n_models)
        ]

    def learn_one(self, x: dict, y: dict):
        """y = {group_or_target: value}, e.g. {"Class": 5.0, "Popularity": 60.0}."""
        ye: dict = {}
        for g, slots in self._indicators.items():
            if g not in y:
                raise KeyError(f"y is missing class group '{g}'")
            v = y[g]
            if v not in slots:
                raise ValueError(
                    f"value {v!r} was not declared for class group '{g}' "
                    f"(declared: {list(slots)})"
                )
            # Full one-hot EVERY step: all indicators see every sample, keeping
            # per-target n equal at every node — the pooled-fraction math in
            # convert_isoup_mixed relies on that.
            for val, name in slots.items():
                ye[name] = 1.0 if val == v else 0.0
        for t in self.reg_targets:
            if t not in y:
                raise KeyError(f"y is missing regression target '{t}'")
            ye[t] = float(y[t]) * self.reg_scale.get(t, 1.0)

        for m, rng in zip(self.models, self._rngs):
            k = self._poisson(self.lambda_value, rng)
            if k > 0:
                m.learn_one(x, ye, w=k)
        return self

    def predict_one(self, x: dict) -> dict:
        """Member-averaged prediction: argmax label per class group, un-scaled
        mean per regression target. {} while all members are empty."""
        sums: dict = {}
        voters = 0
        for m in self.models:
            p = m.predict_one(x)
            if not p:
                continue
            voters += 1
            for t, v in p.items():
                sums[t] = sums.get(t, 0.0) + v
        if voters == 0:
            return {}
        out: dict = {}
        for g, slots in self._indicators.items():
            out[g] = max(slots, key=lambda val: sums.get(slots[val], 0.0))
        for t in self.reg_targets:
            f = self.reg_scale.get(t, 1.0) or 1.0
            out[t] = (sums.get(t, 0.0) / voters) / f
        return out


def convert_isoup_mixed(
    model, n_samples: int | None = None, eval_data: list | None = None,
    class_groups: dict | None = None, reg_targets: list | None = None,
    reg_scale: dict | None = None,
) -> Forest:
    """Convert a MixedISOUPForest (or a raw list of iSOUP trees trained on
    indicator + regression targets) to a task="mixed" TreeFlow Forest.

    Emits, on the SAME tree nodes, one histogram per class group (indicator
    mean × node n per label) followed by one mean per regression target;
    impurity = [Gini per class group..., std per target...]. Regression means
    and stds are divided by reg_scale so the visualizer shows natural units.

    model:      MixedISOUPForest (class_groups/reg_targets/reg_scale read from
                it) or a list of iSOUPTreeRegressor (then class_groups — and
                reg_targets if any — must be passed explicitly).
    n_samples:  real samples seen; rescales node counts as in convert_arf.
    eval_data:  optional (x, y) list; per-target prediction Pearson matrices
                (indicators included) are averaged into the tree correlation
                matrix, and per-tree accuracy/MAE metrics are computed.
    """
    lambda_value = None
    if hasattr(model, "models") and hasattr(model, "class_groups"):
        models = list(model.models)
        class_groups = model.class_groups
        reg_targets = model.reg_targets
        reg_scale = model.reg_scale
        lambda_value = model.lambda_value
    else:
        models = list(model) if isinstance(model, (list, tuple)) else [model]
        if class_groups is None:
            raise TypeError(
                "convert_isoup_mixed needs a MixedISOUPForest, or class_groups "
                "passed explicitly alongside a raw list of iSOUP trees"
            )
        reg_targets = list(reg_targets or [])
    reg_scale = dict(reg_scale or {})

    group_names = list(class_groups)  # dict insertion order = group order
    indicator_targets = [
        _indicator_name(g, v) for g in group_names for v in class_groups[g]
    ]
    targets = indicator_targets + list(reg_targets)
    n_ind = len(indicator_targets)
    n_groups = len(group_names)

    group_spans: list[tuple[int, int]] = []
    off = 0
    for g in group_names:
        group_spans.append((off, len(class_groups[g])))
        off += len(class_groups[g])

    def emit(parts: list, scale: float) -> tuple[list, list]:
        # parts[j] = pooled (n, mean, var) of targets[j] at this node.
        # Indicator mean == class fraction, so n*mean == class mass. Never round
        # per slot: rounding zeroes small classes and breaks mass conservation.
        value = [max(0.0, parts[j][0] * parts[j][1]) * scale for j in range(n_ind)]
        imp = [_gini_from_values(value[o:o + s]) for o, s in group_spans]
        for k, t in enumerate(reg_targets):
            _, mean_k, var_k = parts[n_ind + k]
            f = reg_scale.get(t, 1.0) or 1.0
            value.append(mean_k / f)
            imp.append(math.sqrt(max(0.0, var_k)) / f)
        return value, imp

    trees: dict[int, Tree] = {}
    for i, m in enumerate(models):
        metrics = _mixed_tree_metrics(m, eval_data, class_groups, reg_targets, reg_scale) if eval_data else {}
        trees[i] = _tree_from_isoup_generic(
            m, targets, n_samples=n_samples, emit=emit,
            lambda_value=lambda_value, metrics=metrics,
        )

    forest = Forest(
        trees=trees, task="mixed",
        class_labels={str(g): [str(v) for v in class_groups[g]] for g in group_names},
        target_labels=[str(t) for t in reg_targets],
    )

    # Per-target color domains, scanned at the mixed value/impurity offsets.
    K = len(reg_targets)
    if K:
        means: list[list[float]] = [[] for _ in range(K)]
        stds: list[list[float]] = [[] for _ in range(K)]
        for t in forest.trees.values():
            for nd in t.nodes:
                for k in range(K):
                    if n_ind + k < len(nd.value):
                        means[k].append(nd.value[n_ind + k])
                    if isinstance(nd.impurity, list) and n_groups + k < len(nd.impurity):
                        stds[k].append(nd.impurity[n_groups + k])
        if any(means):
            forest.target_range = [[min(m), max(m)] if m else [0.0, 0.0] for m in means]
            forest.std_range = [[0.0, max(s)] if s else [0.0, 0.0] for s in stds]

    _warn_on_scale_imbalance(models, reg_targets, n_ind, targets)

    if eval_data:
        # One task per class GROUP (all its indicators share it) and one per
        # regression target, aligned with `targets` — so an 11-class group and
        # a single regression target contribute equally to the whole-tree map.
        tasks = ([("class", g) for g in group_names for _ in class_groups[g]]
                 + [("reg", t) for t in reg_targets])
        forest.correlation_matrix = _correlation_matrix_reg_multi(
            models, eval_data, targets, tasks=tasks)
    return forest


def _mixed_tree_metrics(mtree, eval_data, class_groups, reg_targets, reg_scale) -> dict:
    """Per-tree metrics on eval_data: argmax accuracy per class group, MAE per
    regression target (single group/target drops the name suffix)."""
    slots = {g: [(v, _indicator_name(g, v)) for v in vals] for g, vals in class_groups.items()}
    correct = {g: 0 for g in class_groups}
    abs_err = {t: 0.0 for t in reg_targets}
    n = 0
    for x, y in eval_data:
        p = mtree.predict_one(x)
        if not p:
            continue
        n += 1
        for g, pairs in slots.items():
            pred = max(pairs, key=lambda pair: p.get(pair[1], 0.0))[0]
            if g in y and pred == y[g]:
                correct[g] += 1
        for t in reg_targets:
            f = reg_scale.get(t, 1.0) or 1.0
            if t in y:
                abs_err[t] += abs(p.get(t, 0.0) / f - float(y[t]))
    if n == 0:
        return {}
    metrics: dict[str, float] = {}
    for g in class_groups:
        key = "accuracy" if len(class_groups) == 1 else f"accuracy {g}"
        metrics[key] = correct[g] / n
    for t in reg_targets:
        key = "mae" if len(reg_targets) == 1 else f"mae {t}"
        metrics[key] = abs_err[t] / n
    return metrics


def _warn_on_scale_imbalance(models, reg_targets, n_ind, targets) -> None:
    """Warn when a (scaled) regression target's variance dwarfs the indicators'.

    iSOUP's split merit averages RAW per-target variances; indicators cap at
    0.25, so a target variance of >> 1 means splits are chosen for the target
    alone and the class groups ride along unsplit. Checked on root stats."""
    if not reg_targets:
        return
    for m in models:
        root = getattr(m, "_root", None)
        if root is None or not getattr(root, "stats", None):
            continue
        for k, t in enumerate(reg_targets):
            try:
                _, _, var = _read_var(root.stats[targets[n_ind + k]])
            except (KeyError, TypeError, AttributeError):
                continue
            if var > 25.0:  # std > 5 vs indicator std <= 0.5
                warnings.warn(
                    f"regression target '{t}' has variance {var:.3g} in the "
                    "learned (scaled) space — it dominates the 0/1 class "
                    "indicators in iSOUP's split merit, so trees will mostly "
                    "ignore the class groups. Pass reg_scale to bring it to "
                    "roughly unit range (e.g. {'"
                    f"{t}': {1.0 / max(math.sqrt(var), 1e-9):.2g}" "}).",
                    stacklevel=3,
                )
        break  # one representative tree is enough


# ── Correlation helper ────────────────────────────────────────────────────────

def _correlation_matrix(tree_models: list, eval_data: list) -> list[list[float]] | None:
    """Pearson correlation between per-tree prediction vectors over eval_data.

    Returns a T×T matrix in [-1, 1], or None if fewer than 2 trees or no data.
    Each tree model must support predict_proba_one(x) -> dict[class, float].
    """
    T = len(tree_models)
    if T < 2 or not eval_data:
        return None

    classes: list | None = None
    preds: list[list[float]] = [[] for _ in range(T)]
    for x, _ in eval_data:
        for t, tree in enumerate(tree_models):
            proba = tree.predict_proba_one(x)
            if not proba:
                preds[t].append(0.0)
                continue
            if classes is None:
                classes = sorted(proba.keys(), key=str)
            pred_label = max(proba, key=proba.get)
            preds[t].append(float(classes.index(pred_label)))

    n = len(eval_data)
    return _pearson_matrix(preds, n)


def _correlation_matrix_reg(tree_models: list, eval_data: list) -> list[list[float]] | None:
    """Pearson correlation between per-tree CONTINUOUS prediction vectors over eval_data.

    Regression analogue of _correlation_matrix: uses raw predict_one(x) values instead
    of class indices. Returns a T×T matrix in [-1, 1], or None if < 2 trees / no data.
    """
    T = len(tree_models)
    if T < 2 or not eval_data:
        return None
    preds: list[list[float]] = [[] for _ in range(T)]
    for x, _ in eval_data:
        for t, tree in enumerate(tree_models):
            try:
                p = tree.predict_one(x)
            except Exception:
                p = 0.0
            preds[t].append(float(p) if p is not None else 0.0)
    return _pearson_matrix(preds, len(eval_data))


def _correlation_matrix_reg_multi(
    tree_models: list, eval_data: list, targets: list, tasks: list | None = None,
) -> list[list[float]] | None:
    """Multi-target analogue of _correlation_matrix_reg.

    Each tree predicts a dict {target: value}; one Pearson matrix is computed per
    target (so differing target scales cannot dominate), then combined into a
    single T×T result in [-1, 1].

    tasks: one key per target, naming the task it belongs to. Matrices are
           averaged WITHIN each task first, then ACROSS tasks, so every task
           weighs the same however many targets it spans. None means each target
           is its own task (a plain mean — right for pure multi-target
           regression, where one target already is one task).

           This matters for mixed forests. A class group is expanded into one
           indicator target per class, so a plain mean over targets hands an
           11-class group 11 votes against a single regression target's one —
           the "whole-tree" similarity would then be ~92% classification. The
           Forest Map is meant to describe the tree across ALL its tasks, so
           each task gets an equal share instead.
    """
    T = len(tree_models)
    K = len(targets)
    if T < 2 or not eval_data or K == 0:
        return None
    n = len(eval_data)
    # series[k][t] = tree t's predictions for target k over eval_data
    series = [[[0.0] * n for _ in range(T)] for _ in range(K)]
    for s, (x, _) in enumerate(eval_data):
        for t, tree in enumerate(tree_models):
            try:
                p = tree.predict_one(x) or {}
            except Exception:
                p = {}
            for k, target in enumerate(targets):
                v = p.get(target, 0.0)
                series[k][t][s] = float(v) if v is not None else 0.0
    mats = [_pearson_matrix(series[k], n) for k in range(K)]

    def mean_of(ms: list) -> list[list[float]]:
        return [[sum(m[i][j] for m in ms) / len(ms) for j in range(T)] for i in range(T)]

    if tasks is None:
        return mean_of(mats)
    by_task: dict = {}
    for k, key in enumerate(tasks):
        by_task.setdefault(key, []).append(mats[k])
    return mean_of([mean_of(ms) for ms in by_task.values()])


def _pearson_matrix(preds: list[list[float]], n: int) -> list[list[float]]:
    """T×T Pearson correlation matrix from per-tree value vectors (shared core)."""
    T = len(preds)
    means = [sum(p) / n for p in preds]
    stds = [math.sqrt(sum((v - means[t]) ** 2 for v in preds[t]) / n) for t in range(T)]
    mat = [[1.0 if i == j else 0.0 for j in range(T)] for i in range(T)]
    for i in range(T):
        for j in range(i + 1, T):
            if stds[i] == 0.0 or stds[j] == 0.0:
                corr = 0.0
            else:
                cov = sum((preds[i][k] - means[i]) * (preds[j][k] - means[j]) for k in range(n)) / n
                corr = max(-1.0, min(1.0, cov / (stds[i] * stds[j])))
            mat[i][j] = corr
            mat[j][i] = corr
    return mat


# ── Mondrian (AMF) helpers ──────────────────────────────────────────────────────

def _predict_label(mtree, x):
    """Most likely class for one sample from a single Mondrian tree."""
    proba = mtree.predict_proba_one(x)
    return max(proba, key=proba.get) if proba else None


def _tree_from_mondrian(
    mtree,
    class_pos: dict,
    n_classes: int,
    metrics: dict[str, float],
) -> Tree:
    """Build a TreeFlow Tree from a River MondrianTreeClassifier (DFS preorder).

    class_pos: maps each class label to its column in the forest-wide value
               vector (a tree's local counts are indexed by its own
               _idx_to_class order, which need not match the forest order).
    """
    root = mtree._root
    if root is None or root.n_samples == 0:
        return Tree(nodes=[], height=0, n_nodes=0, metrics=metrics)

    idx_to_class = mtree._idx_to_class

    # DFS preorder (parent, then left subtree, then right subtree) carrying the
    # depth explicitly, so the emitted depth is always parent_depth + 1 — what
    # the frontend's buildHierarchy assumes — regardless of node.depth.
    #
    # Prune any node with no samples: every Mondrian split creates a fresh
    # sibling leaf that stays empty until a sample reaches it, and these render
    # as meaningless zero-count nodes. A node's children can never have seen
    # more samples than the node itself, so skipping a zero-sample node never
    # drops a non-empty descendant.
    order: list = []                 # (node, depth)
    stack = [(root, 0)]
    while stack:
        node, depth = stack.pop()
        if node.n_samples == 0:
            continue
        order.append((node, depth))
        if not node.is_leaf:
            left, right = node.children
            stack.append((right, depth + 1))   # push right first → left popped first
            stack.append((left, depth + 1))

    nodes: list[Node] = []
    height = 0
    for nid, (node, depth) in enumerate(order):
        # Class histogram for the pie/impurity. On a Mondrian branch this is the
        # node's own post-split counts (River's `replant` does not carry counts
        # across a split), so it captures the class *proportions* the node
        # predicts from, not an absolute total.
        value = [0.0] * n_classes
        for local_idx, cnt in enumerate(node.counts):
            if cnt:
                value[class_pos[idx_to_class[local_idx]]] += cnt
        # Size/branch-width use n_samples (true samples routed through the node).
        # Unlike sum(counts), this never inverts parent < child, so the
        # botanical layout's child/parent ratios stay within [0, 1].
        nodes.append(Node(
            id=nid,
            depth=depth,
            n_node_samples=node.n_samples,
            impurity=_gini_from_values(value),
            value=value,
        ))
        height = max(height, depth)

    return Tree(nodes=nodes, height=height, n_nodes=len(nodes), metrics=metrics)


def _gini_from_values(value: list) -> float:
    total = sum(value)
    if total == 0.0:
        return 0.0
    return 1.0 - sum((v / total) ** 2 for v in value)


# ── Shared helpers ────────────────────────────────────────────────────────────

def _tree_from_htree(
    htree,
    metrics: dict[str, float],
    n_samples: int | None = None,
    lambda_value: float | None = None,
) -> Tree:
    """Build a TreeFlow Tree from a River HoeffdingTree-compatible object.

    n_samples:    when provided, all node counts are scaled so the root equals
                  n_samples (corrects for Poisson resampling and drift resets).
    lambda_value: Poisson lambda used by the ensemble; when provided, tree_age
                  (estimated real samples seen by this tree) is added to metrics.
    """
    if htree._root is None:
        return Tree(nodes=[], height=0, n_nodes=0, metrics=metrics)

    df = htree.to_dataframe()

    if df is None:
        # Single root leaf — to_dataframe() only traverses branch edges.
        stats = htree._root.stats or {}
        classes = sorted(stats.keys(), key=str)
        raw_root = sum(stats.values())
        if lambda_value and raw_root > 0:
            metrics = {**metrics, "tree_age": round(raw_root / lambda_value)}
        scale = (n_samples / raw_root) if (n_samples is not None and raw_root > 0) else 1.0
        node = Node(
            id=0,
            depth=0,
            n_node_samples=round(raw_root * scale),
            impurity=_gini(stats),
            value=[stats.get(c, 0.0) * scale for c in classes],
        )
        return Tree(nodes=[node], height=0, n_nodes=1, metrics=metrics)

    classes = sorted(htree.classes, key=str)

    # River assigns node IDs in BFS order (root=0, then level by level).
    # Children always get higher IDs than their parent.
    rows = {nid: row for nid, row in df.iterrows()}

    # ── Step 1: adjacency map ─────────────────────────────────────────────────
    children: dict[int, list[int]] = {nid: [] for nid in rows}
    root_id: int = next(iter(rows))  # lowest ID = root

    for nid, row in rows.items():
        pid = row["parent"]
        try:
            children[int(pid)].append(nid)
        except (ValueError, TypeError):
            root_id = nid  # pd.NA → root node

    # ── Step 2: bottom-up sample correction ──────────────────────────────────
    # Branch stats are frozen at split time while leaf stats keep accumulating.
    # Processing IDs from high→low guarantees every child is corrected before
    # its parent (because children always have higher IDs).
    corrected: dict[int, dict] = {}
    for nid in sorted(rows, reverse=True):
        if rows[nid]["is_leaf"]:
            corrected[nid] = dict(rows[nid]["stats"])
        else:
            merged: dict = {}
            for cid in children[nid]:
                for cls, cnt in corrected[cid].items():
                    merged[cls] = merged.get(cls, 0.0) + cnt
            corrected[nid] = merged

    # ── Step 3: compute scale factor + tree_age ──────────────────────────────
    raw_root = sum(corrected[root_id].values())
    if lambda_value and raw_root > 0:
        metrics = {**metrics, "tree_age": round(raw_root / lambda_value)}
    scale = (n_samples / raw_root) if (n_samples is not None and raw_root > 0) else 1.0

    # ── Step 4: DFS preorder ──────────────────────────────────────────────────
    # Frontend buildHierarchy requires nodes in DFS preorder (parent before its
    # subtree, left subtree before right subtree).
    dfs_order: list[int] = []
    stack = [root_id]
    while stack:
        nid = stack.pop()
        dfs_order.append(nid)
        for cid in reversed(children[nid]):   # push right first → left popped first
            stack.append(cid)

    nodes = [
        Node(
            id=int(nid),
            depth=int(rows[nid]["depth"]),
            n_node_samples=round(sum(corrected[nid].values()) * scale),
            impurity=_gini(corrected[nid]),
            value=[corrected[nid].get(c, 0.0) * scale for c in classes],
        )
        for nid in dfs_order
    ]

    return Tree(
        nodes=nodes,
        height=htree.height or 0,
        n_nodes=htree.n_nodes,
        metrics=metrics,
    )


def _gini(stats: dict) -> float:
    total = sum(stats.values())
    if total == 0.0:
        return 0.0
    return 1.0 - sum((v / total) ** 2 for v in stats.values())


# ── Regression tree builder ─────────────────────────────────────────────────────

def _read_var(v) -> tuple[float, float, float]:
    """Read (n, mean, var) from a River stats.Var leaf-target estimator."""
    try:
        return (float(v.mean.n), float(v.mean.get()), max(0.0, float(v.get())))
    except (AttributeError, TypeError):
        return (0.0, 0.0, 0.0)


def _combine_stats(parts: list[tuple[float, float, float]]) -> tuple[float, float, float]:
    """Pool a list of (n, mean, var) into one (n, mean, var) via the parallel-variance
    formula: combined second moment minus the squared combined mean."""
    N = sum(p[0] for p in parts)
    if N <= 0:
        return (0.0, 0.0, 0.0)
    mean = sum(p[0] * p[1] for p in parts) / N
    m2 = sum(p[0] * (p[2] + p[1] * p[1]) for p in parts) / N
    return (N, mean, max(0.0, m2 - mean * mean))


def _tree_from_htree_reg(
    htree,
    metrics: dict[str, float],
    n_samples: int | None = None,
    lambda_value: float | None = None,
) -> Tree:
    """Build a regression TreeFlow Tree from a River HoeffdingTreeRegressor-compatible object.

    Mirrors _tree_from_htree but the per-node statistic is a target Var (mean + variance)
    rather than a class histogram. Branch Vars freeze at split time while leaf Vars keep
    accumulating, so internal-node stats are recomputed bottom-up from the leaves (pooled
    mean + variance). Emits value=[mean], impurity=sqrt(var)."""
    if htree._root is None:
        return Tree(nodes=[], height=0, n_nodes=0, metrics=metrics)

    df = htree.to_dataframe()

    if df is None:
        # Single root leaf — to_dataframe() only traverses branch edges.
        n_raw, mean, var = _read_var(htree._root.stats)
        if lambda_value and n_raw > 0:
            metrics = {**metrics, "tree_age": round(n_raw / lambda_value)}
        scale = (n_samples / n_raw) if (n_samples is not None and n_raw > 0) else 1.0
        node = Node(
            id=0, depth=0,
            n_node_samples=round(n_raw * scale),
            impurity=math.sqrt(var),
            value=[mean],
        )
        return Tree(nodes=[node], height=0, n_nodes=1, metrics=metrics)

    rows = {nid: row for nid, row in df.iterrows()}

    # Adjacency (River assigns children higher IDs than their parent).
    children: dict[int, list[int]] = {nid: [] for nid in rows}
    root_id: int = next(iter(rows))
    for nid, row in rows.items():
        pid = row["parent"]
        try:
            children[int(pid)].append(nid)
        except (ValueError, TypeError):
            root_id = nid

    # Bottom-up corrected (n, mean, var): leaves read directly, internals pooled.
    corrected: dict[int, tuple[float, float, float]] = {}
    for nid in sorted(rows, reverse=True):
        if rows[nid]["is_leaf"]:
            corrected[nid] = _read_var(rows[nid]["stats"])
        else:
            corrected[nid] = _combine_stats([corrected[cid] for cid in children[nid]])

    n_root = corrected[root_id][0]
    if lambda_value and n_root > 0:
        metrics = {**metrics, "tree_age": round(n_root / lambda_value)}
    scale = (n_samples / n_root) if (n_samples is not None and n_root > 0) else 1.0

    # DFS preorder for the frontend's buildHierarchy.
    dfs_order: list[int] = []
    stack = [root_id]
    while stack:
        nid = stack.pop()
        dfs_order.append(nid)
        for cid in reversed(children[nid]):
            stack.append(cid)

    nodes = [
        Node(
            id=int(nid),
            depth=int(rows[nid]["depth"]),
            n_node_samples=round(corrected[nid][0] * scale),
            impurity=math.sqrt(corrected[nid][2]),
            value=[corrected[nid][1]],
        )
        for nid in dfs_order
    ]

    return Tree(
        nodes=nodes,
        height=htree.height or 0,
        n_nodes=htree.n_nodes,
        metrics=metrics,
    )


# ── Multi-output tree builders ───────────────────────────────────────────────

def _tree_from_htree_ml(
    htree,
    code_slots: dict,
    group_spans: list,
    value_len: int,
    n_samples: int | None = None,
    lambda_value: float | None = None,
) -> Tree:
    """Multi-group variant of _tree_from_htree.

    Node stats are keyed by MultiClassEncoder combo codes; each code credits its
    count to one value slot per label group (code_slots). Emits the concatenated
    per-group histograms and a per-group Gini list. Same scaffolding as
    _tree_from_htree: bottom-up leaf correction, n_samples rescale, DFS preorder.
    """
    metrics: dict[str, float] = {}
    if htree._root is None:
        return Tree(nodes=[], height=0, n_nodes=0, metrics=metrics)

    def emit(stats: dict, scale: float) -> tuple[list, list]:
        value = [0.0] * value_len
        for code, cnt in stats.items():
            for slot in code_slots.get(code, ()):
                value[slot] += cnt * scale
        imp = [_gini_from_values(value[off:off + size]) for off, size in group_spans]
        return value, imp

    df = htree.to_dataframe()

    if df is None:
        # Single root leaf — to_dataframe() only traverses branch edges.
        stats = htree._root.stats or {}
        raw_root = sum(stats.values())
        if lambda_value and raw_root > 0:
            metrics["tree_age"] = round(raw_root / lambda_value)
        scale = (n_samples / raw_root) if (n_samples is not None and raw_root > 0) else 1.0
        value, imp = emit(stats, scale)
        node = Node(id=0, depth=0, n_node_samples=round(raw_root * scale), impurity=imp, value=value)
        return Tree(nodes=[node], height=0, n_nodes=1, metrics=metrics)

    rows = {nid: row for nid, row in df.iterrows()}

    children: dict[int, list[int]] = {nid: [] for nid in rows}
    root_id: int = next(iter(rows))
    for nid, row in rows.items():
        pid = row["parent"]
        try:
            children[int(pid)].append(nid)
        except (ValueError, TypeError):
            root_id = nid

    # Bottom-up correction over combo-code histograms (leaf stats keep
    # accumulating while branch stats freeze at split time).
    corrected: dict[int, dict] = {}
    for nid in sorted(rows, reverse=True):
        if rows[nid]["is_leaf"]:
            corrected[nid] = dict(rows[nid]["stats"])
        else:
            merged: dict = {}
            for cid in children[nid]:
                for code, cnt in corrected[cid].items():
                    merged[code] = merged.get(code, 0.0) + cnt
            corrected[nid] = merged

    raw_root = sum(corrected[root_id].values())
    if lambda_value and raw_root > 0:
        metrics["tree_age"] = round(raw_root / lambda_value)
    scale = (n_samples / raw_root) if (n_samples is not None and raw_root > 0) else 1.0

    dfs_order: list[int] = []
    stack = [root_id]
    while stack:
        nid = stack.pop()
        dfs_order.append(nid)
        for cid in reversed(children[nid]):
            stack.append(cid)

    nodes: list[Node] = []
    for nid in dfs_order:
        value, imp = emit(corrected[nid], scale)
        nodes.append(Node(
            id=int(nid),
            depth=int(rows[nid]["depth"]),
            n_node_samples=round(sum(corrected[nid].values()) * scale),
            impurity=imp,
            value=value,
        ))

    return Tree(nodes=nodes, height=htree.height or 0, n_nodes=htree.n_nodes, metrics=metrics)


def _tree_from_isoup_generic(
    htree,
    targets: list,
    n_samples: int | None = None,
    emit=None,
    lambda_value: float | None = None,
    metrics: dict | None = None,
) -> Tree:
    """Shared iSOUP tree builder: bottom-up per-target (n, mean, var) pooling,
    n_samples rescale, DFS preorder. emit(parts, scale) -> (value, impurity)
    turns a node's pooled stats (aligned with `targets`) into the Node fields —
    regression and mixed emitters differ only there. lambda_value, when given,
    adds tree_age (estimated real samples under Poisson bagging) to metrics.
    """
    metrics = dict(metrics or {})
    if htree._root is None:
        return Tree(nodes=[], height=0, n_nodes=0, metrics=metrics)

    def read(stats) -> list[tuple[float, float, float]]:
        out = []
        for t in targets:
            try:
                out.append(_read_var(stats[t]))
            except (KeyError, TypeError, AttributeError):
                out.append((0.0, 0.0, 0.0))
        return out

    def node_from(nid: int, depth: int, parts: list, scale: float) -> Node:
        value, imp = emit(parts, scale)
        return Node(
            id=nid, depth=depth,
            n_node_samples=round(max((p[0] for p in parts), default=0.0) * scale),
            impurity=imp, value=value,
        )

    df = htree.to_dataframe()

    if df is None:
        parts = read(htree._root.stats or {})
        n_raw = max((p[0] for p in parts), default=0.0)
        if lambda_value and n_raw > 0:
            metrics["tree_age"] = round(n_raw / lambda_value)
        scale = (n_samples / n_raw) if (n_samples is not None and n_raw > 0) else 1.0
        return Tree(nodes=[node_from(0, 0, parts, scale)], height=0, n_nodes=1, metrics=metrics)

    rows = {nid: row for nid, row in df.iterrows()}

    children: dict[int, list[int]] = {nid: [] for nid in rows}
    root_id: int = next(iter(rows))
    for nid, row in rows.items():
        pid = row["parent"]
        try:
            children[int(pid)].append(nid)
        except (ValueError, TypeError):
            root_id = nid

    # Bottom-up pooled (n, mean, var) per target.
    corrected: dict[int, list[tuple[float, float, float]]] = {}
    for nid in sorted(rows, reverse=True):
        if rows[nid]["is_leaf"]:
            corrected[nid] = read(rows[nid]["stats"])
        else:
            corrected[nid] = [
                _combine_stats([corrected[cid][k] for cid in children[nid]])
                for k in range(len(targets))
            ]

    n_root = max((p[0] for p in corrected[root_id]), default=0.0)
    if lambda_value and n_root > 0:
        metrics["tree_age"] = round(n_root / lambda_value)
    scale = (n_samples / n_root) if (n_samples is not None and n_root > 0) else 1.0

    dfs_order: list[int] = []
    stack = [root_id]
    while stack:
        nid = stack.pop()
        dfs_order.append(nid)
        for cid in reversed(children[nid]):
            stack.append(cid)

    nodes = [
        node_from(int(nid), int(rows[nid]["depth"]), corrected[nid], scale)
        for nid in dfs_order
    ]

    return Tree(nodes=nodes, height=htree.height or 0, n_nodes=htree.n_nodes, metrics=metrics)


def _tree_from_isoup(
    htree,
    targets: list,
    n_samples: int | None = None,
) -> Tree:
    """Multi-target regression Tree from an iSOUPTreeRegressor: the generic
    builder with value=[mean per target], impurity=[std per target]."""

    def emit(parts: list, scale: float) -> tuple[list, list]:
        return [p[1] for p in parts], [math.sqrt(p[2]) for p in parts]

    return _tree_from_isoup_generic(htree, targets, n_samples=n_samples, emit=emit)
