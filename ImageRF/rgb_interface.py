"""
TreeFlow interface for the RGB image forest.

Almost everything is reused verbatim from PolSAR/polsar_interface.py, because
that module never touches the patch content — it only walks rf.trees, reads
node.depth / is_leaf / left / right / class_counts / n_samples, and calls
tree.predict_proba / node.test.evaluate. None of that is PolSAR-specific.

The one exception is patch_to_rgb, which raises unless the patch is a 4-D
covariance stack. Here it becomes the identity: the patch already IS a picture,
which is the entire reason ImageRF exists — the reference thumbnails in the
frontend's NodeView are readable aerial scenes rather than false-colour noise,
and MeanImage.tsx's per-node averaging becomes genuinely informative.
"""

from __future__ import annotations

import colorsys
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "PolSAR"))

from polsar_interface import (  # noqa: E402
    convert_polsar_rf,
    evaluate_polsar_rf,
    make_polsar_router,
    route_patch,
)

from rgb_patch import as_prepared, to_uint8_rgb  # noqa: E402

# Domain-native aliases, so ImageRF code never has to say "polsar".
# Both are fully generic; see the module docstring.
convert_image_rf = convert_polsar_rf
evaluate_image_rf = evaluate_polsar_rf


def patch_to_rgb(patch) -> np.ndarray:
    """Displayable uint8 RGB rendition — the default `image=` for add_reference.

    The ImageRF replacement for polsar_interface.patch_to_rgb. For this domain
    it is the identity; it only normalizes shape/dtype so that TreeFlow's PNG
    encoder always receives (h, w, 3) uint8.
    """
    return to_uint8_rgb(patch)


def make_image_router(rf, n_trees=None):
    """Router for treeflow.set_reference_router, closing over the live forest.

    Wraps make_polsar_router so pooled reference samples are prepared before
    routing. This is a separate guard from the one in ImageOnlineDecisionTree
    because polsar_interface._walk_patch calls node.test.evaluate(patch)
    DIRECTLY — it bypasses tree.predict_proba entirely. Preparing here costs
    one table build per reference per push instead of one per node visited.

    If the pooled samples are already PreparedPatch objects (what
    example_rgb_treeflow.py stores), this is a free no-op.
    """
    base = make_polsar_router(rf, n_trees=n_trees)

    def router(samples: list) -> list:
        return base([as_prepared(s) for s in samples])

    return router


# ── Display helpers ──────────────────────────────────────────────────────────

# AID/UCM class names are run together ("denseresidential"); these are the
# tokens needed to split them back apart for display.
# Only ATOMIC tokens belong here: a compound like "denseresidential" would
# match itself before decomposing, defeating the whole point.
_WORDS = [
    "baseball", "storage", "railway", "station", "medium", "sparse", "dense",
    "residential", "commercial", "industrial", "agricultural", "buildings",
    "field", "tanks", "court", "parking", "park", "harbor", "runway", "river",
    "forest", "beach", "bridge", "airport", "center", "church", "desert",
    "farmland", "meadow", "mountain", "playground", "pond", "port", "resort",
    "school", "square", "stadium", "viaduct", "bare", "land", "overpass",
    "chaparral", "freeway", "golf", "course", "mobile", "home", "tennis",
    "intersection", "crop", "circular",
]
_WORDS = sorted(set(_WORDS), key=len, reverse=True)


def _prettify(name: str) -> str:
    """'denseresidential' -> 'Dense Residential'; unknown names are just cased."""
    rest, parts = name.lower(), []
    while rest:
        for w in _WORDS:
            if rest.startswith(w):
                parts.append(w)
                rest = rest[len(w):]
                break
        else:
            parts.append(rest)
            break
    return " ".join(p.capitalize() for p in parts if p)


def scene_label_map(class_names: list) -> dict:
    """{0: 'Airport', 1: 'Bare Land', ...} for treeflow.push(label_map=...).

    The forest is converted with the raw class names, so this only prettifies
    them for display — legend, node widget, similarity axes and references tab
    all pick it up.
    """
    return {i: _prettify(n) for i, n in enumerate(class_names)}


def scene_class_colors(n: int) -> list:
    """n visually distinct hex colours for Forest.class_colors.

    AID has 30 classes, so the frontend's 12-colour curated palette wraps into
    near-duplicates; a golden-angle walk through HSV keeps neighbours distinct.
    """
    out = []
    for i in range(n):
        h = (i * 0.61803398875) % 1.0
        s = 0.55 + 0.20 * (i % 3) / 2.0
        v = 0.95 - 0.25 * (i % 2)
        r, g, b = colorsys.hsv_to_rgb(h, s, v)
        out.append("#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255)))
    return out
