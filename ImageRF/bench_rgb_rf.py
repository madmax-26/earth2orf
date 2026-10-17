"""
Performance benchmark for ImageRF (run manually):

    python bench_rgb_rf.py

Prints the numbers the design rests on, so they stay honest as the code
evolves: projection cost per test family, the summed-area-table storage
comparison that justifies array.array, PreparedPatch build cost and memory (the
batch-size budget), resize backend cost, an extrapolated epoch time, and the
raw-vs-prepared ratio that guards against the prepare-on-the-fly trap.
"""

import array
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rgb_patch import PreparedPatch, integral_image
from rgb_node_tests import RectTest, RectTestSampler


def timed(fn, n: int) -> float:
    """Microseconds per call."""
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n * 1e6


def bench_projections(rng, S=64):
    print("Projection cost (us per call), by family:")
    arr = rng.integers(0, 256, size=(S, S, 3), dtype=np.uint8)
    p = PreparedPatch(arr)
    for proj in ("unary", "diff", "sum", "absdiff"):
        for op in ("mean", "center"):
            n_rects = 1 if proj == "unary" else 2
            t = RectTest(projection=proj,
                         rects=[(4, 4, 12, 20)] * n_rects,
                         operators=[op] * n_rects,
                         channels=[1] * n_rects)
            us = timed(lambda t=t: t.project(p), 30000)
            print(f"  {proj:8} {op:7} {us:6.2f}")


def bench_sat_storage(rng, S=64):
    print("\nSAT storage comparison (us per 2-rect projection):")
    arr = rng.integers(0, 256, size=(S, S, 3), dtype=np.uint8)
    table = integral_image(arr)
    stride = S + 1
    r, c, rh, rw = 4, 4, 12, 20
    top, bot = r * stride + c, (r + rh) * stride + c

    flat_np = np.ascontiguousarray(table[:, :, 1]).ravel()
    flat_arr = array.array("i", flat_np.tolist())
    flat_list = flat_np.tolist()
    cube = table

    variants = {
        "array.array('i')": lambda f=flat_arr: (f[bot + rw] - f[top + rw] - f[bot] + f[top]) / (rh * rw),
        "python list":      lambda f=flat_list: (f[bot + rw] - f[top + rw] - f[bot] + f[top]) / (rh * rw),
        "flat ndarray":     lambda f=flat_np: (f[bot + rw] - f[top + rw] - f[bot] + f[top]) / (rh * rw),
        "3-D ndarray":      lambda f=cube: (f[r + rh, c + rw, 1] - f[r, c + rw, 1]
                                            - f[r + rh, c, 1] + f[r, c, 1]) / (rh * rw),
    }
    base = None
    for name, fn in variants.items():
        us = timed(fn, 50000)
        base = base or us
        print(f"  {name:18} {us:6.3f}   ({us / base:.2f}x)")

    # Memory decides this, not speed: a python list projects ~1.6x FASTER than
    # array.array, but a list of ints costs the pointer array PLUS a ~28-byte
    # int object per distinct value (only -5..256 are interned). sys.getsizeof
    # on the list alone counts just the pointers and understates it ~4x.
    arr_kb = flat_arr.buffer_info()[1] * 4 * 3 / 1024
    list_kb = (sys.getsizeof(flat_list) + sum(sys.getsizeof(v) for v in set(flat_list))) * 3 / 1024
    print(f"  memory (3 channels): array.array {arr_kb:.0f} KB | python list {list_kb:.0f} KB"
          f"  ({list_kb / arr_kb:.1f}x)")
    for bs in (2000,):
        print(f"    at batch_size={bs}: {arr_kb * bs / 1024:.0f} MB vs {list_kb * bs / 1024:.0f} MB"
              " -> array.array wins on the only axis that matters here")


def bench_prepare(rng, S=64):
    print("\nPreparedPatch construction:")
    arr = rng.integers(0, 256, size=(S, S, 3), dtype=np.uint8)
    us = timed(lambda: PreparedPatch(arr), 2000)
    p = PreparedPatch(arr)
    kb = (p.arr.nbytes + sum(a.buffer_info()[1] * 4 for a in p._sat) + len(p._px)) / 1024
    print(f"  {S}x{S}: {us:.0f} us, ~{kb:.0f} KB per patch")
    for bs in (1000, 2000, 5000):
        print(f"    batch_size={bs:5} -> ~{bs * kb / 1024:.0f} MB resident")


def bench_resize(rng):
    print("\nResize backend (us per 190x190 -> 64x64 crop):")
    from rgb_datasets import _resize
    crop = rng.integers(0, 256, size=(190, 190, 3), dtype=np.uint8)
    for backend in ("pil", "nearest", "skimage"):
        try:
            us = timed(lambda b=backend: _resize(crop, 64, b), 100 if backend != "skimage" else 20)
            print(f"  {backend:9} {us:8.0f}")
        except ImportError:
            print(f"  {backend:9} (unavailable)")


def bench_accumulate(rng, S=64, n_candidates=100):
    print(f"\nAccumulate loop ({n_candidates} candidates, as the forest runs it):")
    arr = rng.integers(0, 256, size=(S, S, 3), dtype=np.uint8)
    p = PreparedPatch(arr)
    sampler = RectTestSampler(patch_size=S)
    cands = [sampler.sample(rng) for _ in range(n_candidates)]
    values = [[] for _ in cands]

    def one_sample():
        for i in range(len(cands)):
            try:
                values[i].append(cands[i].project(p))
            except Exception:
                pass

    us = timed(one_sample, 300)
    print(f"  {us:.1f} us/sample  ({us / n_candidates:.2f} us per projection)")
    for n_samples, n_trees in ((20000, 30), (8000, 8)):
        secs = us * 1e-6 * n_samples * n_trees
        print(f"    {n_samples} samples x {n_trees} trees -> ~{secs:.0f}s of accumulation")


def bench_raw_vs_prepared(rng, S=64):
    print("\nPrepare-on-the-fly penalty (guards the perf trap):")
    arr = rng.integers(0, 256, size=(S, S, 3), dtype=np.uint8)
    p = PreparedPatch(arr)
    t = RectTest(projection="diff", rects=[(4, 4, 12, 20), (30, 8, 6, 6)],
                 operators=["mean", "mean"], channels=[1, 1])
    prep = timed(lambda: t.project(p), 20000)
    raw = timed(lambda: t.project(arr), 2000)
    print(f"  prepared {prep:6.2f} us | raw ndarray {raw:8.1f} us  ({raw / prep:.0f}x)")
    print("  -> hot paths must hand over PreparedPatch objects (rgb_forest / rgb_interface do)")


def main():
    rng = np.random.default_rng(0)
    bench_projections(rng)
    bench_sat_storage(rng)
    bench_prepare(rng)
    bench_resize(rng)
    bench_accumulate(rng)
    bench_raw_vs_prepared(rng)


if __name__ == "__main__":
    main()
