"""Mint the golden arrays. Run ONCE, against the PRE-REFACTOR implementation.

    python -m tests.fixtures.make_golden

Re-running after the refactor would defeat the purpose: the fixtures are the
record of what the solver did before it was rewritten.
"""
import numpy as np
from pathlib import Path

from functions.compute_fieldmap import compute_bz
from tests.phantoms import make_phantom

OUT = Path(__file__).resolve().parent
RESOLUTION = np.array([1.0, 1.0, 1.0])
BUFFER = 7

if __name__ == "__main__":
    for mode in ("b0SimISMRM", "edge", "constant"):
        arr = compute_bz(make_phantom(), RESOLUTION, BUFFER, mode)
        np.save(OUT / f"golden_bz_{mode.lower()}.npy", arr)
        print(f"{mode:12s} shape {arr.shape}  "
              f"range [{arr.min():.6f}, {arr.max():.6f}] ppm  "
              f"mean {arr.mean():.6f}")
