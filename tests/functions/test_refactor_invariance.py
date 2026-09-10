"""Pins compute_bz's output so the low-memory refactor is provably a no-op.

The golden arrays are minted once from the pre-refactor implementation and
committed. Any change to the numerics shows up here as a failure, which is what
lets the solver be rewritten for memory without silently moving the physics.

Regenerate (only against the pre-refactor code):
    python -m tests.fixtures.make_golden
"""
import numpy as np
import pytest
from pathlib import Path

from functions.compute_fieldmap import compute_bz, compute_bz_reference
from tests.phantoms import make_phantom

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"
MODES = ["b0SimISMRM", "edge", "constant"]
BUFFER = 7
RESOLUTION = np.array([1.0, 1.0, 1.0])


def test_golden_fixtures_exist():
    missing = [m for m in MODES if not (FIXTURES / f"golden_bz_{m.lower()}.npy").exists()]
    assert not missing, (
        f"missing golden fixtures for {missing} - mint them against the "
        "pre-refactor implementation with: python -m tests.fixtures.make_golden"
    )


@pytest.mark.parametrize("mode", MODES)
def test_compute_bz_matches_golden(mode):
    golden = np.load(FIXTURES / f"golden_bz_{mode.lower()}.npy")
    result = compute_bz(make_phantom(), RESOLUTION, BUFFER, mode)

    assert result.shape == golden.shape, \
        f"{mode}: shape changed {golden.shape} -> {result.shape}"
    deviation = np.abs(result - golden).max()
    assert deviation < 1e-12, \
        f"{mode}: max deviation {deviation:.3e} ppm - the refactor changed the numerics"


@pytest.mark.parametrize("shape", [(37, 41, 43), (32, 32, 32), (24, 30, 27), (16, 16, 17)])
@pytest.mark.parametrize("mode", MODES)
def test_matches_reference_implementation(shape, mode):
    """Live head-to-head against the pre-refactor solver.

    The golden fixtures pin one phantom at one buffer; this runs both
    implementations over several shapes and every legacy padding mode, so an
    error that only shows up at a particular parity or size cannot hide.
    """
    chi = make_phantom(shape, seed=11)
    fast = compute_bz(chi, RESOLUTION, BUFFER, mode)
    slow = compute_bz_reference(chi, RESOLUTION, BUFFER, mode)

    assert fast.shape == slow.shape == shape
    deviation = np.abs(fast - slow).max()
    assert deviation < 1e-12, \
        f"{mode} {shape}: max deviation {deviation:.3e} ppm"


def test_reference_and_fast_agree_on_buffer_zero():
    """buffer=0 takes a different crop branch in both implementations."""
    chi = make_phantom((20, 22, 24), seed=12)
    fast = compute_bz(chi, RESOLUTION, 0, "edge")
    slow = compute_bz_reference(chi, RESOLUTION, 0, "edge")
    assert np.abs(fast - slow).max() < 1e-12
