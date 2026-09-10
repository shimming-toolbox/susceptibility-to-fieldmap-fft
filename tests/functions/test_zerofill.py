"""Tests for the -r/--zerofill padding mode.

This is the behavioural change the r1 revision needs: pad each axis to at least
R times its length, filling with air, so the FFT convolution sees an isolated
object rather than an infinite lattice of replicated bodies.
"""
import numpy as np
import pytest
from scipy.fft import next_fast_len

from functions.compute_fieldmap import (compute_bz, fast_padded_shape, _pad_into,
                                        zerofill_padded_shape)
from tests.phantoms import make_phantom

RESOLUTION = np.array([1.0, 1.0, 1.0])
AIR = 0.35


def test_fast_padded_shape_never_shrinks_below_requested_ratio():
    """The rounding must only ever ADD padding, or R=2 silently becomes R<2."""
    for n in (37, 100, 547, 663, 1767, 1870):
        for ratio in (1.25, 1.5, 2.0):
            padded = fast_padded_shape(n, ratio)
            assert padded >= int(np.ceil(ratio * n)), \
                f"n={n} ratio={ratio}: padded to {padded}, below {np.ceil(ratio * n)}"


def test_fast_padded_shape_uses_the_real_table_only_on_the_last_axis():
    """rfftn transforms the last axis as real and the rest as complex; the two
    have different efficient-length tables."""
    assert fast_padded_shape(1767, 2.0, real_axis=True) == next_fast_len(3534, real=True)
    assert fast_padded_shape(547, 2.0, real_axis=False) == next_fast_len(1094)


def test_fast_padded_shape_avoids_the_prime_trap():
    """547 is the motivating case: exact 2N gives 1094 = 2 x 547."""
    assert fast_padded_shape(547, 2.0) != 1094
    padded = fast_padded_shape(547, 2.0)
    remaining = padded
    for factor in (2, 3, 5, 7, 11, 13):
        while remaining % factor == 0:
            remaining //= factor
    assert remaining == 1, f"{padded} still has a large prime factor"


def test_compute_bz_uses_the_real_table_on_the_last_axis():
    """Guards the CALL SITE, not just fast_padded_shape: compute_bz must ask for
    real_axis on the last axis and complex lengths on the others. A mutation that
    passed real_axis=False everywhere survived the unit test on fast_padded_shape
    alone, which is why this exists."""
    shape = (547, 415, 1767)
    padded = zerofill_padded_shape(shape, 2.0)
    assert padded[0] == next_fast_len(1094)
    assert padded[1] == next_fast_len(830)
    assert padded[2] == next_fast_len(3534, real=True)
    # the real table is the stricter one, so the last axis differs from what the
    # complex table would have chosen
    assert padded[2] != next_fast_len(3534)


def test_real_fast_lengths_can_be_odd_so_s_is_load_bearing():
    """Real-FFT fast lengths are NOT always even - 25, 27, 45, 75, 81 are all
    valid 3-5-smooth odd lengths. Two of the seven validated subjects
    (unfErssm011, unfPain002) have a canvas z of 1811, which pads to 3645 at
    R=2. Without s= on irfftn those volumes come back one voxel short, so this
    documents that the guard is load-bearing in production, not defensive."""
    assert zerofill_padded_shape((8, 8, 1811), 2.0)[2] == 3645
    assert zerofill_padded_shape((8, 8, 1811), 2.0)[2] % 2 == 1
    assert zerofill_padded_shape((8, 8, 37), 2.0)[2] == 75


def test_pad_into_matches_np_pad():
    chi = make_phantom((11, 13, 15), seed=4)
    before, after = (2, 3, 4), (5, 1, 2)
    got = _pad_into(chi, before, after, AIR)
    want = np.pad(chi, tuple(zip(before, after)), mode='constant', constant_values=AIR)
    assert got.shape == want.shape
    assert np.array_equal(got, want)


@pytest.mark.parametrize("shape", [(16, 16, 17), (21, 20, 19), (32, 33, 34),
                                   (37, 41, 43), (16, 16, 37)])  # 37 pads to 75, ODD
def test_zerofill_returns_the_input_shape(shape):
    """Asymmetric padding plus fast lengths must still crop back exactly.
    Odd axes are the ones that break if irfftn is not given s=."""
    out = compute_bz(make_phantom(shape, seed=6), RESOLUTION,
                     zerofill=2.0, pad_value=AIR)
    assert out.shape == shape, f"{shape} came back as {out.shape}"


def test_zerofill_pads_with_air():
    """A uniform-air volume must give a uniform field of chi/3 everywhere.
    If the padding used any other value the result would not be uniform."""
    chi = np.full((24, 24, 24), AIR)
    out = compute_bz(chi, RESOLUTION, zerofill=2.0, pad_value=AIR)
    assert np.allclose(out, AIR / 3.0, atol=1e-9)


def test_zerofill_does_not_replicate_tissue_at_the_wrap():
    """The whole point of the change. A compact object surrounded by air must
    give a field close to zero far from it; edge padding instead replicates the
    tissue that runs off the face and biases the result."""
    chi = np.full((48, 48, 48), AIR)
    chi[20:28, 20:28, 20:28] = -9.05          # compact object, well clear of the faces
    air_far = compute_bz(chi, RESOLUTION, zerofill=2.0, pad_value=AIR)[0, 0, 0]
    assert np.isclose(air_far, AIR / 3.0, atol=0.05), \
        f"far-field corner reads {air_far:.4f} ppm, expected ~{AIR / 3.0:.4f}"


def test_zerofill_and_buffer_are_mutually_exclusive():
    chi = make_phantom((16, 16, 16), seed=7)
    with pytest.raises(ValueError, match="mutually exclusive"):
        compute_bz(chi, RESOLUTION, buffer=5, zerofill=2.0)


def test_zerofill_below_one_is_rejected():
    chi = make_phantom((16, 16, 16), seed=7)
    with pytest.raises(ValueError, match="zerofill must be"):
        compute_bz(chi, RESOLUTION, zerofill=0.5)


def test_float32_matches_float64_within_tolerance():
    """float32 halves peak memory; this bounds what it costs in accuracy."""
    chi = make_phantom((48, 50, 52), seed=8)
    f64 = compute_bz(chi, RESOLUTION, zerofill=1.5, pad_value=AIR, dtype=np.float64)
    f32 = compute_bz(chi, RESOLUTION, zerofill=1.5, pad_value=AIR, dtype=np.float32)
    deviation = np.abs(f64 - f32).max()
    assert deviation < 1e-4, f"float32 drifted {deviation:.3e} ppm from float64"


def test_default_buffer_is_still_fifty_when_zerofill_absent():
    """Backwards compatibility: buffer now defaults to None, and must resolve to
    the historical 50 so existing callers are unaffected."""
    chi = make_phantom((37, 41, 43), seed=0)
    explicit = compute_bz(chi, RESOLUTION, buffer=50, mode='edge')
    implicit = compute_bz(chi, RESOLUTION, mode='edge')
    assert np.array_equal(explicit, implicit)
