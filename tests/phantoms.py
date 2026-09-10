"""Deterministic susceptibility phantoms shared by the invariance tests and the
golden-fixture generator.

Kept out of the test modules themselves so the fixture generator can import it
without a sys.path hack.
"""
import numpy as np


def make_phantom(shape=(37, 41, 43), seed=0):
    """A deterministic chi phantom, in ppm.

    Air background with a water ellipsoid, a lung-like inclusion, and tissue
    running off one face so that edge padding has something to replicate.
    Dimensions are odd on purpose: they exercise the odd/even k-grid handling
    that commit 5abb079 fixed.
    """
    rng = np.random.default_rng(seed)
    chi = np.full(shape, 0.35)

    ii, jj, kk = np.ogrid[:shape[0], :shape[1], :shape[2]]
    ci, cj, ck = (s / 2 for s in shape)
    ell = ((ii - ci) / 12.0) ** 2 + ((jj - cj) / 14.0) ** 2 + ((kk - ck) / 15.0) ** 2

    chi[ell <= 1.0] = -9.05          # body / water
    chi[ell <= 0.25] = -4.2          # lung-like inclusion
    chi[:, :, :3] = -9.05            # tissue touching a face
    chi += rng.normal(0.0, 1e-3, shape)   # break symmetry
    return chi
