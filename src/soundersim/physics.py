"""Scalar wave-physics helpers (relative-power conventions, no absolute calibration)."""

import numpy as np


def fresnel_normal(eps1, eps2):
    """Normal-incidence scalar Fresnel reflection coefficient from medium 1 into
    medium 2, from relative permittivities. Sign preserved (air->ice ~ -0.2807).
    """
    n1, n2 = np.sqrt(eps1), np.sqrt(eps2)
    return (n1 - n2) / (n1 + n2)
