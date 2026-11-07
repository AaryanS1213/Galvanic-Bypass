"""Celsius <-> kelvin conversion at the boundaries (config, CSV, UI). Inside gb/ everything is SI."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

# Definitional offset between the Celsius and kelvin scales (exact, not a physical parameter).
ZERO_C_IN_K: float = 273.15


def c_to_k(temp_c: ArrayLike) -> NDArray[np.float64] | float:
    """Celsius to kelvin. T[K] = T[degC] + 273.15."""
    out = np.asarray(temp_c, dtype=np.float64) + ZERO_C_IN_K
    return float(out) if out.ndim == 0 else out


def k_to_c(temp_k: ArrayLike) -> NDArray[np.float64] | float:
    """Kelvin to Celsius. T[degC] = T[K] - 273.15."""
    out = np.asarray(temp_k, dtype=np.float64) - ZERO_C_IN_K
    return float(out) if out.ndim == 0 else out
