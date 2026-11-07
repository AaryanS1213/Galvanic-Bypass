import numpy as np
import pytest

from gb.units import c_to_k, k_to_c


def test_c_k_roundtrip_scalar() -> None:
    assert c_to_k(25.0) == pytest.approx(298.15)
    assert k_to_c(c_to_k(-40.0)) == pytest.approx(-40.0)


def test_c_k_vectorized() -> None:
    out = c_to_k(np.array([0.0, 30.0, 60.0]))
    assert isinstance(out, np.ndarray) and out.dtype == np.float64
    np.testing.assert_allclose(out, [273.15, 303.15, 333.15])
