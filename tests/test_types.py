import numpy as np
import pytest

from gb.types import Decision, PackState, build_layout


def test_build_layout_8x16() -> None:
    lay = build_layout(8, 16)
    assert lay.n == 128
    assert lay.module_idx.shape == lay.slot_idx.shape == (128,)
    assert lay.neighbors.shape == (8 * 15, 2)
    i, j = lay.neighbors[:, 0], lay.neighbors[:, 1]
    assert np.all(i < j)
    assert np.all(lay.module_idx[i] == lay.module_idx[j])
    assert np.all(lay.slot_idx[j] == lay.slot_idx[i] + 1)
    np.testing.assert_array_equal(lay.module_idx * 16 + lay.slot_idx, np.arange(128))


def test_build_layout_edge_cases() -> None:
    assert build_layout(3, 1).neighbors.shape == (0, 2)
    with pytest.raises(ValueError):
        build_layout(0, 4)


def test_layout_arrays_read_only() -> None:
    lay = build_layout(2, 4)
    with pytest.raises(ValueError):
        lay.module_idx[0] = 5


def _state(n: int) -> PackState:
    soh0 = np.full(n, 0.98)
    return PackState(
        t_s=0.0,
        soc=np.full(n, 0.5),
        T_K=np.full(n, 298.15),
        q_cal=1.0 - soh0,
        q_cyc=np.zeros(n),
        active=np.ones(n, dtype=bool),
        dwell_s=np.zeros(n),
        out_s=np.zeros(n),
        ah_throughput=np.zeros(n),
        T_room_K=298.15,
    )


def test_packstate_soh() -> None:
    s = _state(4)
    s.q_cyc[:] = 0.01
    np.testing.assert_allclose(s.soh, 0.97)


def test_packstate_copy_is_independent() -> None:
    s = _state(4)
    c = s.copy()
    c.soc[0] = 0.1
    c.active[1] = False
    assert s.soc[0] == 0.5 and s.active[1]
    assert c.T_room_K == s.T_room_K


def test_decision_defaults() -> None:
    d = Decision(active=np.ones(3, dtype=bool), P_W=1000.0)
    assert d.note == "" and d.alert == ""
