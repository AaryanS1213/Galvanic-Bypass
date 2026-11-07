"""Tests for gb.ingest."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from gb.ingest import (
    TelemetryWarning,
    column_map_from_config,
    load_csv,
    to_arrays,
    wide_to_long,
)
from gb.units import c_to_k
from tests.conftest import SmallPack

N_CELLS = 8  # small_pack fixture: 2 x 4
T0 = 1.7e9


def make_long(n_cells: int = N_CELLS, n_steps: int = 5, dt_s: float = 60.0) -> pd.DataFrame:
    """Tiny canonical long telemetry with epoch-second timestamps and bypass/room columns."""
    t = T0 + dt_s * np.arange(n_steps)
    tt, cc = np.meshgrid(t, np.arange(n_cells), indexing="ij")
    k = tt.size
    return pd.DataFrame(
        {
            "timestamp": tt.ravel(),
            "cell_id": cc.ravel().astype(np.int64),
            "voltage_V": 2.8 + 0.001 * np.arange(k),
            "current_A": np.full(k, 40.0),
            "temp_C": 25.0 + cc.ravel() + 0.01 * np.arange(k),
            "soc": np.linspace(0.9, 0.5, k),
            "soh": np.full(k, 0.98),
            "resistance_mOhm": np.full(k, 0.5),
            "bypass_state": (cc.ravel() == 3).astype(np.int64),
            "room_temp_C": np.full(k, 24.0),
        }
    )


def test_round_trip_and_shapes(tmp_path: Path, small_pack: SmallPack) -> None:
    df = make_long()
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    for engine in ("pyarrow", "pandas"):
        back = load_csv(p, {}, engine=engine)
        assert list(back.columns) == list(df.columns)
        np.testing.assert_allclose(back["temp_C"], df["temp_C"])
        assert back["cell_id"].dtype == np.int64
        a = to_arrays(back, small_pack.cfg)
        assert a["V"].shape == a["T_K"].shape == a["soc"].shape == a["soh"].shape == (5, N_CELLS)
        assert a["bypass"].shape == (5, N_CELLS) and a["bypass"].dtype == bool
        assert a["V"].dtype == np.float64
        assert a["t_s"].shape == a["I_A"].shape == (5,)
        np.testing.assert_allclose(a["t_s"], 60.0 * np.arange(5))
        assert a["t0_epoch_s"] == pytest.approx(T0)
        np.testing.assert_allclose(a["I_A"], 40.0)
        np.testing.assert_allclose(a["T_K"], c_to_k(df["temp_C"].to_numpy().reshape(5, N_CELLS)))
        np.testing.assert_array_equal(a["active"], ~a["bypass"])
        assert a["bypass"][:, 3].all() and not a["bypass"][:, [0, 1, 2, 4]].any()
        np.testing.assert_allclose(a["R_ohm"], 0.5e-3)
        assert a["T_room_K"] == pytest.approx(c_to_k(24.0))


def test_chunked_reading_matches(tmp_path: Path) -> None:
    df = make_long(n_steps=40)
    p = tmp_path / "t.csv"
    df.to_csv(p, index=False)
    whole = load_csv(p, engine="pandas")
    pd.testing.assert_frame_equal(load_csv(p, engine="pandas", chunk_rows=7), whole)
    pd.testing.assert_frame_equal(load_csv(p, engine="pyarrow", block_bytes=1024), whole)


def test_room_temp_defaults_to_site(small_pack: SmallPack) -> None:
    df = make_long().drop(columns=["room_temp_C", "bypass_state"])
    a = to_arrays(df, small_pack.cfg)
    assert a["T_room_K"] == pytest.approx(c_to_k(small_pack.cfg["site"]["room_C"]))
    assert "bypass" not in a and "active" not in a


def test_percent_soc_rescaled_in_arrays(small_pack: SmallPack) -> None:
    df = make_long()
    df["soc"] *= 100.0
    with pytest.warns(TelemetryWarning, match="percent"):
        a = to_arrays(df, small_pack.cfg)
    assert a["soc"].max() <= 1.0
    assert df["soc"].max() > 1.0  # input not mutated


def test_duplicates_raise(small_pack: SmallPack) -> None:
    df = make_long()
    with pytest.raises(ValueError, match="duplicate"):
        to_arrays(pd.concat([df, df.iloc[[0]]]), small_pack.cfg)


def test_missing_required_column_raises(small_pack: SmallPack) -> None:
    with pytest.raises(ValueError, match="missing required"):
        to_arrays(make_long().drop(columns=["voltage_V"]), small_pack.cfg)


def test_single_gap_forward_filled(small_pack: SmallPack) -> None:
    df = make_long()
    full = to_arrays(df, small_pack.cfg)
    # cell 2 misses step 2 (row dropped); cell 5 misses step 0 (NaN value)
    gappy = df.drop(index=2 * N_CELLS + 2).copy()
    gappy.loc[5, "temp_C"] = np.nan
    with pytest.warns(TelemetryWarning, match="missing rows"):
        a = to_arrays(gappy, small_pack.cfg)
    assert a["T_K"][2, 2] == full["T_K"][1, 2]
    assert a["V"][2, 2] == full["V"][1, 2]
    assert a["T_K"][0, 5] == full["T_K"][1, 5]
    assert not np.isnan(a["T_K"]).any()


@pytest.mark.filterwarnings("ignore::gb.ingest.TelemetryWarning")
def test_longer_gap_raises(small_pack: SmallPack) -> None:
    df = make_long()
    gappy = df.drop(index=[1 * N_CELLS + 2, 2 * N_CELLS + 2])
    with pytest.raises(ValueError, match="2\\+ consecutive"):
        to_arrays(gappy, small_pack.cfg)


@pytest.mark.filterwarnings("ignore::gb.ingest.TelemetryWarning")
def test_absent_cell_raises(small_pack: SmallPack) -> None:
    df = make_long()
    with pytest.raises(ValueError, match="consecutive"):
        to_arrays(df[df["cell_id"] != 7], small_pack.cfg)


def test_cell_id_out_of_pack_raises(small_pack: SmallPack) -> None:
    with pytest.raises(ValueError, match="cell_id"):
        to_arrays(make_long(n_cells=9), small_pack.cfg)


def test_current_mismatch(small_pack: SmallPack) -> None:
    df = make_long()
    df.loc[df["cell_id"] == 0, "current_A"] = 50.0
    with pytest.warns(TelemetryWarning, match="disagrees"):
        a = to_arrays(df, small_pack.cfg)
    np.testing.assert_allclose(a["I_A"], 40.0)  # median
    with pytest.raises(ValueError, match="disagrees"):
        to_arrays(df, small_pack.cfg, current_mismatch="raise")
    # bypassed cells reading 0 A are excluded from the median and the check
    df2 = make_long()
    df2.loc[df2["bypass_state"] == 1, "current_A"] = 0.0
    a2 = to_arrays(df2, small_pack.cfg, current_mismatch="raise")
    np.testing.assert_allclose(a2["I_A"], 40.0)


def test_column_mapping_renamed(tmp_path: Path, small_pack: SmallPack) -> None:
    df = make_long()
    cmap = {"timestamp": "Time", "cell_id": "CellNo", "temp_C": "T_cell", "voltage_V": "Vcell"}
    src = df.rename(columns=cmap)  # canonical -> REON-style source names
    src["extra_junk"] = 1.0
    p = tmp_path / "reon.csv"
    src.to_csv(p, index=False)
    back = load_csv(p, cmap)
    assert "extra_junk" not in back.columns
    assert list(back.columns) == list(df.columns)
    np.testing.assert_allclose(back["temp_C"], df["temp_C"])
    a = to_arrays(back, small_pack.cfg)
    assert a["T_K"].shape == (5, N_CELLS)


def test_column_map_from_config_ignores_unknown(cfg: dict) -> None:
    c = copy.deepcopy(cfg)
    c["column_map"]["columns"] = {"temp_C": "T", "temperature": "temperature", "bogus": "x"}
    assert column_map_from_config(c) == {"temp_C": "T"}
    assert column_map_from_config({}) == {}
    # works on whatever the shipped YAML holds
    assert set(column_map_from_config(cfg)) <= set(make_long().columns)


def test_epoch_vs_iso_timestamps(tmp_path: Path, small_pack: SmallPack) -> None:
    df = make_long()
    iso = df.copy()
    iso["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True).dt.strftime("%Y-%m-%dT%H:%M:%S%z")
    p_e, p_i = tmp_path / "e.csv", tmp_path / "i.csv"
    df.to_csv(p_e, index=False)
    iso.to_csv(p_i, index=False)
    a_e = to_arrays(load_csv(p_e), small_pack.cfg)
    a_i = to_arrays(load_csv(p_i), small_pack.cfg)
    assert a_e["t0_epoch_s"] == pytest.approx(a_i["t0_epoch_s"])
    np.testing.assert_allclose(a_e["t_s"], a_i["t_s"])
    np.testing.assert_allclose(a_e["T_K"], a_i["T_K"])


def test_wide_to_long(tmp_path: Path, small_pack: SmallPack) -> None:
    long = make_long().drop(columns=["resistance_mOhm"])
    n_steps = 5
    wide = pd.DataFrame({"timestamp": T0 + 60.0 * np.arange(n_steps), "current_A": 40.0, "room_temp_C": 24.0})
    for field in ("voltage_V", "temp_C", "soc", "soh", "bypass_state"):
        vals = long[field].to_numpy().reshape(n_steps, N_CELLS)
        for i in range(N_CELLS):
            wide[f"{field}_{i:03d}"] = vals[:, i]
    out = wide_to_long(wide)
    assert len(out) == n_steps * N_CELLS
    assert set(out.columns) == set(long.columns)
    a_long = to_arrays(long, small_pack.cfg)
    a_wide = to_arrays(out, small_pack.cfg)
    for k in ("V", "T_K", "soc", "soh", "bypass", "t_s", "I_A"):
        np.testing.assert_array_equal(a_long[k], a_wide[k])

    # through load_csv with a renamed per-cell prefix and timestamp
    src = wide.rename(columns=lambda c: c.replace("temp_C_", "T_")).rename(columns={"timestamp": "time"})
    p = tmp_path / "wide.csv"
    src.to_csv(p, index=False)
    back = load_csv(p, {"temp_C": "T", "timestamp": "time"}, wide=True)
    a_csv = to_arrays(back, small_pack.cfg)
    np.testing.assert_allclose(a_csv["T_K"], a_long["T_K"])
    assert a_csv["T_room_K"] == pytest.approx(c_to_k(24.0))


def test_epoch_milliseconds(tmp_path: Path, small_pack: SmallPack) -> None:
    df = make_long()
    ms = df.copy()
    ms["timestamp"] = (df["timestamp"] * 1000.0).astype(np.int64)
    p_s, p_ms = tmp_path / "s.csv", tmp_path / "ms.csv"
    df.to_csv(p_s, index=False)
    ms.to_csv(p_ms, index=False)
    with pytest.warns(TelemetryWarning, match="milliseconds"):
        back = load_csv(p_ms, cfg=small_pack.cfg)
    a_ms = to_arrays(back, small_pack.cfg)
    a_s = to_arrays(load_csv(p_s, cfg=small_pack.cfg), small_pack.cfg)
    assert a_ms["t0_epoch_s"] == pytest.approx(T0)
    np.testing.assert_allclose(a_ms["t_s"], a_s["t_s"])
    np.testing.assert_allclose(a_ms["t_s"], 60.0 * np.arange(5))


def test_shipped_column_map_is_canonical_identity(cfg: dict) -> None:
    from gb.schema import CANONICAL_COLUMNS

    cm = column_map_from_config(cfg)
    assert cm == {c: c for c in CANONICAL_COLUMNS}
    assert set(cfg["column_map"]["columns"]) == set(CANONICAL_COLUMNS)


def test_tolerances_come_from_config(small_pack: SmallPack) -> None:
    df = make_long()
    df.loc[df["cell_id"] == 0, "current_A"] = 41.0  # 1 A off the 40 A median
    with pytest.warns(TelemetryWarning, match="disagrees"):
        to_arrays(df, small_pack.cfg)
    loose = copy.deepcopy(small_pack.cfg)
    loose["ingest"]["current_agree_abs_A"] = 2.0
    to_arrays(df, loose, current_mismatch="raise")  # within the looser tolerance: no error
