"""Tests for gb.schema."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from gb.schema import (
    CANONICAL_COLUMNS,
    REQUIRED_COLUMNS,
    ColumnSpec,
    parse_timestamps,
    to_epoch_seconds,
    validate,
)


def make_long(n_cells: int = 4, n_steps: int = 5, dt_s: float = 60.0, t0: float = 1.7e9) -> pd.DataFrame:
    """Tiny canonical long-format telemetry, epoch-second timestamps."""
    t = t0 + dt_s * np.arange(n_steps)
    tt, cc = np.meshgrid(t, np.arange(n_cells), indexing="ij")
    k = tt.size
    rng = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "timestamp": tt.ravel(),
            "cell_id": cc.ravel().astype(np.int64),
            "voltage_V": 2.8 + 0.01 * rng.standard_normal(k),
            "current_A": np.full(k, 40.0),
            "temp_C": 25.0 + cc.ravel() + 0.1 * np.arange(k),
            "soc": np.linspace(0.9, 0.5, k),
            "soh": np.full(k, 0.98),
        }
    )


def test_schema_constants() -> None:
    assert set(REQUIRED_COLUMNS) == {"timestamp", "cell_id", "voltage_V", "current_A", "temp_C", "soc", "soh"}
    assert {"resistance_mOhm", "bypass_state", "room_temp_C"} <= set(CANONICAL_COLUMNS)
    spec = ColumnSpec(name="x", required=False, unit="V", lo=0, hi=1)
    assert spec.hi == 1


def test_clean_data_ok() -> None:
    rep = validate(make_long(), n_cells=4)
    assert rep.ok, rep.errors
    assert rep.warnings == []
    assert rep.n_duplicate_rows == 0
    assert rep.nominal_dt_s == pytest.approx(60.0)


def test_missing_required_column() -> None:
    rep = validate(make_long().drop(columns=["soh"]))
    assert not rep.ok
    assert rep.missing_required == ["soh"]


def test_percent_soc_detected_and_rescaled_without_mutating() -> None:
    df = make_long()
    df["soc"] = df["soc"] * 100.0
    original = df["soc"].copy()
    rep = validate(df)
    assert rep.ok, rep.errors
    assert "soc" in rep.rescaled_columns
    assert any("percent" in w for w in rep.warnings)
    np.testing.assert_allclose(rep.df["soc"].to_numpy(), original.to_numpy() / 100.0)
    pd.testing.assert_series_equal(df["soc"], original)  # caller's df untouched


def test_fraction_slightly_above_one_is_error_not_rescaled() -> None:
    df = make_long()
    df.loc[0, "soh"] = 1.2
    rep = validate(df)
    assert not rep.ok
    assert rep.out_of_range == {"soh": 1}
    assert rep.rescaled_columns == []


@pytest.mark.parametrize(("col", "bad"), [("temp_C", 120.0), ("temp_C", -50.0), ("voltage_V", 6.0)])
def test_unit_sanity(col: str, bad: float) -> None:
    df = make_long()
    df.loc[3, col] = bad
    rep = validate(df)
    assert not rep.ok
    assert rep.out_of_range[col] == 1


def test_bypass_must_be_binary() -> None:
    df = make_long()
    df["bypass_state"] = 0
    df.loc[2, "bypass_state"] = 2
    rep = validate(df)
    assert rep.out_of_range == {"bypass_state": 1}


def test_duplicates_detected_across_timestamp_formats() -> None:
    df = make_long(n_cells=2, n_steps=3)
    df["timestamp"] = parse_timestamps(df["timestamp"]).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    dup = df.iloc[[1]].copy()
    rep = validate(pd.concat([df, dup], ignore_index=True))
    assert rep.n_duplicate_rows == 1
    assert not rep.ok


def test_sampling_gap_and_missing_cell_rows() -> None:
    df = make_long(n_cells=3, n_steps=6)
    t = parse_timestamps(df["timestamp"])
    df = df[t != t.unique()[3]]  # whole-pack gap at step 3
    df = df[~((df["cell_id"] == 1) & (df.index == 4))]  # cell 1 misses step 1
    rep = validate(df, n_cells=4)
    assert rep.ok, rep.errors
    assert len(rep.gaps) == 1
    assert rep.gaps[0].duration_s == pytest.approx(120.0)
    assert rep.cells_missing_rows == {1: 1, 3: 5}  # cell 3 never reported


def test_cell_id_out_of_range() -> None:
    rep = validate(make_long(n_cells=4), n_cells=3)
    assert not rep.ok
    assert any("cell_id" in e for e in rep.errors)


def test_parse_epoch_and_iso_agree() -> None:
    epoch = pd.Series([1.7e9, 1.7e9 + 60])
    iso = pd.Series(["2023-11-14T22:13:20Z", "2023-11-14T23:14:20+01:00"])
    naive = pd.Series(["2023-11-14 22:13:20", "2023-11-14 22:14:20"])
    numeric_str = pd.Series(["1700000000", "1700000060"])
    for s in (epoch, iso, naive, numeric_str):
        np.testing.assert_allclose(to_epoch_seconds(parse_timestamps(s)), [1.7e9, 1.7e9 + 60])


def test_unparseable_timestamp_is_error() -> None:
    df = make_long(n_cells=1, n_steps=2)
    df["timestamp"] = ["yesterday", "today"]
    rep = validate(df)
    assert not rep.ok
