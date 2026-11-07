"""Telemetry schema: canonical column names, dtypes and validation for REON per-cell CSV.

Canonical long format, one row per cell per timestamp:

required
    timestamp   ISO 8601 string (naive = UTC) or epoch seconds
    cell_id     int, 0..n-1
    voltage_V   cell voltage [V]
    current_A   string current [A], + = discharge
    temp_C      cell temperature [degC]
    soc         state of charge [1], 0..1 (0..100 auto-detected and rescaled)
    soh         state of health [1], 0..1 (0..100 auto-detected and rescaled)
optional
    resistance_mOhm  cell DC resistance [mOhm]
    bypass_state     0/1, 1 = bypassed (cell out of the string)
    room_temp_C      room air temperature [degC]

`validate(df)` never mutates its input. It returns a `ValidationReport` whose `.df` is a
copy of the input with `timestamp` parsed to tz-aware UTC datetimes and soc/soh rescaled
to 0..1 when they were detected as percent.

Data-quality tolerances (percent detection, gap factor, epoch-ms threshold) come from
config/ingest.yaml via the `cfg` argument; cfg=None loads the default config. The numeric
bounds below are sanity bounds for catching unit/format mistakes, not physical
parameters of the cell (those live in config/).
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

# --- Validation bounds (sanity checks on units, not physical parameters) ---------------
TEMP_MIN_C: float = -40.0
TEMP_MAX_C: float = 90.0
VOLTAGE_MIN_V: float = 0.0
VOLTAGE_MAX_V: float = 5.0
PERCENT_SCALE: float = 100.0  # definitional: percent -> fraction
MS_PER_S: float = 1000.0  # definitional
# How many items to list in an error/warning message.
MAX_LISTED: int = 10


class TelemetryWarning(UserWarning):
    """Non-fatal telemetry data-quality issue (gaps, percent rescaling, epoch ms, current mismatch)."""


def ingest_settings(cfg: Mapping[str, Any] | None) -> Mapping[str, Any]:
    """The `ingest` section of cfg (config/ingest.yaml); cfg=None loads the default config."""
    if cfg is None:
        from gb.config import load_config

        cfg = load_config()
    return cfg["ingest"]


class ColumnSpec(BaseModel):
    """Specification of one canonical telemetry column."""

    model_config = ConfigDict(frozen=True)

    name: str
    required: bool
    unit: str = Field(description="unit as written in the column name; '1' = dimensionless")
    description: str = ""
    lo: float | None = Field(default=None, description="inclusive lower sanity bound")
    hi: float | None = Field(default=None, description="inclusive upper sanity bound")
    percent_rescalable: bool = Field(default=False, description="0..100 is auto-detected and rescaled")
    binary: bool = Field(default=False, description="values must be 0 or 1")


SCHEMA: tuple[ColumnSpec, ...] = (
    ColumnSpec(name="timestamp", required=True, unit="s", description="ISO 8601 or epoch seconds"),
    ColumnSpec(name="cell_id", required=True, unit="1", description="cell index 0..n-1", lo=0),
    ColumnSpec(name="voltage_V", required=True, unit="V", lo=VOLTAGE_MIN_V, hi=VOLTAGE_MAX_V),
    ColumnSpec(name="current_A", required=True, unit="A", description="string current, + = discharge"),
    ColumnSpec(name="temp_C", required=True, unit="degC", lo=TEMP_MIN_C, hi=TEMP_MAX_C),
    ColumnSpec(name="soc", required=True, unit="1", lo=0.0, hi=1.0, percent_rescalable=True),
    ColumnSpec(name="soh", required=True, unit="1", lo=0.0, hi=1.0, percent_rescalable=True),
    ColumnSpec(name="resistance_mOhm", required=False, unit="mOhm", lo=0.0),
    ColumnSpec(name="bypass_state", required=False, unit="1", description="1 = bypassed", binary=True),
    ColumnSpec(name="room_temp_C", required=False, unit="degC", lo=TEMP_MIN_C, hi=TEMP_MAX_C),
)

SPECS: dict[str, ColumnSpec] = {s.name: s for s in SCHEMA}
REQUIRED_COLUMNS: tuple[str, ...] = tuple(s.name for s in SCHEMA if s.required)
OPTIONAL_COLUMNS: tuple[str, ...] = tuple(s.name for s in SCHEMA if not s.required)
CANONICAL_COLUMNS: tuple[str, ...] = tuple(s.name for s in SCHEMA)
# Columns holding numeric telemetry values (everything except the two keys).
VALUE_COLUMNS: tuple[str, ...] = tuple(c for c in CANONICAL_COLUMNS if c not in ("timestamp", "cell_id"))

_EPOCH = pd.Timestamp(0, tz="UTC")


def _from_epoch(num: pd.Series, ms_threshold: float) -> pd.Series:
    """Numeric epoch -> UTC datetimes; median > ms_threshold means milliseconds (warns)."""
    vals = num.astype("float64")
    med = float(np.nanmedian(vals.to_numpy())) if len(vals) else 0.0
    if med > ms_threshold:
        warnings.warn(
            f"timestamp: numeric values look like epoch milliseconds (median {med:.4g} > "
            f"{ms_threshold:.4g}); converted to seconds",
            TelemetryWarning,
            stacklevel=3,
        )
        vals = vals / MS_PER_S
    return pd.to_datetime(vals, unit="s", utc=True)


def parse_timestamps(s: pd.Series, cfg: Mapping[str, Any] | None = None) -> pd.Series:
    """Parse a timestamp column to tz-aware UTC datetimes.

    Accepts datetimes (naive = UTC), numbers or numeric strings (epoch seconds, or epoch
    milliseconds when the median exceeds ingest.epoch_ms_threshold, with a
    TelemetryWarning) or ISO 8601 strings (naive = UTC, offsets honoured). Raises
    ValueError if the column cannot be parsed. cfg=None loads the default config.
    """
    if isinstance(s.dtype, pd.DatetimeTZDtype):
        return s.dt.tz_convert("UTC")
    if pd.api.types.is_datetime64_any_dtype(s):
        return s.dt.tz_localize("UTC")
    if pd.api.types.is_bool_dtype(s):
        raise ValueError("timestamp column is boolean")
    ms_threshold = float(ingest_settings(cfg)["epoch_ms_threshold"])
    if pd.api.types.is_numeric_dtype(s):
        return _from_epoch(s, ms_threshold)
    num = pd.to_numeric(s, errors="coerce")
    if len(s) and num.notna().all():
        return _from_epoch(num, ms_threshold)
    try:
        return pd.to_datetime(s, utc=True, format="ISO8601")
    except (ValueError, TypeError) as exc:
        raise ValueError(f"cannot parse timestamp column: {exc}") from exc


def to_epoch_seconds(ts: pd.Series | pd.Index) -> np.ndarray:
    """Tz-aware UTC datetimes -> float64 seconds since 1970-01-01 UTC."""
    return np.asarray((ts - _EPOCH) / pd.Timedelta(seconds=1), dtype=np.float64)


class Gap(BaseModel):
    """A step between consecutive (pack-wide) timestamps longer than ingest.gap_factor x median step."""

    start_epoch_s: float
    end_epoch_s: float
    duration_s: float


class ValidationReport(BaseModel):
    """Result of `validate`. `ok` is True when there are no errors (warnings are allowed).

    `df` is a validated copy of the input (timestamps parsed to UTC, soc/soh rescaled to
    0..1 if they were percent); the caller's DataFrame is never modified.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    missing_required: list[str] = Field(default_factory=list)
    rescaled_columns: list[str] = Field(default_factory=list)
    out_of_range: dict[str, int] = Field(default_factory=dict, description="column -> number of bad values")
    n_duplicate_rows: int = 0
    nominal_dt_s: float | None = None
    gaps: list[Gap] = Field(default_factory=list)
    cells_missing_rows: dict[int, int] = Field(
        default_factory=dict, description="cell_id -> number of timestamps with no row for that cell"
    )
    df: Any = Field(default=None, repr=False, exclude=True)

    @property
    def ok(self) -> bool:
        """True if no errors were found."""
        return not self.errors

    def raise_if_errors(self) -> None:
        """Raise ValueError listing all errors, if any."""
        if self.errors:
            raise ValueError("telemetry validation failed:\n  " + "\n  ".join(self.errors))


def _listed(items: list[Any]) -> str:
    head = ", ".join(str(x) for x in items[:MAX_LISTED])
    return head + (f", ... ({len(items)} total)" if len(items) > MAX_LISTED else "")


def _check_values(out: pd.DataFrame, rep: ValidationReport, percent_min: float) -> None:
    """Coerce value columns to float64, rescale percent columns, check bounds.

    A fraction column with max in (percent_min, 100] and min >= 0 is treated as percent;
    values in (1, percent_min] are reported as out of range rather than silently rescaled.
    """
    for name in VALUE_COLUMNS:
        if name not in out.columns:
            continue
        spec = SPECS[name]
        col = pd.to_numeric(out[name], errors="coerce").astype("float64")
        n_bad_parse = int(col.isna().sum() - out[name].isna().sum())
        if n_bad_parse > 0:
            rep.errors.append(f"{name}: {n_bad_parse} non-numeric values")
        if spec.percent_rescalable and col.notna().any():
            lo, hi = float(col.min()), float(col.max())
            if percent_min < hi <= PERCENT_SCALE and lo >= 0.0:
                col = col / PERCENT_SCALE
                rep.rescaled_columns.append(name)
                rep.warnings.append(f"{name}: values look like percent (max {hi:g}); rescaled to 0..1")
        bad = pd.Series(False, index=col.index)
        if spec.lo is not None:
            bad |= col < spec.lo
        if spec.hi is not None:
            bad |= col > spec.hi
        if spec.binary:
            bad |= col.notna() & ~col.isin([0.0, 1.0])
        n_bad = int(bad.sum())
        if n_bad:
            rep.out_of_range[name] = n_bad
            bounds = "{0, 1}" if spec.binary else f"[{spec.lo}, {spec.hi}] {spec.unit}"
            rep.errors.append(f"{name}: {n_bad} values outside {bounds}")
        out[name] = col


def validate(
    df: pd.DataFrame, n_cells: int | None = None, cfg: Mapping[str, Any] | None = None
) -> ValidationReport:
    """Validate a canonical long-format telemetry DataFrame.

    Tolerances come from cfg["ingest"] (config/ingest.yaml); cfg=None loads the default config.

    Checks: missing required columns; parseable timestamps; integer, non-negative cell_id
    (and < n_cells if given); unit sanity (temp_C and room_temp_C in [-40, 90] degC,
    voltage_V in [0, 5] V, soc/soh in [0, 1] with 0..100 auto-detected and rescaled,
    resistance_mOhm >= 0, bypass_state in {0, 1}); duplicate (timestamp, cell_id) rows;
    sampling gaps in the pack-wide timestamp sequence; cells with missing rows (and, if
    n_cells is given, cells with no rows at all).

    Errors: missing columns, unparseable values, out-of-range values, bad cell ids,
    duplicates. Warnings: percent rescaling, epoch-ms timestamps, sampling gaps, cells with
    missing rows. The input is not modified; see `ValidationReport.df`.
    """
    if cfg is None:
        from gb.config import load_config

        cfg = load_config()
    ing = ingest_settings(cfg)
    rep = ValidationReport()
    out = df.copy()
    rep.df = out
    rep.missing_required = [c for c in REQUIRED_COLUMNS if c not in out.columns]
    if rep.missing_required:
        rep.errors.append(f"missing required columns: {rep.missing_required}")

    _check_values(out, rep, float(ing["percent_scale_min"]))

    keys_ok = True
    if "timestamp" in out.columns:
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                out["timestamp"] = parse_timestamps(out["timestamp"], cfg)
            for w in caught:  # record ours in the report, pass anything else through
                if issubclass(w.category, TelemetryWarning):
                    rep.warnings.append(str(w.message))
                else:
                    warnings.warn_explicit(w.message, w.category, w.filename, w.lineno)
            if out["timestamp"].isna().any():
                rep.errors.append(f"timestamp: {int(out['timestamp'].isna().sum())} missing values")
                keys_ok = False
        except ValueError as exc:
            rep.errors.append(str(exc))
            keys_ok = False
    else:
        keys_ok = False

    if "cell_id" in out.columns:
        cid = pd.to_numeric(out["cell_id"], errors="coerce")
        if cid.isna().any() or not np.all(np.mod(cid.to_numpy(dtype=np.float64), 1.0) == 0.0):
            rep.errors.append("cell_id: non-integer or missing values")
            keys_ok = False
        else:
            out["cell_id"] = cid.astype("int64")
            bad_ids = out["cell_id"] < 0
            if n_cells is not None:
                bad_ids |= out["cell_id"] >= n_cells
            if bad_ids.any():
                ids = sorted(out.loc[bad_ids, "cell_id"].unique().tolist())
                rng = f"0..{n_cells - 1}" if n_cells is not None else ">= 0"
                rep.errors.append(f"cell_id outside {rng}: {_listed(ids)}")
    else:
        keys_ok = False

    if keys_ok and len(out):
        _check_keys(out, rep, n_cells, float(ing["gap_factor"]))
    return rep


def _check_keys(out: pd.DataFrame, rep: ValidationReport, n_cells: int | None, gap_factor: float) -> None:
    """Duplicates, sampling gaps and per-cell missing rows (needs parsed timestamp/cell_id)."""
    dup = out.duplicated(subset=["timestamp", "cell_id"], keep="first")
    rep.n_duplicate_rows = int(dup.sum())
    if rep.n_duplicate_rows:
        rep.errors.append(f"{rep.n_duplicate_rows} duplicate (timestamp, cell_id) rows")

    t_unique = np.unique(to_epoch_seconds(out["timestamp"]))
    if len(t_unique) >= 2:
        dt = np.diff(t_unique)
        rep.nominal_dt_s = float(np.median(dt))
        idx = np.flatnonzero(dt > gap_factor * rep.nominal_dt_s)
        rep.gaps = [
            Gap(start_epoch_s=float(t_unique[i]), end_epoch_s=float(t_unique[i + 1]), duration_s=float(dt[i]))
            for i in idx
        ]
        if rep.gaps:
            rep.warnings.append(
                f"{len(rep.gaps)} sampling gaps longer than {gap_factor:g} x median step "
                f"({rep.nominal_dt_s:g} s); longest {max(g.duration_s for g in rep.gaps):g} s"
            )

    n_t = len(t_unique)
    per_cell = out.drop_duplicates(subset=["timestamp", "cell_id"]).groupby("cell_id").size()
    if n_cells is not None:
        per_cell = per_cell.reindex(range(n_cells), fill_value=0)
    missing = (n_t - per_cell)[lambda s: s > 0]
    rep.cells_missing_rows = {int(k): int(v) for k, v in missing.items()}
    if rep.cells_missing_rows:
        rep.warnings.append(
            f"{len(rep.cells_missing_rows)} cells have missing rows (of {n_t} timestamps): "
            f"{_listed(sorted(rep.cells_missing_rows))}"
        )
