import math
from pathlib import Path
from typing import Any

import pytest
import yaml

from gb.config import DEFAULT_CONFIG_DIR, SOURCE_RE, ConfigError, assumptions_table, load_config

FILES = {"cell", "pack", "site", "aging", "controllers", "econ", "column_map", "identify", "ingest", "scenarios"}


def _count_leaves(node: Any) -> int:
    assert isinstance(node, dict), f"non-mapping node without a source: {node!r}"
    if "value" in node:
        return 1
    return sum(_count_leaves(v) for v in node.values())


def _walk_values(node: Any):
    if isinstance(node, dict):
        for v in node.values():
            yield from _walk_values(v)
    elif isinstance(node, list):
        for v in node:
            yield from _walk_values(v)
    else:
        yield node


def test_config_loads(cfg) -> None:
    assert set(cfg) == FILES


def test_every_leaf_has_a_source(cfg) -> None:
    n_leaves = 0
    for f in DEFAULT_CONFIG_DIR.glob("*.yaml"):
        n_leaves += _count_leaves(yaml.safe_load(f.read_text(encoding="utf-8")))
    table = assumptions_table(cfg)
    assert len(table) == n_leaves
    assert list(table.columns) == ["file", "key", "value", "source"]
    assert all(SOURCE_RE.match(s) for s in table["source"])


def test_references_resolve(cfg) -> None:
    c = cfg["controllers"]
    assert c["T_crit_C"] == 57
    assert c["threshold_bypass"]["T_on_C"] == c["T_band_edge_C"] == 30
    assert c["threshold_bypass"]["T_off_C"] == 27
    assert c["derate"]["T_start_C"] == 30
    assert c["derate"]["T_stop_C"] == c["T_crit_C"]
    assert c["zone_soc_cap"]["caps"] == [cfg["cell"]["soc_max"], 0.80, 0.70]
    leftover = [v for v in _walk_values(cfg) if isinstance(v, str) and v.startswith("=")]
    assert leftover == []


def test_numeric_parsing(cfg) -> None:
    assert isinstance(cfg["cell"]["R_dc_ref_ohm"], float)
    assert cfg["cell"]["R_dc_ref_ohm"] == pytest.approx(0.56e-3)
    assert cfg["aging"]["B_cyc"] == pytest.approx(5.0e-6)


def test_derived_values_consistent(cfg) -> None:
    cell, pack, scale = cfg["cell"], cfg["pack"], cfg["site"]["scale"]
    string_kwh = pack["n_modules"] * pack["cells_per_module"] * cell["Q_nom_Ah"] * cell["V_nom"] / 1000
    assert string_kwh == pytest.approx(58.5, abs=0.05)
    assert scale["strings_per_cess"] == math.ceil(scale["cess_energy_kWh"] / string_kwh)


def test_small_pack_fixture(cfg, small_pack) -> None:
    assert small_pack.layout.n == 8
    assert small_pack.cfg["pack"]["n_modules"] == 2
    assert cfg["pack"]["n_modules"] == 8  # original untouched
    assert small_pack.cfg.sources == cfg.sources


def _load(tmp_path: Path, text: str):
    (tmp_path / "t.yaml").write_text(text, encoding="utf-8")
    return load_config(tmp_path)


@pytest.mark.parametrize(
    "text, match",
    [
        ("x: {value: 1}\n", "exactly keys"),
        ("x: {value: 1, source: guess}\n", "source"),
        ("x: {value: 1, source: assumed, unit: K}\n", "exactly keys"),
        ("x: 1\n", "leaf or a group"),
        ("g:\n  y: 3\n", "leaf or a group"),
        ("x: {value: '= nope', source: assumed}\n", "not found"),
        ("x: {value: '= y', source: assumed}\ny: {value: '= x', source: assumed}\n", "cycle"),
        ("x: {value: '= y * 2', source: assumed}\ny: {value: 1, source: assumed}\n", "bad reference"),
        ("a:\n  k: {value: 1, source: assumed}\nb:\n  k: {value: 2, source: assumed}\n"
         "x: {value: '= k', source: assumed}\n", "ambiguous"),
    ],
)
def test_bad_config_rejected(tmp_path: Path, text: str, match: str) -> None:
    with pytest.raises(ConfigError, match=match):
        _load(tmp_path, text)


def test_reference_forms(tmp_path: Path) -> None:
    cfg = _load(
        tmp_path,
        "a:\n  k: {value: 10, source: assumed}\n"
        "b:\n  k: {value: 2.5, source: assumed}\n"
        "x: {value: '= a.k', source: assumed}\n"
        "y: {value: '= t.b.k + 0.5', source: assumed}\n"
        "z: {value: ['= x', '= y - 1'], source: assumed}\n",
    )
    t = cfg["t"]
    assert t["x"] == 10 and t["y"] == pytest.approx(3.0) and t["z"] == [10, pytest.approx(2.0)]


def test_empty_dir(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_config(tmp_path)
