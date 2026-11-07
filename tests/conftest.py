"""Shared fixtures."""

import copy
from dataclasses import dataclass

import pytest

from gb.config import Config, load_config
from gb.types import Layout, build_layout


@dataclass
class SmallPack:
    """A 2 x 4 = 8 cell pack for fast tests: cfg with pack size overridden, plus its layout."""

    cfg: Config
    layout: Layout


@pytest.fixture
def cfg() -> Config:
    """The default config from config/*.yaml (fresh per test, safe to mutate)."""
    return load_config()


@pytest.fixture
def small_pack(cfg: Config) -> SmallPack:
    small = copy.deepcopy(cfg)
    small["pack"]["n_modules"] = 2
    small["pack"]["cells_per_module"] = 4
    return SmallPack(cfg=small, layout=build_layout(2, 4))
