"""Load parameter YAML files from config/ and build the assumptions table.

Every leaf in a YAML file is {value: ..., source: "..."}; any other mapping is a group.
load_config returns {file_stem: {key: value | {subkey: value}}} with plain values,
in the units written in the key names (e.g. _C, _h, _Ah); callers convert to SI.

References: a string value "= name" (optionally "= name + 3" / "= name - 3") is replaced
by another leaf's value. `name` is a leaf key looked up in the same file first, then in
all files (it must be unique), or an explicit dotted path "file.group.key".
References may appear inside lists and chain; cycles are an error.
"""

from __future__ import annotations

import copy
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

DEFAULT_CONFIG_DIR: Path = Path(__file__).resolve().parent.parent / "config"

SOURCE_RE = re.compile(r"^(?:(?:datasheet|paper|derived):\s*\S|(?:assumed|TBD-REON)(?![\w-]))")
REF_RE = re.compile(
    r"^=\s*([A-Za-z_][\w.]*)\s*(?:([+-])\s*(\d+(?:\.\d+)?(?:[eE][+-]?\d+)?))?\s*$"
)

KeyPath = tuple[str, ...]


class ConfigError(ValueError):
    """Malformed config file, missing source, or unresolvable reference."""


class Config(dict):
    """Nested dict of resolved plain values. `.sources[file][dotted_key]` is each leaf's source."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.sources: dict[str, dict[str, str]] = {}


def _fmt(path: KeyPath) -> str:
    return f"{path[0]}.yaml:" + ".".join(path[1:])


def _collect(node: Any, path: KeyPath, leaves: dict[KeyPath, Any], sources: dict[KeyPath, str]) -> None:
    if not isinstance(node, dict):
        raise ConfigError(f"{_fmt(path)}: expected a {{value, source}} leaf or a group, got {node!r}")
    if "value" in node or "source" in node:
        if len(path) < 2:
            raise ConfigError(f"{_fmt(path)}: top level must be a mapping of parameters")
        if set(node) != {"value", "source"}:
            raise ConfigError(f"{_fmt(path)}: leaf must have exactly keys value, source; got {sorted(node)}")
        src = node["source"]
        if not isinstance(src, str) or not SOURCE_RE.match(src):
            raise ConfigError(
                f"{_fmt(path)}: source {src!r} must start with datasheet:, paper:, derived:, assumed or TBD-REON"
            )
        leaves[path] = node["value"]
        sources[path] = src
        return
    if not node:
        raise ConfigError(f"{_fmt(path)}: empty group")
    for key, child in node.items():
        if not isinstance(key, str) or "." in key:
            raise ConfigError(f"{_fmt(path)}: invalid key {key!r}")
        _collect(child, path + (key,), leaves, sources)


class _Resolver:
    def __init__(self, leaves: dict[KeyPath, Any]) -> None:
        self.leaves = leaves
        self.done: dict[KeyPath, Any] = {}
        self.by_name: dict[str, list[KeyPath]] = defaultdict(list)
        for p in leaves:
            self.by_name[p[-1]].append(p)

    def leaf(self, path: KeyPath, stack: tuple[KeyPath, ...] = ()) -> Any:
        if path in self.done:
            return self.done[path]
        if path in stack:
            chain = " -> ".join(_fmt(p) for p in stack + (path,))
            raise ConfigError(f"reference cycle: {chain}")
        val = self._value(self.leaves[path], path, stack + (path,))
        self.done[path] = val
        return val

    def _value(self, v: Any, path: KeyPath, stack: tuple[KeyPath, ...]) -> Any:
        if isinstance(v, str) and v.lstrip().startswith("="):
            return self._ref(v, path, stack)
        if isinstance(v, list):
            return [self._value(x, path, stack) for x in v]
        if isinstance(v, dict):
            return {k: self._value(x, path, stack) for k, x in v.items()}
        return v

    def _ref(self, expr: str, at: KeyPath, stack: tuple[KeyPath, ...]) -> Any:
        m = REF_RE.match(expr.strip())
        if not m:
            raise ConfigError(f"{_fmt(at)}: bad reference {expr!r} (use '= name' or '= name +/- number')")
        name, op, num = m.groups()
        val = copy.deepcopy(self.leaf(self._find(name, at), stack))
        if op:
            if isinstance(val, bool) or not isinstance(val, (int, float)):
                raise ConfigError(f"{_fmt(at)}: cannot apply {op}{num} to non-numeric {val!r}")
            offset: int | float = int(num) if num.isdigit() else float(num)
            val = val + offset if op == "+" else val - offset
        return val

    def _find(self, name: str, at: KeyPath) -> KeyPath:
        if "." in name:
            parts = tuple(name.split("."))
            for cand in (parts, (at[0],) + parts):
                if cand in self.leaves:
                    return cand
            raise ConfigError(f"{_fmt(at)}: reference '{name}' not found")
        cands = self.by_name.get(name, [])
        local = [c for c in cands if c[0] == at[0]]
        pick = local or cands
        if len(pick) == 1:
            return pick[0]
        if not pick:
            raise ConfigError(f"{_fmt(at)}: reference '{name}' not found")
        opts = ", ".join(".".join(c) for c in pick)
        raise ConfigError(f"{_fmt(at)}: reference '{name}' is ambiguous ({opts}); use a dotted path")


def load_config(config_dir: Path | str = DEFAULT_CONFIG_DIR) -> Config:
    """Load every *.yaml in `config_dir` (keyed by file stem), validate sources, resolve references."""
    config_dir = Path(config_dir)
    files = sorted(config_dir.glob("*.yaml"))
    if not files:
        raise FileNotFoundError(f"no *.yaml files in {config_dir}")
    leaves: dict[KeyPath, Any] = {}
    sources: dict[KeyPath, str] = {}
    for f in files:
        with f.open(encoding="utf-8") as fh:
            raw = yaml.safe_load(fh)
        _collect({} if raw is None else raw, (f.stem,), leaves, sources)

    resolver = _Resolver(leaves)
    cfg = Config()
    for path in leaves:
        node: dict[str, Any] = cfg
        for key in path[:-1]:
            node = node.setdefault(key, {})
        node[path[-1]] = resolver.leaf(path)
        cfg.sources.setdefault(path[0], {})[".".join(path[1:])] = sources[path]
    return cfg


def assumptions_table(cfg: Config) -> pd.DataFrame:
    """One row per leaf parameter: [file, key, value, source]; value is the resolved one."""
    rows = []
    for file, keys in cfg.sources.items():
        for key, src in keys.items():
            node: Any = cfg[file]
            for k in key.split("."):
                node = node[k]
            rows.append({"file": file, "key": key, "value": node, "source": src})
    return pd.DataFrame(rows, columns=["file", "key", "value", "source"])
