"""Tests for provibench.bench.targets."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import pytest

from provibench.bench.targets import Target, load_targets, parse_run_spec, resolve_api_key


def _target(
    name: str = "openrouter", kind: Literal["openrouter", "anthropic"] = "openrouter"
) -> Target:
    return Target(name=name, url="https://x/v1/messages", api_key_env="X_KEY", kind=kind)


def test_parse_run_spec_basic() -> None:
    targets = {"deepseek": _target("deepseek", "anthropic")}
    spec = parse_run_spec("deepseek:deepseek-flash", targets)
    assert spec.target.name == "deepseek"
    assert spec.model == "deepseek-flash"
    assert spec.providers == []
    assert spec.label == "deepseek:deepseek-flash"
    assert spec.slug == "deepseek-deepseek-flash"


def test_parse_run_spec_with_providers() -> None:
    targets = {"openrouter": _target()}
    spec = parse_run_spec("openrouter:deepseek/deepseek-v4.1-flash@novita,siliconflow", targets)
    assert spec.model == "deepseek/deepseek-v4.1-flash"
    assert spec.providers == ["novita", "siliconflow"]
    assert spec.label == "openrouter:deepseek/deepseek-v4.1-flash@novita,siliconflow"
    assert spec.slug == "openrouter-deepseek-deepseek-v4.1-flash-novita-siliconflow"


def test_parse_run_spec_preset_suffix_stays_in_model() -> None:
    targets = {"openrouter": _target()}
    spec = parse_run_spec("openrouter:deepseek/deepseek-v4.1-flash@preset/foo", targets)
    assert spec.model == "deepseek/deepseek-v4.1-flash@preset/foo"
    assert spec.providers == []


def test_parse_run_spec_unknown_target_lists_known() -> None:
    with pytest.raises(ValueError, match=r"unknown target.*deepseek"):
        parse_run_spec("nope:model", {"deepseek": _target("deepseek", "anthropic")})


def test_parse_run_spec_provider_pin_requires_openrouter_kind() -> None:
    targets = {"deepseek": _target("deepseek", "anthropic")}
    with pytest.raises(ValueError, match="openrouter"):
        parse_run_spec("deepseek:deepseek-flash@novita", targets)


def test_parse_run_spec_empty_model() -> None:
    targets = {"openrouter": _target()}
    with pytest.raises(ValueError, match="empty model"):
        parse_run_spec("openrouter:", targets)


def test_parse_run_spec_missing_colon() -> None:
    with pytest.raises(ValueError, match="missing ':'"):
        parse_run_spec("openrouter", {"openrouter": _target()})


def test_load_targets_good_shape(tmp_path: Path) -> None:
    toml = """
[targets.openrouter]
url = "https://openrouter.ai/api/v1/messages"
api_key_env = "OPENROUTER_KEY"
kind = "openrouter"

[targets.deepseek]
url = "https://api.deepseek.com/anthropic/v1/messages"
api_key_env = "DEEPSEEK_API_KEY"
kind = "anthropic"

[targets.deepseek.prices."deepseek-flash"]
input = 0.15
cache_read = 0.003
cache_write = 0.15
output = 0.60
"""
    path = tmp_path / "targets.toml"
    path.write_text(toml)
    targets = load_targets(path)
    assert set(targets) == {"openrouter", "deepseek"}
    assert targets["deepseek"].prices["deepseek-flash"].input == 0.15
    assert targets["openrouter"].prices == {}


def test_load_targets_missing_key_raises_with_path_and_key(tmp_path: Path) -> None:
    path = tmp_path / "targets.toml"
    path.write_text('[targets.openrouter]\nurl = "https://x"\n')
    with pytest.raises(ValueError, match=r"targets\.openrouter"):
        load_targets(path)


def test_load_targets_missing_targets_table(tmp_path: Path) -> None:
    path = tmp_path / "targets.toml"
    path.write_text("foo = 1\n")
    with pytest.raises(ValueError, match="targets"):
        load_targets(path)


def test_load_targets_bad_toml_syntax(tmp_path: Path) -> None:
    path = tmp_path / "targets.toml"
    path.write_text("not = [valid\n")
    with pytest.raises(ValueError, match="invalid TOML"):
        load_targets(path)


def test_resolve_api_key_missing_names_env_var() -> None:
    with pytest.raises(ValueError, match="X_KEY"):
        resolve_api_key(_target(), {})


def test_resolve_api_key_empty_string_is_missing() -> None:
    with pytest.raises(ValueError, match="X_KEY"):
        resolve_api_key(_target(), {"X_KEY": ""})


def test_resolve_api_key_present() -> None:
    assert resolve_api_key(_target(), {"X_KEY": "secret"}) == "secret"
