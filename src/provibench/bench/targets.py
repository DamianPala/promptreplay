"""targets.toml parsing and endpoint resolution."""

from __future__ import annotations

import re
import tomllib
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from pydantic import BaseModel, Field

_SLUG_UNSAFE = re.compile(r"[^A-Za-z0-9._-]")
_PRESET_PREFIX = "preset/"


class Prices(BaseModel):
    """USD per 1M tokens."""

    input: float
    cache_read: float
    cache_write: float
    output: float


class Target(BaseModel):
    name: str
    url: str
    api_key_env: str
    kind: Literal["openrouter", "anthropic"]
    prices: dict[str, Prices] = Field(default_factory=dict)
    aliases: dict[str, str] = Field(default_factory=dict)
    """`"<openrouter slug>" = "<native model>"`, so a sweep of the slug also probes this target."""
    litellm_provider: str | None = None
    """How LiteLLM namespaces this target's models (`deepseek` in `deepseek/deepseek-flash`).

    It is the first key the price lookup tries; without it only the bare model name is, so a
    target never gets another provider's rate by sharing a model name.
    """


def load_targets(path: Path) -> dict[str, Target]:
    """Parse the `[targets.<name>]` / `[targets.<name>.prices."<model>"]` TOML shape."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{path}: invalid TOML: {exc}") from exc

    section = raw.get("targets")
    if not isinstance(section, dict):
        raise ValueError(f"{path}: missing [targets] table")
    tables = cast(dict[str, Any], section)

    targets: dict[str, Target] = {}
    for name, table in tables.items():
        if not isinstance(table, dict):
            raise ValueError(f"{path}: targets.{name} must be a table")
        fields = cast(dict[str, Any], table)
        try:
            targets[name] = Target(
                name=name,
                url=fields["url"],
                api_key_env=fields["api_key_env"],
                kind=fields["kind"],
                prices=fields.get("prices", {}),
                aliases=fields.get("aliases", {}),
                litellm_provider=fields.get("litellm_provider"),
            )
        except KeyError as exc:
            raise ValueError(f"{path}: targets.{name} missing key {exc}") from exc
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{path}: targets.{name}: {exc}") from exc
    return targets


class RunSpec(BaseModel):
    target: Target
    model: str
    providers: list[str] = Field(default_factory=list)

    @property
    def label(self) -> str:
        base = f"{self.target.name}:{self.model}"
        if self.providers:
            base += "@" + ",".join(self.providers)
        return base

    @property
    def slug(self) -> str:
        return _SLUG_UNSAFE.sub("-", self.label)

    @property
    def kind(self) -> str:
        """The target's wire kind, so an endpoint can be compared without its whole target."""
        return self.target.kind


def _split_providers(rest: str) -> tuple[str, list[str]]:
    """Split "<model>[@p1,p2]" on the LAST '@' whose suffix is not a preset.

    A model id may itself carry an OpenRouter preset (`model@preset/foo`), and that suffix
    stays part of the model. It is the `preset/` prefix that says so, not a slash anywhere
    in the suffix: a provider tag may hold one too (`@novita/fp8`), and a pin this parser
    dropped would be a silently unpinned request against a model name that does not exist.
    """
    at = rest.rfind("@")
    while at != -1:
        candidate = rest[at + 1 :]
        if not candidate.startswith(_PRESET_PREFIX):
            model = rest[:at]
            providers = [p.strip() for p in candidate.split(",") if p.strip()]
            return model, providers
        at = rest.rfind("@", 0, at)
    return rest, []


def parse_run_spec(spec: str, targets: dict[str, Target]) -> RunSpec:
    """Parse "<target>:<model>[@p1,p2]"."""
    target_name, sep, rest = spec.partition(":")
    if not sep:
        raise ValueError(f"invalid endpoint {spec!r}: missing ':' between target and model")
    target = targets.get(target_name)
    if target is None:
        known = ", ".join(sorted(targets)) or "(none)"
        raise ValueError(f"unknown target {target_name!r} in {spec!r}; known targets: {known}")

    model, providers = _split_providers(rest)
    if not model:
        raise ValueError(f"invalid endpoint {spec!r}: empty model")
    if providers and target.kind != "openrouter":
        raise ValueError(
            f"invalid endpoint {spec!r}: provider pinning ('@...') requires "
            f'kind="openrouter", target {target_name!r} is kind={target.kind!r}'
        )
    return RunSpec(target=target, model=model, providers=providers)


def resolve_api_key(target: Target, env: Mapping[str, str]) -> str:
    key = env.get(target.api_key_env)
    if not key:
        raise ValueError(
            f"target {target.name!r} needs env var {target.api_key_env!r}, "
            "which is missing or empty"
        )
    return key
