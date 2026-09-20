"""Declared configuration (I2): settings, their sources, precedence, and the TOML loader.

A tool declares its settings once as `Setting` values. `resolve_settings` applies the fixed
precedence `flag > environment > configuration file > built-in default` and records the
winning source for `config show`. Paths from flags and the environment resolve against
the working directory; paths inside the TOML file resolve against the file's directory; a
plain relative default (`CwdPath`) resolves against the working directory as well.
"""

import tomllib
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from provibench.core.errors import InvalidInput
from provibench.core.input import InputTooLarge, InvalidUtf8, read_bounded_file

CONFIG_PATH = "config_path"
"""Name of the setting that selects the configuration file itself."""

MAX_CONFIG_BYTES = 65_536
STDIN_MARKER = "-"


class Source(StrEnum):
    """Where a resolved value came from."""

    FLAG = "flag"
    ENV = "env"
    CONFIG = "config"
    DEFAULT = "default"
    PACKAGED = "packaged"


@dataclass(frozen=True, slots=True)
class XdgPath:
    """A default path under an XDG base directory, with a home-relative fallback."""

    env: str
    fallback: str
    tail: str

    def resolve(self, env: Mapping[str, str], home: Path) -> Path:
        """The path under `$env` when set, otherwise under `home/fallback`."""
        base = env.get(self.env)
        root = Path(base) if base else home / self.fallback
        return root / self.tail


@dataclass(frozen=True, slots=True)
class CwdPath:
    """A default path resolved against the working directory, such as `./traces`."""

    tail: str

    def resolve(self, cwd: Path) -> Path:
        """The path under the working directory."""
        return cwd / self.tail


type Validator = Callable[[str], str | None]
"""Returns a problem description for a raw value, or `None` when the value is acceptable."""


@dataclass(frozen=True, slots=True)
class Setting:
    """One declared setting and every source that may supply it."""

    name: str
    description: str
    flag: str | None = None
    env: str | None = None
    config_key: str | None = None
    default: str | XdgPath | CwdPath | None = None
    secret: bool = False
    is_path: bool = False
    validate: Validator | None = None
    packaged: Callable[[], Path] | None = None
    """A file this path setting falls back to naming, in `config show` only, when its
    resolved default does not exist on disk; shown with `Source.PACKAGED` instead of
    `Source.DEFAULT`. Resolution itself (`resolve_settings`) never uses this: a command
    that needs the value still gets the (possibly nonexistent) default path and decides
    for itself whether to fall back to a packaged copy."""


@dataclass(frozen=True, slots=True)
class Resolved:
    """A setting's effective value and the source that supplied it."""

    value: str | None
    source: Source


def core_settings(program: str) -> list[Setting]:
    """The starter's shared configuration-file setting."""
    prefix = program.upper().replace("-", "_")
    return [
        Setting(
            CONFIG_PATH,
            "Configuration file, at most 65536 UTF-8 bytes; --config replaces the user file "
            "instead of augmenting it",
            flag="config",
            env=f"{prefix}_CONFIG",
            default=XdgPath("XDG_CONFIG_HOME", ".config", f"{program}/config.toml"),
            is_path=True,
        ),
    ]


def resolve_settings(
    declaration: Sequence[Setting],
    *,
    flags: Mapping[str, str | None],
    env: Mapping[str, str],
    cwd: Path,
    home: Path,
) -> dict[str, Resolved]:
    """Every declared setting resolved in the fixed precedence.

    `flags` maps flag names (`config`, `targets`) to the values given on the command line.
    """
    config_setting = next(setting for setting in declaration if setting.name == CONFIG_PATH)
    config = _resolve_config_path(config_setting, flags=flags, env=env, cwd=cwd, home=home)
    file_values: Mapping[str, str] = {}
    file_dir: Path | None = None
    if config.value is not None:
        file_values = load_config_file(Path(config.value), declaration)
        file_dir = Path(config.value).parent
    resolved = {CONFIG_PATH: config}
    for setting in declaration:
        if setting.name == CONFIG_PATH:
            continue
        resolved[setting.name] = _resolve_one(
            setting,
            flags=flags,
            env=env,
            file_values=file_values,
            file_dir=file_dir,
            cwd=cwd,
            home=home,
        )
    return resolved


def load_config_file(path: Path, declaration: Sequence[Setting]) -> dict[str, str]:
    """The TOML file's values keyed by config key; unknown keys and wrong types are rejected."""
    try:
        text = read_bounded_file(path, max_bytes=MAX_CONFIG_BYTES)
        data = tomllib.loads(text)
    except InputTooLarge as error:
        raise InvalidInput(
            f"Configuration file {path} exceeds the {MAX_CONFIG_BYTES} byte limit"
        ) from error
    except InvalidUtf8 as error:
        raise InvalidInput(f"Configuration file {path} is not valid UTF-8") from error
    except OSError as error:
        raise InvalidInput(f"Cannot read configuration file {path}: {error.strerror}") from error
    except tomllib.TOMLDecodeError as error:
        raise InvalidInput(f"Configuration file {path} is not valid TOML: {error}") from error
    known = {setting.config_key for setting in declaration if setting.config_key is not None}
    values: dict[str, str] = {}
    for key, value in data.items():
        if key not in known:
            raise InvalidInput(
                f"Unknown key {key!r} in configuration file {path}",
                hint=f"Known keys: {', '.join(sorted(known))}",
            )
        if not isinstance(value, str):
            raise InvalidInput(f"Key {key!r} in configuration file {path} must be a string")
        values[key] = value
    return values


def _resolve_config_path(
    setting: Setting,
    *,
    flags: Mapping[str, str | None],
    env: Mapping[str, str],
    cwd: Path,
    home: Path,
) -> Resolved:
    explicit = _explicit_value(setting, flags=flags, env=env)
    if explicit is not None:
        raw, source = explicit
        path = cwd / raw
        if not path.is_file():
            raise InvalidInput(
                f"Configuration file {path} does not exist",
                hint="Pass --config with an existing file or unset the variable",
            )
        return Resolved(str(path), source)
    if isinstance(setting.default, XdgPath):
        default = setting.default.resolve(env, home)
        if default.is_file():
            return Resolved(str(default), Source.DEFAULT)
    return Resolved(None, Source.DEFAULT)


def _resolve_one(
    setting: Setting,
    *,
    flags: Mapping[str, str | None],
    env: Mapping[str, str],
    file_values: Mapping[str, str],
    file_dir: Path | None,
    cwd: Path,
    home: Path,
) -> Resolved:
    explicit = _explicit_value(setting, flags=flags, env=env)
    if explicit is not None:
        raw, source = explicit
        base = cwd
    elif setting.config_key is not None and setting.config_key in file_values:
        raw, source = file_values[setting.config_key], Source.CONFIG
        base = file_dir
    else:
        default = _default_value(setting, env=env, cwd=cwd, home=home)
        if isinstance(default, Resolved):
            return default
        if default is None:
            return Resolved(None, Source.DEFAULT)
        raw, source, base = default, Source.DEFAULT, None
    if setting.validate is not None:
        problem = setting.validate(raw)
        if problem is not None:
            raise InvalidInput(f"Invalid {setting.name} from {_origin(setting, source)}: {problem}")
    if setting.is_path and raw != STDIN_MARKER and base is not None:
        raw = str(base / raw)
    return Resolved(raw, source)


def _default_value(
    setting: Setting, *, env: Mapping[str, str], cwd: Path, home: Path
) -> Resolved | str | None:
    """The setting's plain string default, or its already-resolved path default."""
    if isinstance(setting.default, XdgPath):
        return Resolved(str(setting.default.resolve(env, home)), Source.DEFAULT)
    if isinstance(setting.default, CwdPath):
        return Resolved(str(setting.default.resolve(cwd)), Source.DEFAULT)
    return setting.default


def _explicit_value(
    setting: Setting, *, flags: Mapping[str, str | None], env: Mapping[str, str]
) -> tuple[str, Source] | None:
    """The flag or environment value when one is given; an empty variable counts as unset."""
    if setting.flag is not None:
        flag_value = flags.get(setting.flag)
        if flag_value is not None:
            return flag_value, Source.FLAG
    if setting.env is not None:
        env_value = env.get(setting.env)
        if env_value:
            return env_value, Source.ENV
    return None


def _origin(setting: Setting, source: Source) -> str:
    match source:
        case Source.FLAG:
            return f"--{setting.flag}"
        case Source.ENV:
            return f"{setting.env}"
        case Source.CONFIG:
            return f"configuration key {setting.config_key!r}"
        case Source.DEFAULT:
            return "the built-in default"
        case Source.PACKAGED:
            # Resolution itself never assigns this source (see `Setting.packaged`), so a
            # validated value is never blamed on it.
            raise AssertionError("PACKAGED is a config-show display source, never a resolved one")
