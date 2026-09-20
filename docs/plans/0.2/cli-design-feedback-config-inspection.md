# Field report to the CLI Design Standard: `config show` names no way to change a setting

Date: 2026-09-20. Tool: provibench 0.2.0 (claim `0.1.0-draft.7`). Rules touched: I2a, I2c. Starter: `core/config_command.py`, `core/settings.py`.

## The case

A reader of the README asked whether the Configuration table (setting, flag, environment variable, config key, default) could be replaced by a pointer to something.
It could not.
The table is the only place where the names `PROVIBENCH_CONFIG`, `PROVIBENCH_TARGETS`, `PROVIBENCH_TRACES_DIR` and `PROVIBENCH_RUNS_DIR` appear:

- `provibench config show` prints `setting | value | source`. It says a value came from `env` or `default`, but not which variable, flag or file key would change it.
- `provibench --help` describes `--config` and `--targets` as replaceable by "a flag or environment selection" without naming the variable.
- `provibench schema` has no settings section.

So an agent that has only the tool cannot learn how to pin `runs_dir` before its first paid run; the skill has to hardcode `PROVIBENCH_RUNS_DIR`, and the README has to carry a table the tool could have printed.

## What the standard asks today

I2a: sources, including tool-specific environment variables, MUST be documented.
I2c: layered configuration SHOULD expose each resolved value and its source and, when the source has a location, its resolved location.

Both are satisfied by provibench, and the gap stays open: I2a is met by prose in the README, I2c by a record that names the source kind but not the handle.
The record I2c describes is also what D9a's "complete argument list for the next call" relies on, and that list needs the handle, not the kind.

## Proposal for the standard

Loosen nothing; widen one clause of I2c instead of adding a rule:

> **I2c:** Inspection: layered configuration SHOULD expose each resolved value, its source, the handle that source uses (flag, environment variable, or configuration key), and, when the source has a location, its resolved location, while masking secrets.

With that clause, I2a's documentation duty for environment variables is met by the tool's own inspection output, and a README can point at `config show` instead of repeating it.

## Proposal for the starter

`Setting` in `core/settings.py` already carries `flag`, `env` and `config_key`; `config show` in `core/config_command.py` drops them on the way out.

1. `config show` adds the columns `flag`, `env` and `config_key` (nullable strings in the JSON schema, `-` in text), taken from the declaration. About 20 lines plus tests.
2. The help of a flag that selects a file (`--config`, `--targets`) names the variable: "or `$MYTOOL_CONFIG`" instead of "a flag or environment selection".

Whether `schema` should also list settings is a contract question (D6/D8) and is left out of this report.

## What provibench did

Kept the table, shrunk to setting, flag, environment variable and default, with one sentence saying a config key is the setting's name.
The starter change above is queued for 0.3 so that the README section can become a sentence pointing at `config show`.
