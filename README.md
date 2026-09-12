# provibench

Records real agent-harness API traffic (Claude Code, Codex) once, then replays it
byte-for-byte against LLM providers so their prompt-cache behaviour and cost are
comparable on the exact same payload instead of a re-run trajectory.
It ships a recording reverse proxy, a replay engine that reproduces the recorded turn
sequence with `max_tokens: 1`, and a report that aggregates cache hit ratio, cost, and
latency per provider.

## Setup

Requires Python 3.13 or newer and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
```

This creates `.venv/` with the tool and its development dependencies (ruff, pyright,
pytest) and installs `provibench` in editable mode.

## Commands

```sh
provibench --help
provibench schema                  # the machine-readable interface
provibench config show             # effective settings and their source
provibench completion --install    # shell completions
```

`record`, `inspect`, `replay`, `report`, and `endpoints` are the domain commands built on
top of `src/provibench/bench/`; see [docs/design.md](docs/design.md) for the full design
(trace/replay format, `targets.toml`, cost model). A typical session:

```sh
# 1. Record real traffic once, by pointing the harness at the proxy.
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
provibench record --name my-session --upstream https://api.anthropic.com
# ^ runs until Ctrl-C; appends to <traces-dir>/my-session.jsonl

# 2. See what got captured.
provibench inspect my-session

# 3. List a model's OpenRouter endpoints and their prices, to build a run spec.
provibench endpoints deepseek/deepseek-v4.1-flash

# 4. Replay the recorded turns against one or more targets (spends API credit).
provibench replay my-session \
  --run openrouter:deepseek/deepseek-v4.1-flash@novita \
  --run deepseek:deepseek-flash \
  --yes

# 5. Summarise (or re-summarise) a run, optionally as markdown.
provibench report latest --md report.md
```

`--run` takes a `<target>:<model>[@provider[,provider...]]` spec resolved against
`targets.toml` (`@provider` is only valid for `kind = "openrouter"` targets); `report`
accepts a run directory, a trace name (its newest run), or `latest` (the newest run
across every trace). Every command accepts `--json` and the other global flags
documented by `provibench schema`.

## Configuration

Precedence is flag, then environment variable, then configuration file, then built-in
default. `provibench config show` prints every effective value with its source.

| Setting | Flag | Environment | Config key | Default |
|---|---|---|---|---|
| `config_path` | `--config`, `-c` | `PROVIBENCH_CONFIG` | (not a key) | `$XDG_CONFIG_HOME/provibench/config.toml`, else `~/.config/provibench/config.toml` |
| `targets_path` | `--targets` | `PROVIBENCH_TARGETS` | `targets_path` | `$XDG_CONFIG_HOME/provibench/targets.toml`, else `~/.config/provibench/targets.toml` |
| `traces_dir` | none | `PROVIBENCH_TRACES_DIR` | `traces_dir` | `./traces` |
| `runs_dir` | none | `PROVIBENCH_RUNS_DIR` | `runs_dir` | `./runs` |

A packaged default `targets.toml` (in `src/provibench/data/`) ships alongside the tool
as a fallback for a fresh install; `docs/design.md` documents its format.

## Tests

```sh
uv run pytest
```

`tests/test_seams.py` enforces the architecture: `core/` imports nothing from
`commands/` or `bench/`, and no module exceeds 400 lines.
