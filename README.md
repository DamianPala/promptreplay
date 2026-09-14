# provibench

Records real agent-harness API traffic (Claude Code, Codex) once, then replays it
byte-for-byte against LLM providers so their prompt-cache behaviour and cost are
comparable on the exact same payload instead of a re-run trajectory.
It ships a recording reverse proxy, a probe that measures prompt caching on a few real
turns for a fraction of the cost, a full replay engine that reproduces the whole recorded
turn sequence with `max_tokens: 1`, and a report that aggregates cache hit ratio, cost,
and latency per provider.

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

`record`, `inspect`, `probe`, `replay`, `report`, and `endpoints` are the domain commands
built on top of `src/provibench/bench/`; see [docs/design.md](docs/design.md) for the full
design (trace/replay format, `targets.toml`, cost model).

## Probe

The probe is the default measurement; it answers "does this provider cache my prompt, and
what does that cost" on a few real turns instead of replaying the whole conversation.

1. A rung is a recorded turn `k`; by default the smallest, middle and largest turn of the
   conversation, each of which needs a following turn.
2. The cold request sends turn `k` once and writes the provider's prefix cache.
3. The warm requests send turn `k+1` `--repeats` times: its prompt starts with the same
   bytes, so what comes back measures whether the provider cached the prefix and whether
   the next request landed on a replica that has it.
4. Every rung stamps its own nonce (`provibench-probe:<run hex>:<k>`) into the first system
   block, so a cold request really is cold and rung 30 cannot read what rung 1 wrote;
   `--warm` sends no nonce and measures the cache as found.
5. A cold request that fails skips its rung — a failed write is not a miss — and the run is
   written to disk either way; the command exits non-zero when anything failed or was
   skipped, so a script can tell.

```sh
provibench inspect my-session                     # see the turns and their sizes
provibench probe my-session deepseek:deepseek-flash \
  --rungs 1,13,30 --repeats 6,2,2 --budget 0.5
provibench report latest
```

`SPEC` is `<target>:<model>[@provider[,provider...]]`, resolved against `targets.toml`
(`@provider` is only valid for `kind = "openrouter"` targets); repeat it to probe several
providers in one run. Before sending anything, `probe` prints the worst case per spec —
every token it will send at the listed input price, assuming no cache hit — and `--budget`
refuses to run above it, because weak resellers bill close to that number.

What the columns mean:

| Column | Meaning |
|---|---|
| `hit %` | Warm reads that found any part of the prefix ÷ warm reads that were served |
| `prefix %` | On a hit, how much of the cold prompt was cached, averaged; 100 % is the whole prefix |
| `eff $/M` | What a prompt token costs at that hit rate: `(1 − h)·input + h·cache read`, `h = hit % × prefix %` |
| `in $/M` | The listed input price that calculation used: the OpenRouter endpoint's, or `targets.toml`'s |
| `cold ms` / `warm ms` | Median wall clock; the gap is the prefill the cache saved |
| `hits` | One cell per warm read: `x` failed, `1` from 0.98 up, else the cached fraction (`0.9` = a 90 % prefix) |
| `cached cold` | What the cold write read back; anything above 0 means the rung was not really cold |

A run is persisted under `runs/<trace>/<timestamp>/` as the options, the endpoint snapshot
and prices it was priced with, and one `<spec>.jsonl` per spec; `report` re-reads it with no
network. `--json` prints the same summaries for scripts.

## Full replay

Where the probe samples a few turns, `replay` sends every recorded turn, which is what you
want for the per-turn cache curve and for the "what would my own session have cost" number.
It stamps one nonce (`provibench-run:<run hex>`) into every turn unless `--warm` is given,
so turn 2 can read what turn 1 wrote — the effect being measured — without inheriting an
earlier run's cache.

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
  --budget 2 --yes

# 5. Summarise (or re-summarise) a run, optionally as markdown.
provibench report latest --md report.md
```

`--run` takes the same spec shape as `probe`'s positional `SPEC`; `report` accepts a run
directory, a trace name (its newest run), or `latest` (the newest run across every trace).
Every command accepts `--json` and the other global flags documented by `provibench schema`.

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
