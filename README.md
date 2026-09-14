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

`record`, `inspect`, `probe`, `replay`, `report`, `scrub`, and `endpoints` are the domain
commands built on top of `src/provibench/bench/`; see [docs/design.md](docs/design.md) for
the full design (trace/replay format, `targets.toml`, cost model).

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
5. After the warm reads, one streamed generation on the same prompt (`max_tokens 256`,
   `temperature 0`) gives time-to-first-token, tokens per second and the model's first
   output tokens; `--no-throughput` skips it, and it costs output tokens. The fingerprint
   is for eyeballing: at temperature 0 two providers of one model still open differently,
   because quantization and the chat template change what a model says, not how it caches,
   so a difference there is not a fault. The tokens and models a spec was served by are
   compared, in `drift` and in `--json`.
6. `--ttl 60,300` re-reads the first served rung's cache that many seconds after its last
   warm read, which is what turns "it is cached" into "it is cached for at least this
   long"; a rung whose cold request failed wrote nothing to re-read, so the next one
   carries the offsets. It is off by default because it costs wall time, and `--gap` must
   be smaller than the smallest offset.
7. A cold request that fails skips its rung — a failed write is not a miss — and the run is
   written to disk either way; the command exits non-zero when anything failed or was
   skipped, so a script can tell.

```sh
provibench inspect my-session                     # see the turns and their sizes
provibench probe my-session deepseek:deepseek-flash \
  --rungs 1,13,30 --repeats 6,2,2 --budget 0.5
provibench probe my-session openrouter:deepseek/deepseek-v4.1-flash@novita \
  --ttl 60,300 --yes
provibench report latest
```

`SPEC` is `<target>:<model>[@provider[,provider...]]`, resolved against `targets.toml`
(`@provider` is only valid for `kind = "openrouter"` targets); repeat it to probe several
providers in one run. Before sending anything, `probe` prints the worst case per spec —
every prompt token at the listed input price and the throughput request's output tokens at
the output price, assuming no cache hit — and `--budget` refuses to run above it, because
weak resellers bill close to that number.

What the columns mean:

| Column | Meaning |
|---|---|
| `hit %` | Warm reads that found any part of the prefix ÷ warm reads that were served |
| `prefix %` | On a hit, how much of the cold prompt was cached, averaged; 100 % is the whole prefix |
| `eff $/M` | What a prompt token costs at that hit rate: `(1 − h)·input + h·cache read`, `h = hit % × prefix %` |
| `in $/M` | The listed input price that calculation used: the OpenRouter endpoint's, or `targets.toml`'s |
| `cold ms` / `warm ms` | The first rung's cold and warm prefill; the gap is what the cache saved |
| `TTFT ms` / `tok/s` | Time to the first streamed output token, and output tokens per second after it; the median over the spec's rungs, so one slow rung cannot set the number |
| `errors` | Requests that produced no usable answer |
| `drift` | Short markers versus the reference spec: `provider` (the served provider differs from the pinned one, or varies), `model` (the response model differs or varies), `tokens±N%` (prompt size differs by ≥ 1 %); `-` when nothing drifts |
| `hits` | One cell per warm read: `x` failed, `1` from 0.98 up, else the cached fraction (`0.9` = a 90 % prefix) |
| `ttft ms` / `tok/s` | The same two numbers per rung, taken from that rung's streamed request |
| `ttl` | One cell per `--ttl` offset: `60s:1` means the cache was still there at 60 s, `300s:0` that it was not, `60s:1 (+6)` that the read went out six seconds late |
| `cached cold` | What the cold write read back; anything above 0 means the rung was not really cold |

When every spec shares one `target:model`, the tables move it into a `specs:` line above them
and the `spec` column shows just the provider tails (`@novita`, `@gmicloud`); a spec of
another model keeps its whole name. The tables are planned to fit 120 columns without
shortening two rows into the same name; a rung that carries a late `--ttl` read is the one
thing allowed to run longer, because that marker is worth the columns.

A run is persisted under `runs/<trace>/<timestamp>/` as the options, the endpoint snapshot
and prices it was priced with, and one `<spec>.jsonl` per spec; `report` re-reads it with no
network. `--json` prints the same summaries for scripts, including the fields the terminal
tables leave out: the served provider names (`served`), every model the responses named
(`models_seen`), and the raw per-rung records the columns are folded from.

## Full replay

Where the probe samples a few turns, `replay` sends every recorded turn, which is what you
want for the per-turn cache curve and for the "what would my own session have cost" number.
It stamps one nonce (`provibench-run:<run hex>`) into every turn unless `--warm` is given,
so turn 2 can read what turn 1 wrote — the effect being measured — without inheriting an
earlier run's cache. A typical session:

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

## Sharing a trace

A trace records request bytes verbatim, so it carries home paths, the operator's
instruction files, `metadata.user_id`, and any key that travelled in a header or a body.
Before sharing a recording, write a scrubbed copy:

```sh
provibench scrub my-session --out my-session.shared.jsonl.gz
provibench scrub my-session --out clean.jsonl \
  --turns 20 --replace acme-corp=example --user haz
```

`scrub` drops `body.metadata`, rewrites every `/home/<name>` and `/Users/<name>` path to
`/home/user` (the names it saw are reported, not written), including the copy Claude Code
encodes into its state directories (`-home-<name>-recwork`), and masks secrets and e-mail
addresses as `[scrubbed:<kind>]`: Anthropic (`sk-ant-`), OpenAI-style (`sk-` followed by
20+ key characters, so `sk-proj-…` too), OpenRouter (`sk-or-`), `Bearer <token>`, AWS
(`AKIA…`), GitHub (`ghp_`, `gho_`, `github_pat_`) and Slack (`xox[abp]-`) tokens.
`--allow-email` keeps an address you are allowed to share and `--turns N` keeps only the
first N entries of the main conversation (a trace without conversation keys keeps its
first N entries instead; the report says which rule chose them). `--force` overwrites an
existing `--out`. The report lists the per-rule replacement counts, the user names, the
entries dropped and the payload sizes; `--json` gives the same object.

Two options remove what the rules cannot know. `--replace OLD=NEW` does a literal
replacement in every string, after the built-in rules; `--user NAME` replaces whole-word
occurrences of a name with `user`, for `ls -l` owner columns (`-rw-rw-r-- 1 haz haz`) and
prose. Both are repeatable. `--user` is not automatic because a name can be an ordinary
word: `--replace` for a repository, a customer or a hostname, `--user` for a person.
Read the copy before sending it either way. Nested `metadata` blocks inside the request
body are left in place — only the request's own `body.metadata` is removed — and values
that are not paths, secrets or addresses (a bare `user_id`, for instance) come through
unchanged.

`TRACE` is an existing path (gzipped or not), a name under `traces_dir`, or `sample`, the
packaged example trace that `inspect`, `replay` and `scrub` all accept; `--out` ending in
`.gz` is written compressed. Recording still writes plain jsonl.

The packaged sample is the first 30 turns of a real Claude Code session on DeepSeek V4.1
Flash (raising test coverage in the public [acpc](https://github.com/DamianPala/acpc)
repository), recorded from an isolated home and scrubbed with this command: prompts grow
from 20k to 93k tokens, 1,9 M prompt tokens in total.

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
