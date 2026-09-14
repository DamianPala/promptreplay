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

`record`, `inspect`, `probe`, `sweep`, `replay`, `report`, `history`, `compare`, `scrub`,
`endpoints`, and `prices` are the domain commands built on top of `src/provibench/bench/`; see
[docs/design.md](docs/design.md) for the full design (trace/replay format, `targets.toml`,
cost model).

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
weak resellers bill close to that number. Its spec column is the tables' one, caption and
all, so the rows of a 16-endpoint sweep are told apart before the run is paid for; the
refusal's hint names the ways to fit a budget: fewer specs or rungs, lower `--repeats`, or
`--no-throughput`.

A native target is priced from LiteLLM's community table
(`model_prices_and_context_window.json`), cached under
`$XDG_CACHE_HOME/provibench/litellm-prices.json` and refetched at most once a week, or on
demand with `provibench prices --update`; `provibench prices` prints each model's four prices
and the source they resolved to. A `targets.toml` `prices` entry wins over that table: precedence is
`targets.toml`, then LiteLLM, then nothing, and a model neither lists stays `n/a`, counting
tokens only, which `--budget` refuses. The caveat the community table carries is that it
lists peak rates — DeepSeek, for one, halves them off-peak — and that is why the packaged
`targets.toml` keeps its own off-peak entry for `deepseek-flash` rather than take the table's.

What the columns mean:

| Column | Meaning |
|---|---|
| `hit %` | Warm reads that found any part of the prefix ÷ warm reads that were served |
| `prefix %` | On a hit, how much of the cold prompt was cached, averaged; 100 % is the whole prefix |
| `eff $/M` | What a prompt token costs at that hit rate: `(1 − h)·input + h·cache read`, `h = hit % × prefix %` |
| `in $/M` | The listed input price that calculation used: the OpenRouter endpoint's, or `targets.toml`'s |
| `cold ms` / `warm ms` | The first rung's cold and warm prefill; the gap is what the cache saved |
| `TTFT ms` / `tok/s` | Time to the first streamed output token, and output tokens per second after it; the median over the spec's rungs, so one slow rung cannot set the number. A provider that buffers the answer and flushes it whole leaves no generation window to divide, so its `tok/s` is `-` and the spec's notes say `burst delivery on rung N` |
| `errors` | Requests that produced no usable answer; when every cold write failed the same way, the notes under the tables say what the provider called it — `skipped, not_found: Paid model training violation (account settings)` for an endpoint the account's OpenRouter settings exclude |
| `drift` | Short markers versus the reference spec: `provider` (the served provider differs from the pinned one, or varies), `model` (the response model differs or varies), `tokens±N%` (prompt size differs by ≥ 1 %); `-` when nothing drifts |
| `hits` | One cell per warm read: `x` failed, `1` from 0.98 up, else the cached fraction (`0.9` = a 90 % prefix) |
| `ttft ms` / `tok/s` | The same two numbers per rung, taken from that rung's streamed request |
| `ttl` | One cell per `--ttl` offset: `60s:1` means the cache was still there at 60 s, `300s:0` that it was not, `60s:1 (+6)` that the read went out six seconds late |
| `cached cold` | What the cold write read back; anything above 0 means the rung was not really cold |

When every spec shares one `target:model`, the tables move it into a `specs:` line above
them and the `spec` column shows just the provider tails (`@novita`, `@gmicloud`); a spec
of another model keeps its whole name: that row is the reference the `@tags` are compared
against, so it wins the width it needs to stay readable. The tables are planned to fit
120 columns without shortening two rows into the same name; a rung that carries a late
`--ttl` read is allowed to run longer, because that marker is worth the columns, and so
is a table whose reference row had to be wider than the budget. `history` is the one table
outside that budget: its trace and protocol columns say which recording and which
measurement a row is, and that is worth more than the width they take.

A run is persisted under `runs/<trace>/<timestamp>/` as the options, the endpoint snapshot —
one record per spec: the endpoint it pinned, with its quantization, context length, uptime 1d
and status at the time — the `prices` block it was priced with, and one `<spec>.jsonl` per
spec; `report` re-reads it with no network. `--json` prints the same summaries for scripts,
including the fields the terminal tables leave out: the served provider names (`served`),
every model the responses named
(`models_seen`), and the raw per-rung records the columns are folded from.

## Sweep

`probe` measures the specs you name; `sweep` names them for you. It answers "which reseller
should I use for this model this week" in one command: it lists the model's OpenRouter
endpoints, picks candidates by the criteria below, probes them, adds a spec for every native
target that carries the model, and orders the table by the effective price it measured.
Everything after the candidate list is the probe: the same estimate and `--budget`, the same
confirmation, the same `runs/<trace>/<timestamp>/` record and the same two tables.

```sh
provibench sweep sample deepseek/deepseek-v4.1-flash --parallel 4 --budget 3
provibench sweep sample deepseek/deepseek-v4.1-flash --top 5 --sort uptime --zdr --yes
provibench sweep sample deepseek/deepseek-v4.1-flash --yes --json >> sweeps.ndjson
```

Every endpoint becomes a spec pinned to its tag (`@novita`, `@novita/fp8`) — the exact form
`probe` takes by hand, so the two measure the same endpoint the same way; `provibench
endpoints MODEL` prints the tags. `--include` and `--exclude` filter by tag prefix, so
`--include novita` keeps `novita/fp8` too; a tag no endpoint has is an error rather than a
silently thinner run, and so is a filter that leaves nothing to probe.

Native targets join the sweep through `[targets.<name>.aliases]`, which maps an OpenRouter
slug to the name that target serves it under:

```toml
[targets.deepseek.aliases]
"deepseek/deepseek-v4.1-flash" = "deepseek-flash"
```

`--target NAME` picks which `kind = "openrouter"` target is swept (the first one in
`targets.toml` by default). The native specs are the reference the rows are read against;
they are never filtered, cut or pre-checked, because a tag is not how they were selected.

### Which endpoints it picks

The listing is a pre-filter, not a verdict: it says which endpoints were worth paying to
measure and why the others were not, and the effective $/M in the report decides between
them. The criteria are, in order:

1. **Stability floor**, always on: an endpoint with a negative `status`, or under 97 %
   uptime over the last day, is dropped and named in the listing with its reason
   (`dropped: relace/fp4 (status -2), siliconflow/fp8 (uptime 1d 77.90 %)`). The reason
   prints two decimals so a value that rounds up to the floor (`96.96`) cannot look like it
   contradicts the criterion; the candidate table prints two there as well and one
   elsewhere. An endpoint that did not report a status or an uptime is not degraded, it is
   unmeasured, so it stays. `--include TAG` keeps an endpoint past the floor.
2. **`--zdr`** intersects the list with OpenRouter's [Zero Data Retention
   endpoints](https://openrouter.ai/docs/features/zdr); the ones left out are listed as
   `not ZDR`.
3. **`--sort`** ranks what is left: `price` (default) is the listed prompt price ascending,
   ties going to the endpoint with the higher throughput p50 and then the better 1-day
   uptime; `uptime` is 1-day uptime descending, ties to the cheaper endpoint;
   `throughput` is throughput p50 descending; `latency` is latency p50 ascending. The two
   percentile keys need the target's API key — OpenRouter returns those fields only for a
   keyed request — and a candidate the API has no value for is an input error rather than
   an endpoint quietly ranked last.
4. **`--top N`** keeps the N best after the availability check below. Native targets are
   never cut and count outside N, and a candidate that fails the check does not consume a
   slot: the next one in the ranking is probed instead.

### The availability check

`--top` (and `--check` on its own) sends one `max_tokens 1` request per candidate, in
ranking order, **after you confirm the run**: the smallest rung's cold body without a nonce,
so it costs about one warm read each. An endpoint the account's settings exclude answers
that request with a 404 instead of failing three colds later, and the run reports it as
`not probed, unavailable for this key: Paid model training violation (account settings)`
with no summary row; the candidate that would have taken its `--top` slot is the next one
in the ranking. Nothing is sent before the confirmation, so the check is priced in the
estimate you agree to: the line `pre-check: up to 5 requests, 61,000 tokens, $0.0100` is
part of that total, and a refusal says how much of the total the check is when dropping the
check would fit, unless it names the upper bound below instead. The estimate's spec rows are
the endpoints `--top` would probe, so a candidate promoted by a removal is not covered by
that total: a `--top` run adds the line
`upper bound if the N priciest candidates are the ones that answer: $0.0130` under the
check, and `--budget` is compared against that bound — the check can promote any N of the
ranked candidates, and that is what the run can cost.

### Reading a sweep

The run records how it chose (`run.json`'s `sweep` block: the criteria, the ranking it
ranked from, and every endpoint it dropped with the reason), so `report` prints a
`selection: sort=price, top=5, zdr=off; dropped: …` line under its tables with no network —
in the terminal and in the `--html` file.

A sweep is the one run that sorts its tables: by effective $/M ascending, ties to the higher
hit rate, so the first row is the endpoint to use and the rest are the alternatives.

`--parallel N` probes up to N specs at once. Specs do not share a cache — they are different
providers — so nothing leaks between them, and one spec is always sequential because its
requests are a timeline. What does move is the wall clock: with several specs in flight the
prefill and TTFT numbers sit next to other traffic on the same connection pool, which is
worth a rerun at `--parallel 1` before believing a latency difference of a few
milliseconds.

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

## Runs over time

Providers change routing, quantization, cache configuration and prices from week to week,
so the realistic use is a re-run every few days rather than one benchmark. Every `probe`,
`sweep` and `replay` already writes its run directory under `runs/<trace>/<timestamp>/`, so
the run directories are the record and a cron line is the whole setup:

```sh
# every Monday at 09:03, a fresh sweep of the model you actually use
3 9 * * 1 cd ~/bench && provibench sweep sample deepseek/deepseek-v4.1-flash --top 3 --yes --budget 1
```

`cd` matters: the runs dir defaults to `./runs`, so the sweep has to run from the same
directory every time. Nothing else needs keeping — `history` and `compare` read those
directories offline, and the listed price they show is the price of the week that run
happened in.

```sh
provibench history deepseek/deepseek-v4.1-flash             # every run of the model
provibench history deepseek/deepseek-v4.1-flash --since 14d # the last two weeks only
provibench compare previous latest                           # what changed since last week
provibench compare runs/sample/20260912-090301 runs/sample/20260919-090302  # two named directories
```

`history` prints one row per run and provider — date, trace, protocol, spec, hit %, effective
$/M, TTFT, tok/s, errors, and the listed $/M from that run's `prices` block — ordered by
provider and then by date. Under the table one sparkline per spec, trace and protocol draws
its hit rate across the runs it has, oldest on the left, with the run count. `MODEL` is the
model the runs' specs carry, so a native spec is asked about by the name its own endpoint serves
(`deepseek-flash`); with no `MODEL`, every model in the runs dir lists.

```
specs: openrouter:deepseek/deepseek-v4.1-flash@<provider>
date             | trace  | protocol | spec    | hit % | eff $/M | TTFT ms | tok/s | errors | in $/M
---------------- | ------ | -------- | ------- | ----- | ------- | ------- | ----- | ------ | ------
2026-09-12 09:03 | sample | probe    | @novita | 100.0 | 0.030   | 400     | 40.0  | 0      | 0.300
2026-09-19 09:03 | sample | probe    | @novita | 0.0   | 0.240   | 400     | 40.0  | 0      | 0.240
2026-09-12 09:03 | sample | probe    | @relace | 100.0 | 0.030   | 400     | 40.0  | 0      | 0.300
2026-09-19 09:03 | sample | probe    | @relace | 100.0 | 0.030   | 400     | 40.0  | 0      | 0.200

@novita  █▁  2 run(s)
@relace  ██  2 run(s)
```

`compare RUN_A RUN_B` names each run the way `report` does — a directory, a trace name (its
newest run), or `latest`/`previous` — and prints one row per spec both runs measured with
the metric as `A → B` and the change between them: hit rate, effective $/M, TTFT and
tok/s. The listed prices are compared from the two `prices` blocks, a spec only one run
measured is named under the table, and runs of different traces or protocols are refused
(`invalid_input`) unless `--force`, because a full replay's per-turn totals and a probe's
warm-read hit rate are not the same measurement. The delta is `B − A` of the arguments
given, so `compare previous latest` reads forward in time.

Every run records what its endpoints said about themselves at the time (`run.json`'s
`endpoints` block: quantization, context length, uptime 1d and status, per gateway spec).
The `prices` block carries the listed input and cache-read prices and their source, which
is what makes a history row's listed price the price of *that* week and lets `compare` say
that a price moved. A run that recorded no price still lists and compares — its listed price
reads `-`.

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
