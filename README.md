# provibench

Which provider should serve your agent?

provibench answers that by replaying a recorded agent session, byte for byte, against every endpoint that serves the same model.
It compares what each endpoint really charges per prompt token, how much of the prompt its cache serves, and how fast it answers.
A cache miss pays the input price, a hit pays the cache price, so a listed price says little until the cache is measured.
Replaying the same bytes is what makes the numbers comparable: re-running the task instead takes a different trajectory through randomness, quantization, and tool-call errors.

## What a run looks like

`sweep sample-trace deepseek/deepseek-v4.1-flash --top 3` on 2026-09-20, the three cheapest listed OpenRouter endpoints plus native DeepSeek, ranked by measured price:

```text
endpoints: openrouter:deepseek/deepseek-v4.1-flash@<provider>

endpoint                | hit % | 1st hit % | cached % | in $/M | cache $/M | eff $/M | cold ms | warm ms | TTFT ms | tok/s | errors | drift
----------------------- | ----- | --------- | -------- | ------ | --------- | ------- | ------- | ------- | ------- | ----- | ------ | -----
deepseek:deepseek-flash | 100.0 | 100.0     | 99.9     | 0.150  | 0.003     | 0.003   | 1690    | 1439    | 1083    | 353.3 | 0      | -
@relace/fp4             | 80.0  | 100.0     | 99.8     | 0.130  | 0.003     | 0.028   | 4410    | 2116    | 1677    | 86.5  | 0      | -
@deepinfra/fp8          | 90.0  | 66.7      | 90.8     | 0.140  | 0.004     | 0.029   | 1589    | 1292    | 1874    | 54.5  | 0      | -
@morph                  | 80.0  | 66.7      | 99.9     | 0.135  | 0.004     | 0.030   | 4677    | 2984    | 2837    | 21.4  | 0      | -

"cold (nonce)": each request carried a unique marker, so every cache hit in this table was written by this run; none came from earlier traffic.
deepseek:deepseek-flash: ignored the one-token limit on the cache probes and generated 3,362 tokens. That raised this run's cost, not the prices above.

Of the OpenRouter providers, the run took the 3 cheapest by listed price. @parasail/fp8 was skipped because OpenRouter reported it as degraded.
pre-check: 3 requests, $0.0081
spent $0.1712 (worst case $0.4708)
prompt bill for a session like this trace (1.8 M prompt tokens):
  deepseek:deepseek-flash  $0.0056
  @relace/fp4              $0.0510
  @deepinfra/fp8           $0.0525
  @morph                   $0.0547
```

All three resellers list a lower input price than DeepSeek's own API and cost about ten times more per prompt token once the cache is measured.
Two of them miss on the first read after a write, which is the case a fresh session hits.
The run itself cost 17 cents.
This is one day of measurement, not a standing ranking: routing, quantization, and prices move from week to week, which is what `history` is for.

## Quickstart

Install from PyPI:

```sh
uv tool install provibench
# or
pip install provibench
```

From a checkout:

```sh
uv sync
uv run provibench
```

The packaged defaults already configure an OpenRouter target and native DeepSeek.
Export the two keys they name:

```sh
export OPENROUTER_API_KEY=...
export DEEPSEEK_API_KEY=...
```

To add a target or change a price, `provibench config init` copies the packaged `targets.toml` into your config directory; its comments show the shape of a target, a model alias, and a price override.

The packaged trace is called `sample-trace`; it goes wherever a command takes a TRACE:

```sh
provibench inspect sample-trace
provibench sweep sample-trace deepseek/deepseek-v4.1-flash --top 3 --dry-run
provibench sweep sample-trace deepseek/deepseek-v4.1-flash --top 3 --budget 1.2
provibench report latest
provibench report latest --format html --output-file report.html
```

The sample is the first 30 turns of a real Claude Code session, with prompts growing from 20k to 93k tokens and 1.8 M prompt tokens in total.
Most runs use it or a trace someone shared; recording your own session is optional, and `record` is under [Full replay](#full-replay).

`--dry-run` prices the run and sends nothing, so you can read the estimate before deciding on a budget.
Every run prints its worst-case estimate before the first request and refuses to start above `--budget`.
`--budget 1.2` clears this sample's own `--top 3` upper bound with headroom; see `--top` under [Sweep](#sweep) for why a `--top` run needs one.

Python 3.12 or newer is required, on Linux, macOS, or Windows; [uv](https://docs.astral.sh/uv/) is optional when installing with pip.
Every command documents its flags: `provibench COMMAND --help`.

## Reference

### Probe

`probe` measures prompt-cache behaviour on a few recorded turns.
An endpoint is `<target>:<model>[@provider[,provider...]]`, resolved from `targets.toml`; provider pins apply to OpenRouter targets.

1. A rung is a recorded turn `k`; the defaults are the smallest, middle, and largest turns that have a following turn.
2. The cold request sends turn `k` once and writes the provider's prefix cache; each repeat request sends turn `k+1`, whose prompt starts with the same bytes.
3. Each rung gets a nonce in the first system block, so a cold request cannot inherit a cache from an earlier run or another rung; `--warm` drops the nonce and measures the cache as found.
4. One streamed generation per rung measures time to first token, tokens per second, and a short output fingerprint; `--no-throughput` skips it.
5. `--ttl` re-reads the first served rung after the requested offsets; a failed cold write skips its rung, and the run still persists and exits non-zero.

```sh
provibench probe TRACE ENDPOINTS... --rungs 1,13,30 --repeats 6,2,2 --budget 0.5
```

Repeat `ENDPOINTS...` to probe several providers in one run.
The estimate prices every prompt token at the listed input price plus throughput output tokens at the output price, assuming no cache hit, and native prices follow the precedence in [Prices](#prices).

| Column | Meaning |
|---|---|
| `hit %` | Repeat requests that found any part of the prefix, out of the repeats that were served. |
| `1st hit %` | The same fraction over each rung's first repeat only: one write, then one read, with nothing else warm. Equal to `hit %` with `--repeats 1`. |
| `cached %` | On a hit, the share of the prompt served from the provider's cache; 100 % is the whole prompt. |
| `in $/M` | The listed input price used for the calculation: what a cache miss pays. |
| `cache $/M` | The listed price per 1M cached prompt tokens: what a hit pays. |
| `eff $/M` | Hit-weighted prompt price, `(1 - h) * input + h * cache read` with `h = hit % * cached %` pooled over the repeats: what the endpoint charged at the hit rate this run measured. |
| `cold ms` / `warm ms` | First-rung cold and repeat prefill latency; the gap is the cache's latency benefit. |
| `TTFT ms` / `tok/s` | Median time to the first streamed token and output tokens per second across the endpoint's rungs. |
| `errors` | Requests without a usable answer; a failed cold write is not counted as a cache miss, and a 429 among them is counted and noted separately. |
| `drift` | `provider`, `model`, or `tokens±N%` markers versus the reference endpoint; `-` means no drift, and the column is never truncated. |

`1st hit %` is the cold-start case: a fresh session, a restart, a gateway switching providers.
An endpoint far below its own `hit %` there needs a few requests before its cache helps, and bills closer to the first-read rate in the first minutes of a session; the pooled `eff $/M` stands in for the steady state of a long one, in which everything but the last turn has been sent before.
For an unpinned OpenRouter endpoint, `in $/M` is a rate fitted from what the provider that actually served the run billed, not a listed price, and a note under the table says so; pinned and native endpoints keep their listed price.

Under the endpoint table the run prints a per-turn table, one row per endpoint and rung; four rows from the run above:

```text
endpoint                | rung | prompt | cached cold | hits           | cold ms | warm ms | TTFT ms | tok/s | errors
----------------------- | ---- | ------ | ----------- | -------------- | ------- | ------- | ------- | ----- | ------
deepseek:deepseek-flash | 1    | 20185  | 0           | 1 1 1 1 1 1    | 1690    | 1439    | 921     | 353.3 | 0
@relace/fp4             | 1    | 20147  | 0           | 1 1 1 1 1 0    | 4410    | 2116    | 679     | 105.9 | 0
@deepinfra/fp8          | 29   | 91888  | 0           | 0.23 1         | 3801    | 7190    | 4771    | -     | 0
@morph                  | 1    | 20071  | 0           | 0 1 1 1 1 1    | 4677    | 2984    | 2837    | 20.2  | 0
```

`prompt` is the rung's prompt size, `cached cold` what the cold write read back (a non-zero value indicates contamination), and `hits` has one cell per repeat: `1` for 98 % or better, `x` for a failed read, otherwise the cached fraction.
The latency and throughput columns are the endpoint table's, per rung; a `-` under `tok/s` is a turn answered in one burst.
With `--ttl`, one more cell per requested offset follows, for example `60s:1` when the cache was still available.

A run that spends money prints `spent $X (worst case $Y)`: `X` is what it actually cost, `Y` what the estimate priced beforehand assuming no cache hit.
When `--top` ran an availability check, a `pre-check: N requests, $X` line precedes it; those requests persist to `precheck.jsonl` and never enter an endpoint's hit rate, error count, or drift.
The closing block projects what a session shaped like the recorded trace would bill in prompt tokens, one line per endpoint at its measured effective price.
Runs persist under `runs/<trace>/<timestamp>/` as options, endpoint snapshots, prices, and one JSONL file per endpoint, so every table above can be reprinted offline.

`report RUN --output-file PATH` writes exactly what stdout would have shown to PATH, leaves stdout empty, and replaces an existing file.
Without `--format`, `report RUN` prints its JSON document on a non-TTY stdout and the text table on a terminal; `--format text` gives the table regardless, and `--format html` writes a standalone page with the same numbers.

### Sweep

`sweep TRACE MODEL` lists OpenRouter endpoints, selects candidates, adds native targets whose aliases map to `MODEL`, probes the selected endpoints, and orders the measured report by effective price.
Native endpoints are never cut by `--top` and are not filtered by the pre-check.
`provibench endpoints MODEL` prints the tags.

Selection is applied in this order:

1. `--zdr` keeps only OpenRouter's Zero Data Retention endpoints.
2. Drop degraded endpoints and endpoints below 97 % one-day uptime; `--include PREFIX` keeps only endpoints whose tag starts with PREFIX and keeps them past this floor, `--exclude PREFIX` drops them.
   A prefix no endpoint has is an error, not a thinner run, and a missing status or uptime counts as unmeasured rather than degraded.
3. `--sort price|uptime|throughput|latency` ranks the survivors, with price as the default; price ties prefer throughput and then uptime, and the percentile keys require API data.
4. `--top N` keeps the N best that pass the availability check.

That check sends one `max_tokens: 1` request on the smallest rung per candidate, after confirmation and priced inside the estimate you agreed to; an endpoint the account cannot reach is reported as unavailable and its slot goes to the next ranked candidate.
Because any candidate up to the cutoff can be promoted that way, a `--top` run prints an upper bound beside the plain estimate, and `--budget` is compared against the plain estimate rather than that bound.
A `--top 3` sweep of the packaged sample, for example, estimates about 0.62 USD with an upper bound near 1.06 USD, which is why the quickstart passes `--budget 1.2`.
`--parallel N` overlaps endpoints, but each endpoint remains sequential, so rerun with `--parallel 1` before trusting small latency differences.

The run records its selection criteria, ranking, and dropped endpoints, so `report` can show the choice offline.
The measured effective price and cache rate are the verdict; the listing only chooses candidates.

### Runs over time

Providers change routing, quantization, cache configuration, and prices from week to week.
Run the same sweep on a schedule from one fixed directory, or with `PROVIBENCH_RUNS_DIR` set, so every run lands in one runs directory; then read the runs offline:

```sh
provibench history MODEL
provibench compare previous latest
```

`history` reads runs offline and shows, per run and endpoint, the date, trace, protocol, hit rate, effective price, TTFT, throughput, errors, what the endpoint spent, and the listed price that run recorded, plus a hit-rate sparkline.
`spend $` is recomputed from the run's own records, so an older run gets one too, and `-` means the run recorded no price to compute it from.
With no model, `history` lists every model in the runs directory.

`compare RUN_A RUN_B` accepts a run directory, a trace name, `latest`, or `previous`, and prints A to B metrics for endpoints measured in both runs plus listed-price changes.
Endpoints pair by label, so an OpenRouter tag renamed between runs, such as `@novita` then `@novita/fp8`, lands in `only in A` or `only in B` instead of in a delta.
Runs with different traces or protocols are refused unless `--force`, because probe repeat-read rates and full-replay per-turn totals are different measurements.

Each run keeps an endpoint snapshot with the pinned provider, quantization, context length, one-day uptime, and status, along with the prices and their source at measurement time.

### Full replay

`replay` sends every recorded turn; use it for a per-turn cache curve or the cost of a whole session.
It requests one output token by default, discards live output, and stamps a run nonce into turns so the run does not inherit an earlier cache; `--warm` disables the nonce and reads the cache as found.
`--limit` samples a prefix of a trace, `--run` accepts the same endpoint shape as `probe`, and `--dry-run` prices the run and sends nothing.

A trace of your own comes from `record`, a proxy in front of the real API that runs until Ctrl-C and appends to `<traces-dir>/<name>.jsonl`.

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
provibench record --name my-session --upstream https://api.anthropic.com
```

```sh
provibench replay TRACE --run openrouter:deepseek/deepseek-v4.1-flash@novita --budget 2 --yes
```

### Prices

Price precedence for native targets is `targets.toml`, then the cached LiteLLM community table, then no price; OpenRouter endpoint prices come from the endpoint snapshot.
`provibench prices` shows the four prices and their source, and the LiteLLM copy refreshes weekly or with `--update`.
A missing price remains `n/a`, and a budgeted run refuses an unpriced endpoint.

The community table lists peak rates, and a provider can discount off-peak, so the packaged `targets.toml` explicitly prices native `deepseek-flash` at DeepSeek's off-peak rate.

### Sharing a trace

A trace contains request bytes verbatim and can include home paths, instruction files, `metadata.user_id`, and keys in headers or bodies.
Scrub before sharing:

```sh
provibench scrub sample-trace shared.jsonl.gz
provibench scrub sample-trace clean.jsonl --turns 20 --replace acme-corp=example --user NAME
```

`scrub` removes the request's `body.metadata`, rewrites `/home/<name>`, `/Users/<name>`, and encoded Claude Code project paths, and masks API keys, bearer tokens, AWS and GitHub tokens, Slack tokens, and email addresses; `--allow-email` keeps an approved address.
The repeatable `--replace OLD=NEW` handles project names and other literal values the built-in rules cannot know, and `--user NAME` handles whole-word names in owner columns and prose.
Nested metadata inside a request body remains, so read the copy before sending it.

`--turns N` keeps the first N entries of the main conversation, or the first N file entries when conversation keys are absent.
`OUT` is the scrubbed copy, gzip-compressed when its name ends in `.gz`; `--output-file` writes the result summary, and `--force` overwrites an existing copy or summary.

### Configuration

Precedence is flag, environment variable, configuration file, then built-in default, and `provibench config show` prints each effective value and its source.
In the configuration file, a setting's key is its name.
The packaged `targets.toml` is a fallback for a fresh install: until one is written to the default `targets_path`, `show` reports it with source `packaged`, and `provibench config init` copies it there so it can be edited (`--force` overwrites an existing file).

| Setting | Flag | Environment | Default |
|---|---|---|---|
| `config_path` | `--config`, `-c` | `PROVIBENCH_CONFIG` | `~/.config/provibench/config.toml` |
| `targets_path` | `--targets` | `PROVIBENCH_TARGETS` | `~/.config/provibench/targets.toml` |
| `traces_dir` | none | `PROVIBENCH_TRACES_DIR` | `./traces` |
| `runs_dir` | none | `PROVIBENCH_RUNS_DIR` | `./runs` |

`~/.config` follows `$XDG_CONFIG_HOME` when that is set.

### Terminal width

The probe and sweep tables are laid out for 120 columns and run past 140 when the labels are long, as in the example above; `history` is wider again because it adds trace and protocol columns.
At 80 columns rich elides long names, and a label longer than 24 columns is elided in the middle.

### Agents

Agent workflows and recipes, including the `PROVIBENCH_RUNS_DIR` an agent should pin before its first paid run: [`skills/provibench/SKILL.md`](skills/provibench/SKILL.md).
`provibench schema` prints the machine-readable interface the skill relies on.

### Development

The gate is:

```sh
TERM=xterm uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

The test suite includes an architecture seam check.
CI runs the same gate on Python 3.12, 3.13, and 3.14 on Linux and on 3.12 on macOS and Windows, then builds the wheel, checks that it carries the packaged data, and smoke-tests it in an isolated environment.
