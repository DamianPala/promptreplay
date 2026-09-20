# provibench

provibench tests providers of one model against a real recorded agent session.
Most users never record anything themselves: they run the packaged sample, or a trace someone shared, against the endpoints they are choosing between.
Record a real agent session once, replay it byte for byte against many endpoints, and compare prompt-cache hit rate, effective price, latency, and throughput per endpoint.
Replaying preserves the exact prompt bytes and turn sequence; re-running a task creates a different trajectory through randomness, quantization, and tool-call errors, so caching and cost are not comparable.

## What a run looks like

The quickstart's `sweep sample deepseek/deepseek-v4.1-flash --top 3` on 2026-09-20, the three cheapest listed endpoints plus native DeepSeek, ranked by measured price:

```text
endpoints: openrouter:deepseek/deepseek-v4.1-flash@<provider>

endpoint                | hit % | 1st hit % | cached % | in $/M | cache $/M | eff $/M | cold ms | warm ms | TTFT ms | tok/s | errors | drift
----------------------- | ----- | --------- | -------- | ------ | --------- | ------- | ------- | ------- | ------- | ----- | ------ | --------
deepseek:deepseek-flash | 100.0 | 100.0     | 99.9     | 0.150  | 0.003     | 0.003   | 1690    | 1439    | 1083    | 353.3 | 0      | -
@relace/fp4             | 80.0  | 100.0     | 99.8     | 0.130  | 0.003     | 0.028   | 4410    | 2116    | 1677    | 86.5  | 0      | provider
@deepinfra/fp8          | 90.0  | 66.7      | 90.8     | 0.140  | 0.004     | 0.029   | 1589    | 1292    | 1874    | 54.5  | 0      | provider
@morph                  | 80.0  | 66.7      | 99.9     | 0.135  | 0.004     | 0.030   | 4677    | 2984    | 2837    | 21.4  | 0      | -

pre-check: 3 requests, $0.0081
spent $0.1712 (worst case $0.4708)
```

The three resellers list a lower input price than the native API and cost ten times more per prompt token once the cache is measured; two of them also miss on the first read after a write.
The run cost 17 cents.
This is one day's measurement, not a standing ranking: routing, quantization, and prices shift week to week, which is what `history` is for.

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

Write your own `targets.toml` to add a target or change a price; `provibench config show` names the path it reads.

An agent scripting this tool should pin `PROVIBENCH_RUNS_DIR`, or always run from the same project directory, before its first paid run: a run and the `report` that reads it later have to resolve `runs_dir` to the same path, otherwise the report looks for a run it already paid for in the wrong directory.

Run the packaged sample:

```sh
provibench inspect sample
provibench sweep sample deepseek/deepseek-v4.1-flash --top 3 --dry-run
provibench sweep sample deepseek/deepseek-v4.1-flash --top 3 --budget 1.2
provibench report latest
provibench report latest --format html --output-file report.html
```

`--dry-run` prices the run and stops there, sending nothing, so you can read the estimate before deciding on a budget.
The sweep prints its worst-case estimate before anything is sent, and refuses the run when it exceeds `--budget`.
`--budget 1.2` clears this sample's own `--top 3` upper bound with headroom; see `--top` under [Sweep](#sweep) for why a `--top` run needs one.

Every command documents its flags: `provibench COMMAND --help`.

## Requirements

Python 3.12 or newer. [uv](https://docs.astral.sh/uv/) is optional when installing with pip.

## Reference

### Probe

`probe` measures prompt-cache behaviour on a few recorded turns.
An endpoint is `<target>:<model>[@provider[,provider...]]`, resolved from `targets.toml`; provider pins apply to OpenRouter targets.

1. A rung is a recorded turn `k`; the defaults are the smallest, middle, and largest turns that have a following turn.
2. The cold request sends turn `k` once and writes the provider's prefix cache; each warm request sends turn `k+1`, whose prompt starts with the same bytes.
3. Each rung gets a nonce in the first system block, so cold requests are isolated from earlier runs and from other rungs; `--warm` disables the nonce and measures the cache as found.
4. One streamed generation per rung measures time to first token, tokens per second, and a short output fingerprint; `--no-throughput` skips it.
   A buffered answer has no generation window, so `tok/s` is `-` and the notes identify burst delivery.
5. `--ttl` re-reads the first served rung after the requested offsets; a failed cold write skips its rung, and the run still persists and exits non-zero when requests failed or were skipped.

```sh
provibench probe TRACE ENDPOINTS... --rungs 1,13,30 --repeats 6,2,2 --budget 0.5
```

Repeat `ENDPOINTS...` to probe several providers in one run.
The estimate covers every prompt token at the listed input price plus throughput output tokens at the output price, assuming no cache hit, and appears before requests.
`--budget` refuses a run above that estimate.
`--dry-run` prints that same estimate as the result and sends nothing, so you can read it before deciding to spend; `--yes` alongside it is accepted and ignored.
Native prices follow the precedence in [Prices](#prices).

| Column | Meaning |
|---|---|
| `hit %` | Warm reads that found any part of the prefix divided by warm reads that were served. |
| `1st hit %` | The same fraction, counting only each rung's first warm read: one write, then one read, with nothing else warm. That is the cold-start case (a fresh session, a restart, a gateway switching providers). A gap between it and `hit %` means the endpoint needs a few requests before its cache helps, either because it admits a prefix only on its second sight or because several replicas each need their own copy. Equal to `hit %` with `--repeats 1`. |
| `cached %` | On a hit, the share of the prompt served from the provider's cache; 100 % means the whole prompt. A provider that caches a fixed window from the front shows this falling as the conversation grows. |
| `in $/M` | The listed input price used for the calculation; for an unpinned OpenRouter endpoint, a rate fitted from what the provider that actually served it billed, not a listed price, and said so in a `priced as served: <provider> (rates fitted…)` note under the table. It falls back to the worst-case listed price (`worst case: <provider>`) when those records could not be fit. Pinned and native endpoints keep their listed price unchanged. |
| `cache $/M` | The listed price per 1M cached prompt tokens: what a cache hit costs, next to what a miss costs (`in $/M`) and what the endpoint actually charged at its measured hit rate (`eff $/M`). |
| `eff $/M` | Hit-weighted prompt price: `(1 - h) * input + h * cache read`, where `h = hit % * cached %` over every warm read. The pooled reads stand in for a long session's steady state, in which everything but the last turn has been sent before (Venice, for one, caches 16 384 tokens of a prompt it sees once and the whole prompt the second time). An endpoint whose `1st hit %` is far below its `hit %` will bill closer to the `1st hit %` rate in the first minutes of a session. |
| `cold ms` / `warm ms` | First-rung cold and warm prefill latency; the gap is the cache's latency benefit. |
| `TTFT ms` / `tok/s` | Median time to the first streamed token and output tokens per second across the endpoint's rungs. |
| `errors` | Requests without a usable answer; a failed cold write is not counted as a cache miss. A 429 among them is also counted separately (`rate_limited` in `--json`) and noted, so a run's own pace is not confused with a refusal. |
| `drift` | `provider`, `model`, or `tokens±N%` markers versus the reference endpoint; `-` means no drift. Rendered in full, never truncated. An endpoint whose largest rung was skipped shows no `tokens±N%` marker (comparing prompt sizes across two different turns is not tokenizer drift) and gets a `token drift n/a (rung skipped)` note instead. |
| `hits` | Per-warm-read cells: `x` failed, `1` is 98 % or better, otherwise the cached fraction. |
| `TTFT ms` / `tok/s` | The same two measurements for each rung. |
| `ttl` | One cell per requested offset, for example `60s:1` means the cache remained available. |
| `cached cold` | What the cold write read back; a non-zero value indicates contamination. |

In JSON, `h` (what the prose above calls the hit-weighted cached share) is `hit % * cached %` pooled over every warm read, and `first_h` is its first-read analogue; `eff $/M` prices from `h`.
When no rung of an endpoint served a first warm read at all, `eff $/M` falls back to the pooled reads instead, noted as `eff $/M from pooled reads: no first read served`.
When every endpoint shares one target and model, the table moves them into an `endpoints:` caption and shows provider tails; a different model keeps its complete name as the reference row.
Reports also retain the served provider, response models, raw per-rung records, and the price source in JSON.
Runs persist under `runs/<trace>/<timestamp>/` as options, endpoint snapshots, prices, and one JSONL file per endpoint.

A `probe` that spends money prints `spent $X (worst case $Y)` after it runs, and a `report` of the run shows the same line; `X` sums every endpoint's `spend_usd` (OpenRouter's own bill where it reported one, else the usage priced at the listed rates) plus the availability check's own spend, and `Y` is what the estimate priced beforehand, no cache hit.
When `--top` ran an availability check, a `pre-check: N requests, $X` line precedes it; the check's own requests persist to `precheck.jsonl`, never inside an endpoint's own file, and never enter its hit rate, error count, or drift.
A read that returned more output tokens than its `max_tokens: 1` budget (GMICloud and native DeepSeek both do) is noted and billed for those tokens, not silently absorbed into the prompt count.
The summary table is followed by one line projecting what a session shaped like the recorded trace would bill in prompt tokens per endpoint, at each endpoint's measured effective price, for example `prompt bill for a session like this trace (1.8 M prompt tokens): deepseek:deepseek-flash $0.006, @relace/fp4 $0.07`.

Reports: `report RUN --output-file PATH` writes exactly what stdout would have shown to PATH, leaves stdout empty, and replaces an existing file.
Without `--format`, `report RUN` prints its JSON document on a non-TTY stdout (a script or an agent) and the text table on a terminal; pass `--format text` to get the table regardless of where stdout goes.

### Sweep

`sweep TRACE MODEL` lists OpenRouter endpoints, selects candidates, adds native targets whose aliases map to `MODEL`, probes the selected endpoints, and orders the measured report by effective price.
Native endpoints are never cut by `--top` and are not filtered by the pre-check.
`provibench endpoints MODEL` prints the tags.

Selection is applied in this order:

1. `--zdr` keeps only OpenRouter's Zero Data Retention endpoints.
2. Drop degraded endpoints and endpoints below 97 % one-day uptime; `--include PREFIX` keeps only endpoints whose tag starts with PREFIX, and keeps them past this floor; `--exclude PREFIX` drops them.
   A prefix no endpoint has is an error, not a thinner run.
   Missing status or uptime remains unmeasured rather than degraded.
3. `--sort price|uptime|throughput|latency` ranks the survivors, with price as the default; price ties prefer throughput and then uptime, and the percentile keys require API data.
4. `--top N` keeps the N best after availability checks; a failed check does not consume a slot.

The availability check sends one `max_tokens: 1` request on the smallest rung per candidate after confirmation.
Its tokens are included in the estimate, account-level exclusions are reported as unavailable, and nothing is sent before confirmation.
`--top N` keeps the N best-ranked candidates once the check has run, and a candidate the check removes frees its slot for the next ranked one, so any candidate up to the cutoff can end up promoted.
Because of that, the estimate prices the N best as planned, but a `--top` run also prints an upper bound for what the check could promote it to, and `--budget` is compared against the plain estimate rather than that upper bound.
A `--top 3` sweep of the packaged sample, for example, estimates about 0.62 USD but has an upper bound near 1.06 USD, which is why the quickstart passes `--budget 1.2` rather than a tighter number.
`--parallel N` overlaps endpoints, but each endpoint remains sequential, so rerun with `--parallel 1` before trusting small latency differences.

The run records its selection criteria, ranking, and dropped endpoints, so `report` can show the choice offline.
The measured effective price and cache rate are the verdict; the listing only chooses candidates.

### Runs over time

Providers change routing, quantization, cache configuration, and prices from week to week.
A cron job keeps the same working directory so its default `./runs` directory is stable:

```cron
3 9 * * 1 cd ~/bench && provibench sweep sample deepseek/deepseek-v4.1-flash --top 3 --yes --budget 1.2 --json >> sweeps.ndjson
```

```sh
provibench history MODEL
provibench compare previous latest
provibench compare runs/sample/20260912-090301 runs/sample/20260919-090302
```

`history` reads runs offline and shows date, trace, protocol, endpoint, hit rate, effective price, TTFT, throughput, errors, what the endpoint spent (`spend $`, recomputed from the run's own records, so an older run gets one too and `-` means the run recorded no price to compute it from), and the listed price recorded by that run, plus a hit-rate sparkline per endpoint, trace, and protocol.
With no model it lists every model in the runs directory.
In a mixed history containing two models, the elided labels of two long model names can collide in the sparkline column; run `history` per trace or widen the terminal.

`compare RUN_A RUN_B` accepts a run directory, a trace name, `latest`, or `previous`, and prints A → B metrics for endpoints measured in both runs plus listed-price changes.
An endpoint present in only one run is named separately.
Runs with different traces or protocols are refused unless `--force`, because probe warm-read rates and full-replay per-turn totals are different measurements.
Endpoints pair by label, so an OpenRouter tag renamed between runs, such as `@novita` then `@novita/fp8`, lands in `only in A` or `only in B`.

Each run keeps an endpoint snapshot with the pinned provider, quantization, context length, one-day uptime, and status, along with the prices and their source at measurement time.

### Full replay

`replay` sends every recorded turn; use it for a per-turn cache curve or the cost of a whole session.
It requests one output token by default, discards live output, and stamps a run nonce into turns so the run does not inherit an earlier cache; `--warm` disables the nonce and reads the cache as found.
`--limit` samples a prefix of a trace, and `--run` accepts the same endpoint shape as `probe`.
As with `probe`, `--dry-run` prices the run and sends nothing.
A trace of your own comes from `record`, a proxy in front of the real API.

```sh
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787
provibench record --name my-session --upstream https://api.anthropic.com
```

Recording runs until Ctrl-C and appends to `<traces-dir>/<name>.jsonl`.

```sh
provibench replay TRACE --run openrouter:deepseek/deepseek-v4.1-flash@novita --budget 2 --yes
```

### Prices

Price precedence for native targets is `targets.toml`, then the cached LiteLLM community table, then no price.
OpenRouter endpoint prices come from the endpoint snapshot.
`provibench prices` shows the four prices and their source; the LiteLLM copy refreshes weekly or with `--update`.
A missing price remains `n/a`, and a budgeted run refuses an unpriced endpoint.

The community table lists peak rates.
A provider can discount off-peak, so the packaged `targets.toml` explicitly prices native `deepseek-flash` at DeepSeek's off-peak rate.

`cache_write` is the per-token price of writing to the cache, separate from reading it back.
DeepSeek charges its input price for a cache write (0.15); an OpenRouter listing that charges nothing beyond input shows 0.

### Sharing a trace

A trace contains request bytes verbatim and can include home paths, instruction files, `metadata.user_id`, and keys in headers or bodies.
Scrub before sharing:

```sh
provibench scrub sample shared.jsonl.gz
provibench scrub sample clean.jsonl --turns 20 --replace acme-corp=example --user NAME
```

`scrub` removes the request's `body.metadata`, rewrites `/home/<name>`, `/Users/<name>`, and encoded Claude Code project paths, and masks API keys, bearer tokens, AWS and GitHub tokens, Slack tokens, and email addresses.
`--allow-email` keeps an approved address.
`--replace OLD=NEW` handles project names and other literal values the built-in rules cannot know; `--user NAME` handles whole-word names in owner columns and prose.
Both options are repeatable.
Nested metadata inside a request body remains; only the request's own `body.metadata` is removed.

`--turns N` keeps the first N entries of the main conversation, or the first N file entries when conversation keys are absent.
`OUT` is the scrubbed copy, gzip-compressed when its name ends in `.gz`; `--output-file` writes the result summary, and `--force` overwrites an existing copy or summary.
Read the copy before sending it.
The packaged sample is the first 30 turns of a real Claude Code session on DeepSeek V4.1 Flash, with prompts growing from 20k to 93k tokens and 1.8 M prompt tokens in total.

### Configuration

Precedence is flag, environment variable, configuration file, then built-in default.
`provibench config show` prints each effective value and its source.
The packaged `targets.toml` is a fallback for a fresh install: before one is written to the
default `targets_path`, `show` reports it with source `packaged`, and `provibench config init`
copies it there so it can be edited (`--force` to overwrite an existing file).

| Setting | Flag | Environment | Config key | Default |
|---|---|---|---|---|
| `config_path` | `--config`, `-c` | `PROVIBENCH_CONFIG` | none | `$XDG_CONFIG_HOME/provibench/config.toml`, else `~/.config/provibench/config.toml` |
| `targets_path` | `--targets` | `PROVIBENCH_TARGETS` | `targets_path` | `$XDG_CONFIG_HOME/provibench/targets.toml`, else `~/.config/provibench/targets.toml` |
| `traces_dir` | none | `PROVIBENCH_TRACES_DIR` | `traces_dir` | `./traces` |
| `runs_dir` | none | `PROVIBENCH_RUNS_DIR` | `runs_dir` | `./runs` |

### Agents

Agent workflows and recipes: [`skills/provibench/SKILL.md`](skills/provibench/SKILL.md).
`provibench schema` prints the machine-readable interface the skill relies on.

### Development

Run the tests with `uv run pytest`.
The gate is:

```sh
TERM=xterm uv run pytest -q
uv run ruff check .
uv run ruff format --check .
uv run pyright
```

The test suite includes an architecture seam check.
CI runs Python 3.12, 3.13, and 3.14, lint, formatting, type checks, tests, wheel packaging, packaged-data checks, and an isolated wheel smoke test.

## Terminal width

Tables are planned for 120 columns.
`history` runs wider because it includes trace and protocol columns.
At 80 columns, rich elides long names.
A probe/sweep table's label column is planned as wide as its longest label needs, up to 24
columns, rather than as wide as the rest of the table can spare; that keeps the native
reference row readable whole and can push a dense table past 120 columns by up to that
much. A label longer than 24 columns is still elided in the middle.
