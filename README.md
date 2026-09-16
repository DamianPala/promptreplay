# provibench

provibench tests providers of one model against a real recorded agent session.
Most users never record anything themselves: they run the packaged sample, or a trace someone shared, against the endpoints they are choosing between.
Record a real agent session once, replay it byte for byte against many endpoints, and compare prompt-cache hit rate, effective price, latency, and throughput per endpoint.
Replaying preserves the exact prompt bytes and turn sequence; re-running a task creates a different trajectory through randomness, quantization, and tool-call errors, so caching and cost are not comparable.

## Measured result

This 16-endpoint sweep of DeepSeek V4.1 Flash ran on 2026-09-14:

```text
specs: openrouter:deepseek/deepseek-v4.1-flash@<provider>

spec                    | hit % | prefix % | eff $/M | in $/M | cold ms | warm ms | TTFT ms | tok/s | errors | drift
----------------------- | ----- | -------- | ------- | ------ | ------- | ------- | ------- | ----- | ------ | --------
deepseek:deepseek-flash | 100.0 | 99.9     | 0.003   | 0.150  | 2057    | 1674    | 1280    | 305.4 | 0      | -
@relace/fp4             | 100.0 | 99.9     | 0.003   | 0.150  | 2202    | 1375    | 1827    | 238.8 | 0      | provider
@gmicloud/fp8           | 100.0 | 99.9     | 0.006   | 0.300  | 3388    | 3620    | 3066    | 207.9 | 0      | provider
@together               | 88.9  | 99.9     | 0.039   | 0.300  | 8170    | 8450    | 1182    | 90.7  | 1      | -
@alibaba                | 100.0 | 93.7     | 0.047   | 0.300  | 9844    | 4293    | 3094    | 234.1 | 0      | -
@venice/fp8             | 100.0 | 81.7     | 0.075   | 0.375  | 1721    | 993     | 1308    | 418.1 | 0      | provider
@fireworks              | 100.0 | 61.9     | 0.088   | 0.220  | -       | -       | 2995    | 106.3 | 1      | -
@wafer                  | 70.0  | 100.0    | 0.094   | 0.300  | 1373    | 866     | 1195    | 87.4  | 0      | -
@deepinfra/fp8          | 50.0  | 100.0    | 0.103   | 0.200  | 3367    | 3531    | 2212    | 56.6  | 0      | provider
@modal                  | 60.0  | 100.0    | 0.138   | 0.300  | 4551    | 11701   | 923     | 254.7 | 0      | -
@novita/fp8             | 50.0  | 100.0    | 0.153   | 0.300  | 1779    | 2223    | 2219    | 267.0 | 0      | provider
@morph/fp8              | 44.4  | 99.8     | 0.153   | 0.255  | 3067    | 2756    | 2500    | 48.5  | 1      | provider
@parasail/fp8           | 33.3  | 99.7     | 0.202   | 0.300  | 7857    | 1763    | 4300    | 140.2 | 1      | provider
@deepseek               | -     | -        | -       | 0.150  | -       | -       | -       | -     | 3      | -
@io-net/fp8             | -     | -        | -       | 0.285  | -       | -       | -       | -     | 3      | -
@baseten/fp8            | -     | -        | -       | 0.300  | -       | -       | -       | -     | 3      | -
```

Native DeepSeek and `@relace/fp4` cached the whole prefix at `0.003 $/M` effective.
`@venice/fp8` and `@fireworks` cached only a partial prefix.
Three endpoints answered nothing: `@deepseek` is excluded by the account's privacy setting, while `@io-net/fp8` and `@baseten/fp8` were rate-limited or errored.
The run used a private 46-turn recording with rungs 1, 23, and 45, warm reads 6, 2, and 2, and the estimate's worst case was 3,96 USD; actual spend was about 1,5 USD.
Prices are the listed OpenRouter input prices at that time, and `deepseek:deepseek-flash` uses the native API's off-peak rate.

A week later the numbers can differ, which is what `history` is for.

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
export OPENROUTER_GENERAL_BUILDER_API_KEY=...
export DEEPSEEK_API_KEY=...
```

Write your own `targets.toml` to add a target or change a price; `provibench config show` names the path it reads.

An agent scripting this tool should pin `PROVIBENCH_RUNS_DIR`, or always run from the same project directory, before its first paid run: a run and the `report` that reads it later have to resolve `runs_dir` to the same path, or the report looks in the wrong place, or the packaged default, for a run it already paid for.

Run the packaged sample:

```sh
provibench inspect sample
provibench sweep sample deepseek/deepseek-v4.1-flash --top 3 --dry-run
provibench sweep sample deepseek/deepseek-v4.1-flash --top 3 --budget 1.2
provibench report latest
provibench report latest --format html --output-file report.html
```

`--dry-run` prices the run and stops there, sending nothing, so it is the way to read the estimate before deciding on a budget.
The sweep prints its worst-case estimate before anything is sent, and refuses the run when it exceeds `--budget`.
With `--top`, the availability check can promote any of the ranked candidates, so the run also prints an upper bound for the priciest promotions and `--budget` is compared against that; this sample's own worst case is about 0.62 USD and its upper bound about 1.06 USD, so `--budget 0.5` alone would refuse it outright and the quickstart uses `--budget 1.2` to clear the upper bound with headroom.

Every command documents its flags: `provibench COMMAND --help`.

## Requirements

Python 3.12 or newer. [uv](https://docs.astral.sh/uv/) is optional when installing with pip.

## Reference

### Probe

`probe` measures prompt-cache behaviour on a few recorded turns.
A run spec is `<target>:<model>[@provider[,provider...]]`, resolved from `targets.toml`; provider pins apply to OpenRouter targets.

1. A rung is a recorded turn `k`; the defaults are the smallest, middle, and largest turns that have a following turn.
2. The cold request sends turn `k` once and writes the provider's prefix cache; each warm request sends turn `k+1`, whose prompt starts with the same bytes.
3. Each rung gets a nonce in the first system block, so cold requests are isolated from earlier runs and from other rungs; `--warm` disables the nonce and measures the cache as found.
4. One streamed generation per rung measures time to first token, tokens per second, and a short output fingerprint; `--no-throughput` skips it.
   A buffered answer has no generation window, so `tok/s` is `-` and the notes identify burst delivery.
5. `--ttl` re-reads the first served rung after the requested offsets; a failed cold write skips its rung, and the run still persists and exits non-zero when requests failed or were skipped.

```sh
provibench probe TRACE SPECS... --rungs 1,13,30 --repeats 6,2,2 --budget 0.5
```

Repeat `SPECS...` to probe several providers in one run.
The estimate covers every prompt token at the listed input price plus throughput output tokens at the output price, assuming no cache hit, and appears before requests.
`--budget` refuses a run above that estimate.
`--dry-run` prints that same estimate as the result and sends nothing, which is the way to read it before deciding to spend; `--yes` alongside it is accepted and ignored.
Native prices follow the precedence in [Prices](#prices).

| Column | Meaning |
|---|---|
| `hit %` | Warm reads that found any part of the prefix divided by warm reads that were served. |
| `1st hit %` | The same fraction, but counting only each rung's first warm read — the only read an agent loop performs. Pooling every repeat instead (`hit %`) flatters the cache, since reads 2+ re-read what read 1 just wrote. Equal to `hit %` with `--repeats 1`. |
| `prefix %` | On a hit, the average fraction of the cold prompt cached; 100 % means the whole prefix. |
| `eff $/M` | Hit-weighted prompt price: `(1 - h) * input + h * cache read`, where `h = hit % * prefix %`. |
| `in $/M` | The listed input price used for the calculation; for an unpinned OpenRouter spec, a rate fitted from what the provider that actually served it billed, not a listed price, and said so in a `priced as served: <provider> (rates fitted…)` note under the table. It falls back to the worst-case listed price (`worst case: <provider>`) when those records could not be fit. Pinned and native specs keep their listed price unchanged. |
| `cold ms` / `warm ms` | First-rung cold and warm prefill latency; the gap is the cache's latency benefit. |
| `TTFT ms` / `tok/s` | Median time to the first streamed token and output tokens per second across the spec's rungs. |
| `errors` | Requests without a usable answer; a failed cold write is not counted as a cache miss. A 429 among them is also counted separately (`rate_limited` in `--json`) and noted, so a run's own pace is not confused with a refusal. |
| `drift` | `provider`, `model`, or `tokens±N%` markers versus the reference spec; `-` means no drift. Rendered in full, never truncated. A spec whose largest rung was skipped shows no `tokens±N%` marker (comparing prompt sizes across two different turns is not tokenizer drift) and gets a `token drift n/a (rung skipped)` note instead. |
| `hits` | Per-warm-read cells: `x` failed, `1` is 98 % or better, otherwise the cached fraction. |
| `ttft ms` / `tok/s` | The same two measurements for each rung. |
| `ttl` | One cell per requested offset, for example `60s:1` means the cache remained available. |
| `cached cold` | What the cold write read back; a non-zero value indicates contamination. |

When every spec shares one target and model, the table moves them into a `specs:` caption and shows provider tails; a different model keeps its complete name as the reference row.
Reports also retain the served provider, response models, raw per-rung records, and the price source in JSON.
Runs persist under `runs/<trace>/<timestamp>/` as options, endpoint snapshots, prices, and one JSONL file per spec.

A `probe` that spends money prints `spent $X (worst case $Y)` after it runs, and a `report` of the run shows the same line; `X` sums every spec's `spend_usd` (OpenRouter's own bill where it reported one, else the usage priced at the listed rates) plus the availability check's own spend, and `Y` is what the estimate priced beforehand, no cache hit.
When `--top` ran an availability check, a `pre-check: N requests, $X` line precedes it; the check's own requests persist to `precheck.jsonl`, never inside a spec's own file, and never enter its hit rate, error count, or drift.
A read that returned more output tokens than its `max_tokens: 1` budget (GMICloud and native DeepSeek both do) is noted and billed for those tokens, not silently absorbed into the prompt count.
The summary table is followed by one line projecting what a session shaped like the recorded trace would bill in prompt tokens per spec, at each spec's measured effective price — for example `prompt bill for a session like this trace (1.8 M prompt tokens): deepseek:deepseek-flash $0.006, @relace/fp4 $0.07`.

Reports: `report RUN --output-file PATH` writes exactly what stdout would have shown to PATH, leaves stdout empty, and replaces an existing file.
Without `--format`, `report RUN` prints its JSON document on a non-TTY stdout (a script or an agent) and the text table on a terminal; pass `--format text` to get the table regardless of where stdout goes.

### Sweep

`sweep TRACE MODEL` lists OpenRouter endpoints, selects candidates, adds native targets whose aliases map to `MODEL`, probes the selected specs, and orders the measured report by effective price.
Native specs are never cut by `--top` and are not filtered by the endpoint pre-check.
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
`--parallel N` overlaps specs, but each spec remains sequential, so rerun with `--parallel 1` before trusting small latency differences.

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

`history` reads runs offline and shows date, trace, protocol, spec, hit rate, effective price, TTFT, throughput, errors, what the spec spent (`spend $`, recomputed from the run's own records, so an older run gets one too and `-` means the run recorded no price to compute it from), and the listed price recorded by that run, plus a hit-rate sparkline per spec, trace, and protocol.
With no model it lists every model in the runs directory.
In a mixed history containing two models, the elided labels of two long model names can collide in the sparkline column; run `history` per trace or widen the terminal.

`compare RUN_A RUN_B` accepts a run directory, a trace name, `latest`, or `previous`, and prints A → B metrics for specs measured in both runs plus listed-price changes.
A spec present in only one run is named separately.
Runs with different traces or protocols are refused unless `--force`, because probe warm-read rates and full-replay per-turn totals are different measurements.
Specs pair by label, so an OpenRouter tag renamed between runs, such as `@novita` then `@novita/fp8`, lands in `only in A` or `only in B`.

Each run keeps an endpoint snapshot with the pinned provider, quantization, context length, one-day uptime, and status, along with the prices and their source at measurement time.

### Full replay

`replay` sends every recorded turn and is the path for a per-turn cache curve or the cost of a whole session.
It requests one output token by default, discards live output, and stamps a run nonce into turns so the run does not inherit an earlier cache; `--warm` disables the nonce and reads the cache as found.
`--limit` samples a prefix of a trace, and `--run` accepts the same spec shape as `probe`.
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
A missing price remains `n/a`, and a budgeted run refuses an unpriced spec.

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
The packaged sample is the first 30 turns of a real Claude Code session on DeepSeek V4.1 Flash, with prompts growing from 20k to 93k tokens and 1,8 M prompt tokens in total.

### Configuration

Precedence is flag, environment variable, configuration file, then built-in default.
`provibench config show` prints each effective value and its source.
The packaged `targets.toml` is a fallback for a fresh install.

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
