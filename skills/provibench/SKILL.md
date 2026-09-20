---
name: provibench
description: Measure prompt-cache hit rate, effective price and latency of one model across OpenRouter endpoints and native APIs by replaying a recorded Claude Code session; pick the best endpoint from data.
---

# provibench

## What it measures

- Replays a recorded agent session against the same model on each target.
- Uses probe rungs, with cold writes and repeated warm reads at several prompt sizes.
- Reports cache hit rate and the cached fraction.
- Calculates effective USD per million prompt tokens, including cache-read pricing.
- Measures time to first token and generated tokens per second when the provider streams.

Prompt caching is keyed on the exact bytes of the prompt, so a synthetic task run does not
reproduce the requests that Claude Code sent. Running the task everywhere creates a different
trajectory for each provider because of model randomness, quantization and tool-call errors;
replay sends the recorded bodies in order, keeps the prefix sequence identical, and discards
the live output after requesting one token.

## Setup

Install from the repository with `uv sync`; the tool then runs as `uv run provibench`.

Create `targets.toml` with one target per API and set each `api_key_env` to the name of an environment variable holding its key.

Before the first paid run, pin where runs are written: set `PROVIBENCH_RUNS_DIR`, or run every
command from the same project directory. A run and its later `report` have to agree on
`runs_dir`, and an agent that changes directory between them will otherwise look in the wrong
place, or the packaged default, for a run it already paid for.

Check the packaged sample trace with `provibench inspect sample`.

For the complete command and output contract, run `provibench schema` or `provibench schema sweep`.

## Recipe 1: pick the N best endpoints for a model and measure them

Read the estimate first, without spending anything:

```sh
provibench sweep TRACE MODEL --top N --sort KEY --dry-run --json
```

Then run it for real:

```sh
provibench sweep TRACE MODEL --top N --sort KEY --budget USD --yes --json
```

The stability floor comes first: an endpoint must not be degraded and must have one-day uptime
at or above the floor printed by `sweep --help`; `--include PREFIX` keeps only endpoints whose
tag starts with PREFIX, and keeps them past the stability floor. Among the survivors, `--sort`
chooses the candidates: `price` is the default, `throughput` fits interactive agents that
stream long answers, `latency` when time to first token matters, and `uptime` when reliability
matters more than cost. Use `--zdr` when the data must not be retained. The listing only
chooses candidates; the availability pre-check drops what the account cannot use, and the
measured effective $/M in the report is the verdict, never the listed price.

The `--dry-run` document has `estimate`, `total_usd`, `pre_check`, `upper_bound`, `runs_dir`, and
`requires_confirmation` instead of a real run's fields; nothing is sent, and `--yes` alongside it
is accepted and ignored.

With `--json`, a real run's document has `run_dir`, `run_hex`, `conversation`, `rungs`,
`summaries`, `partial`, `changed`, and `sweep`. Each `summaries` item includes `label`,
`hit_rate`, `cached_fraction`, `eff_per_m_prompt`, and `ttft_ms`, so name the winner from those
measured values. `partial: true` means a request failed or a rung was skipped; the run still
persisted, and the command still exits non-zero.

Read `report RUN_DIR` afterwards: on a non-TTY stdout (a script or an agent) it prints the run's
JSON document by default, the same shape `sweep`/`probe --json` did; pass `--format text` for
the human table instead. Use `report RUN_DIR --format html --output-file REPORT.html` to create
an offline HTML report.

## Recipe 2: re-check one endpoint by hand

Copy the endpoint label from a sweep summary row and run:

```sh
provibench probe TRACE ENDPOINT --yes --json --rungs 1,13,30 --repeats 6,2,2 --ttl 60,300
```

Use `--warm` when you want to measure the cache as it currently exists, including possible
contamination from earlier runs. Leave it off for an isolated measurement with a fresh nonce.
Adjust `--rungs`, `--repeats`, and `--ttl` when checking a particular prompt size or cache
lifetime; the estimate and `--budget` still apply before requests are sent.

## Recipe 3: watch a provider over time

Run the README cron job on a stable schedule:

```cron
3 9 * * 1 cd ~/bench && provibench sweep sample deepseek/deepseek-v4.1-flash --top 3 --yes --budget 1.2
```

Then inspect the last month and compare the previous run with the latest one:

```sh
provibench history MODEL --since 30d
provibench compare previous latest
```

## Reading a report

The five columns that matter are `hit %`, `cached %`, `eff $/M`, `TTFT`, and `tok/s`.

- Burst delivery makes `tok/s` undefined.
- A partial cache appears as `cached %` below 100.
- A listed price is not the effective price.
- `compare` pairs endpoints by label, so an OpenRouter tag renamed between runs, such as `@novita` one week and `@novita/fp8` the next, appears under `only in A` or `only in B` instead of as a delta.

## Cost and safety

The tool prints an estimate before anything is sent, and `--budget USD` refuses a run above that estimate.
`--dry-run` is the way to read that estimate without deciding to spend: it prices `probe`, `sweep`, or `replay` and stops there, sending nothing and exiting `0`; `--yes` alongside it is accepted and ignored.
Use `--yes` in scripts, not for unattended consent you have not reviewed.
Keep API keys in environment variables whose names are declared by `targets.toml`; never put a key in a command argument.
