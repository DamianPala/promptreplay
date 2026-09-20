---
name: provibench
description: Measure which providers of one model actually cache an agent's prompt and what a prompt token then costs, by replaying a real agent session (the packaged sample or your own recording) against OpenRouter endpoints and native APIs; pick the endpoint from measured hit rate, effective price and latency.
---

# provibench

provibench replays a few turns of a recorded coding-agent session against several endpoints that serve the same model.
For each endpoint it reports how often the prompt cache hit, what a prompt token cost at that hit rate, time to first token and throughput.
It replays recorded bytes instead of running a task again because a cache is keyed on the exact prompt bytes, and a re-run task takes a different path on every provider.

The tool's words: a rung is a turn of the trace the probe samples; for each rung one uncached request writes the cache and the repeat requests read it.
A cache miss pays the input price, a hit pays the cache price.
On the command line, `--rungs` picks the turns and `--repeats` the number of reads per rung.

## Setup

Install with `uv tool install provibench` or `pip install provibench`; from a checkout, `uv sync`, then `uv run provibench`.

The packaged `targets.toml` already has an OpenRouter target and native DeepSeek; export the keys they name, `OPENROUTER_API_KEY` and `DEEPSEEK_API_KEY`.
`provibench config show` prints the path of the targets file to edit for other targets or prices; each target's `api_key_env` names the environment variable that holds its key, and a key never goes into a command argument.

Before the first paid run, pin where runs are written: set `PROVIBENCH_RUNS_DIR`, or run every command from the same directory.
A run and the later `report` that reads it resolve `runs_dir` independently, so an agent that changes directory between them looks for a run it already paid for in the wrong place.

`provibench inspect sample` shows the packaged trace, 30 turns of a real Claude Code session.
`provibench schema` prints the whole contract, and `provibench schema sweep` the fields of the sweep document.

## Cost and consent

Every paid command prints its worst-case estimate (every prompt token at the input price, no cache hit) before sending anything, then asks for confirmation.
`--dry-run` prints that estimate as the result, sends nothing and exits 0; read it first.
Then run with `--budget USD --yes`: `--budget` refuses a run whose estimate exceeds it, and `--yes` skips only the prompt.
Use `--yes` after you or your user have read the estimate, not as blanket consent.
A `--top N` sweep also prints an upper bound: a candidate the availability check drops frees its slot for the next ranked one, so the run can grow past the plain estimate, and `--budget` is compared with the plain estimate.

## Recipe 1: pick the N best endpoints for a model and measure them

```sh
provibench sweep TRACE MODEL --top N --dry-run --json
provibench sweep TRACE MODEL --top N --budget USD --yes --json
```

`TRACE` is `sample` or a trace name; `MODEL` is the OpenRouter slug, such as `deepseek/deepseek-v4.1-flash`.
The sweep lists the model's OpenRouter endpoints, drops degraded ones and ones below 97 % one-day uptime, ranks the rest by `--sort` (`price` by default, or `uptime`, `throughput`, `latency`; the last two need the OpenRouter key), keeps the N best that pass a one-request availability check, adds the native targets that carry the model, and probes them all.
Native endpoints are never cut by `--top`.
`--include PREFIX` keeps only tags starting with PREFIX and keeps them past the stability floor; `--exclude PREFIX` drops them; `--zdr` keeps only Zero Data Retention endpoints.
`--parallel N` overlaps endpoints; rerun with `--parallel 1` before trusting small latency differences.

The `--dry-run` document has `estimate`, `total_usd`, `pre_check`, `upper_bound`, `runs_dir` and `requires_confirmation`.
A real run's document has `run_dir`, `run_hex`, `summaries`, `sweep`, `spend_usd`, `worst_case_usd`, `partial` and `changed`; `sweep.dropped` names each endpoint the selection or the check removed and why.
`partial: true` means a request failed or a rung was skipped; the run is still written and the command still exits non-zero, so read the document before treating the exit code as a failed run.

## Reading a run

Name the winner from the measured fields of each `summaries` item, never from the listed price.

- `eff_per_m_prompt` (`eff $/M` in the table) is the prompt price at the hit rate this run measured. The report sorts by it, and it is the verdict.
- `session_prompt_usd` (`this trace $` on the HTML page, the "prompt bill for a session like this trace" footer in text) is what a session shaped like the trace would bill in prompt tokens at that price. Quote it to the user; it is the number they can hold against their own bill.
- `hit_rate` (`hit %`) and `cached_fraction` (`cached %`) say how often a repeat found the cache and how much of the prompt it covered. `cached %` below 100 is a partial cache, for example a provider that caches a fixed window from the front of the prompt.
- `first_hit_rate` (`1st hit %`) counts only the first repeat after each cache write. That is the cold-start case: a fresh session, a restart, a gateway switching providers. A gap below `hit_rate` means the endpoint needs a few requests before its cache helps, and a session bills closer to this rate in its first minutes. In JSON, `h` is `hit_rate * cached_fraction` pooled over every repeat and `first_h` the same over first repeats only; `eff_per_m_prompt` is priced from `h`.
- `ttft_ms` (`TTFT ms`) and `gen_tok_s` (`tok/s`) are the median time to first token and output tokens per second. A rung answered in one burst has no `gen_tok_s`; the endpoint's median then comes from the rungs that streamed.
- `errors`, `rate_limited`, `drift` and `notes` hold failed requests, the 429s among them, a served provider or model that differs from the reference row, and the caveats the text report prints under the table.
- `priced_as` says where an endpoint's price came from; an unpinned OpenRouter endpoint is repriced from the provider that actually served it, so its `eff_per_m_prompt` is not a listed price.

Tell the user the winning endpoint, its `eff_per_m_prompt` next to the listed input price, the session bill, and whether `first_hit_rate` sits well below `hit_rate`.
Give the date of the run: routing, quantization and prices shift week to week, so one run is one day's measurement.

`report RUN` re-reads a run offline; `RUN` is a run directory, a trace name or `latest`.
On a non-TTY stdout it prints the JSON document, the same shape the run printed; `--format text` gives the table people read, and `--format html --output-file report.html` writes a page for them.

## Recipe 2: re-check one endpoint

Copy the endpoint from `summaries[].label` (`target:model@provider`, for example `openrouter:deepseek/deepseek-v4.1-flash@relace/fp4`), then:

```sh
provibench probe TRACE ENDPOINT --dry-run --json
provibench probe TRACE ENDPOINT --budget USD --yes --json
```

The defaults probe the smallest, middle and largest turn with `--repeats 6,2,2` reads.
Pass `--rungs` to check a particular prompt size, `--repeats` for more reads, and `--ttl 60,300` to re-read the first rung's cache after that many seconds.
Each rung carries a fresh nonce, so nothing from earlier traffic counts as a hit; `--warm` drops the nonce and measures the cache as it exists now.
Several endpoints in one command are probed in one run.

## Recipe 3: watch a provider over time

Run the same sweep on a schedule from a fixed directory, so every run lands in one `runs_dir`:

```cron
3 9 * * 1 cd ~/bench && provibench sweep sample deepseek/deepseek-v4.1-flash --top 3 --yes --budget 1.2 --json >> sweeps.ndjson
```

Then read the runs offline:

```sh
provibench history MODEL --since 30d
provibench compare previous latest
```

`history` lists one row per run and endpoint with `hit_rate`, `eff_per_m_prompt`, `ttft_ms`, `gen_tok_s`, `errors`, `spend_usd` and the listed prices the run recorded; `--trace` narrows it to one trace.
`compare` pairs the endpoints both runs measured and gives each metric as `a`, `b` and `delta` (B minus A); `previous` and `latest` are the two newest runs of the trace the newest run belongs to.
Endpoints pair by label, so a tag renamed between runs, `@novita` one week and `@novita/fp8` the next, lands in `only_in_a` or `only_in_b` instead of a delta.
Runs of different traces or protocols are refused without `--force`.
