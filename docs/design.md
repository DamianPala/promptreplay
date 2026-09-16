# provibench design

Record real agent-harness API traffic once, replay it byte-for-byte against LLM providers, compare cache behaviour and cost.

## Why replay, not "run the task on every provider"

Prompt caching is keyed on the exact bytes of the prompt.
Which bytes a harness (Claude Code, Codex) sends per turn decides how much of the prompt a provider can match, and synthetic conversations do not reproduce that.
Running the task live on each provider produces a different trajectory every time (model randomness, quantization, tool-call errors), so costs are not comparable without many repetitions.

Replay fixes both: every recorded request N+1 already contains the recorded assistant turn N inside its message history.
Sending the recorded bodies in order reproduces the exact prefix sequence of the original session; the live output is discarded (`max_tokens: 1`), so a run costs only prompt tokens.

Measured per turn: which provider answered, prompt tokens, cached tokens, cache-write tokens, output tokens, latency, and the cost split into input / cache read / cache write / output.
Aggregated per run: cache hit ratio, the per-turn cache curve, cost totals, effective USD per 1M prompt tokens, provider drift, error count, latency percentiles.

## Layout

```
src/provibench/
  app.py, core/, commands/      CLI Design Standard scaffold (from cli-design/build/starter)
  bench/                        domain library, no CLI imports
    trace.py                    trace file format, conversation grouping
    sse.py                      Anthropic Messages response / SSE parsing
    proxy.py                    recording reverse proxy (starlette + uvicorn + httpx)
    targets.py                  targets.toml, RunSpec parsing
    openrouter.py               /generation and /models/{m}/endpoints clients
    pricing.py                  CostBreakdown
    replay.py                   replay engine, run persistence
    summary.py                  RunSummary aggregation, sparkline
    scrub.py                    shareability rules, the walker, the report model
  data/targets.toml             packaged default targets
  data/sample.jsonl.gz          packaged example trace (`TRACE` = sample)
traces/                         recorded traces (gitignored: they contain repo content)
runs/                           replay results (gitignored)
```

`bench/` depends on httpx, pydantic, starlette, uvicorn.
It never imports `core/` or `commands/`.

## Trace format

`traces/<name>.jsonl`, one `TraceEntry` per line (see `bench/trace.py`):

```
seq, ts, path, query, headers{anthropic-version, anthropic-beta}, body, conversation, response{status, latency_ms, ttft_ms, message_id, model, provider, stop_reason, usage, error}
```

`body` is the request body exactly as the harness sent it.
`conversation` is a 12-hex key derived from the first user message: a harness interleaves background calls (title generation, summarisation) with the main loop; `main_conversation()` picks the group with the most request bytes.

`usage` follows Anthropic semantics: `input_tokens` excludes `cache_read_input_tokens` and `cache_creation_input_tokens`; `prompt_total` is the sum of the three.

A trace reads fine gzipped (`.gz` is decompressed transparently); recording always writes plain jsonl. The literal TRACE name `sample` resolves to the packaged `data/sample.jsonl.gz`, next to `targets.toml`.

## Recording

`provibench record --name NAME --upstream https://api.deepseek.com` starts the proxy on `127.0.0.1:8787`.
The harness is pointed at it (`ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic`); the proxy forwards path, query and headers unchanged, streams the response back, and appends a `TraceEntry` for every `POST */v1/messages` once the stream ends.
`accept-encoding` is dropped on the way up so the teed bytes are plain SSE.

Recording against the native API is cheaper than against a reseller and produces an identical body: the harness does not know where it is sending.
Only the `model` field differs, and replay overrides it.

## Targets and run specs

`targets.toml` (default path `$XDG_CONFIG_HOME/provibench/targets.toml`, packaged fallback in `data/`):

```toml
[targets.openrouter]
url = "https://openrouter.ai/api/v1/messages"
api_key_env = "OPENROUTER_GENERAL_BUILDER_API_KEY"
kind = "openrouter"            # enables provider pinning and /generation billing lookup

[targets.deepseek]
url = "https://api.deepseek.com/anthropic/v1/messages"
api_key_env = "DEEPSEEK_API_KEY"
kind = "anthropic"
litellm_provider = "deepseek"   # the namespace used by LiteLLM's price table

[targets.deepseek.aliases]
"deepseek/deepseek-v4.1-flash" = "deepseek-flash"

# USD per 1M tokens; used for kinds without a billing API. Off-peak DeepSeek (2026-09).
[targets.deepseek.prices."deepseek-flash"]
input = 0.15
cache_read = 0.003
cache_write = 0.15
output = 0.60
```

`aliases` maps an OpenRouter slug to the model this target serves it under, and is how `sweep`
knows to include a native endpoint: `provibench sweep TRACE deepseek/deepseek-v4.1-flash`
adds a spec for the native endpoint of every `kind = "anthropic"` target that aliases the
slug, and nothing for the targets that do not. A native target without an entry is still
reachable by naming it in a run spec (`deepseek:deepseek-flash`); the alias is what makes
`sweep` expand to it.

A run spec on the command line is `<target>:<model>[@<provider>[,<provider>...]]`:

```
openrouter:deepseek/deepseek-v4.1-flash@novita
openrouter:deepseek/deepseek-v4.1-flash@siliconflow
openrouter:deepseek/deepseek-v4-flash-0731@openinference
deepseek:deepseek-flash
```

`@providers` is valid only for `kind = "openrouter"` and becomes `body.provider = {"only": [...], "allow_fallbacks": false}`.
Provider identifiers are OpenRouter endpoint `tag`s (lowercase slugs); `provibench endpoints <model>` lists them.
`label` is the spec string; `slug` is the label with anything outside `[A-Za-z0-9._-]` replaced by `-`.

## Replay

For each spec (concurrently across specs, strictly sequential within one):

1. `prepare_body`: deep-copy the recorded body; set `model`, `stream: false`, `max_tokens` (default 1); add `provider` when pinned; with `--strip-thinking` remove `thinking`/`redacted_thinking` blocks from assistant messages (a message left with no blocks gets `[{"type": "text", "text": " "}]`).
   Everything else, including `cache_control` markers, `metadata`, `tools` and `system`, stays byte-identical.
2. POST to `target.url` with `x-api-key` and `Authorization: Bearer` (both, the two API families differ), `anthropic-version` from the trace (default `2023-06-01`), `anthropic-beta` if recorded, timeout 300 s.
3. If the response is HTTP 400 and the message mentions `max_tokens`, `budget` or `thinking`, retry once with the `thinking` request parameter removed and note it (`thinking param dropped`). The `thinking` parameter is not part of the prompt; removing it does not change the cached prefix on non-Anthropic backends.
4. Parse with `sse.parse_json_message`; record `ReplayResult`.
5. Optional `--delay` seconds between turns (a real agent turn takes 10-60 s; back-to-back replay is a best case for short cache TTLs).

After the loop, for `kind = "openrouter"`:

- fetch `GET /api/v1/models/{model}/endpoints` once; map `normalize(provider_name)` and `normalize(tag)` to that endpoint's `Prices`;
- fetch `GET /api/v1/generation?id={message_id}` for every successful turn (bounded concurrency 4, retry on 404 with backoff: indexing lags a few seconds);
- take `prompt_total = native_tokens_prompt`, `cached = native_tokens_cached`, `provider = provider_name`, `billed_total = total_cost`, `cache_discount`;
- compute the breakdown from endpoint prices: `input = prompt_total - cached`, `cache_read = cached`, `cache_write = usage.cache_creation_input_tokens`, `output = native_tokens_completion`; `source = "openrouter-endpoint"`.
  `billed_total` is what OpenRouter charged; the breakdown is our reconstruction, shown next to it.

For `kind = "anthropic"`: `prompt_total` and buckets from `usage`; native price resolution is
`targets.toml` first, then LiteLLM through `litellm_provider`, then none. The selected source is
recorded per spec (`targets`, `litellm`, or no price), so a run does not silently inherit a later
price-table change.

`ReplayResult` fields: `seq, turn, status, latency_ms, message_id, model, provider, requested_providers, usage, prompt_total, cached, cache_write, output_tokens, cost, billed_total, cache_discount, error, note, generation`.

Persistence: `runs/<trace-name>/<UTC yyyymmdd-HHMMSS>/run.json` (trace name, conversation key, options, list of `{label, slug, target, model, providers, file}`) plus one `<slug>.jsonl` per spec with a `ReplayResult` per line.

## Scrubbing

`scrub` writes a shareable copy of a trace (`bench/scrub.py`, pure; `commands/scrub.py`, I/O).
`bench/scrub.py` holds the rule table (`SECRET_RULES`), the recursive walker over entry strings, the `--turns` selection, and the report model; the rules are applied in a fixed order — user paths, secrets, `--user` names, then literal `--replace` pairs — so the same input and options give byte-identical output.

| removed | how |
|---|---|
| `body.metadata` | dropped from every entry (`metadata.user_id` is not needed on replay) |
| `/home/<name>`, `/Users/<name>` | rewritten to `/home/user`; the names are reported, not written |
| `-home-<name>-…` | rewritten to `-home-user-…`: the project path Claude Code encodes into its state directory |
| keys, `Bearer` tokens, e-mail addresses | `[scrubbed:<kind>]`, per the table in `bench/scrub.py` |
| whole-word `--user NAME` | replaced with `user`, for `ls -l` owners and prose |

`collect_user_names` walks the trace once before scrubbing, so the encoded form is rewritten whatever the order of the strings holding it. `--replace OLD=NEW` is literal and repeatable, for project names the rules cannot know; `--user NAME` is for a person's name, which is not automatic because a name can be an ordinary word. `--allow-email` keeps an address. Two dict keys that scrub to the same string are an `invalid_input` naming the entry's `seq`, rather than a silent loss. `--turns N` keeps the first N entries of the main conversation, or of the file when the trace carries no conversation keys; the report's `selection` says which. An OUT ending in `.gz` is written with `gzip.compress(mtime=0)`, so repeated runs produce identical bytes. An existing OUT needs `--force` (`precondition_failed`); an unparsable input line is an `invalid_input` naming the line. `--output-file PATH` writes the result summary instead of stdout.

## Summary

`RunSummary` per spec:

| field | definition |
|---|---|
| `turns`, `ok`, `errors` | turns replayed, HTTP 200s, failures |
| `providers_seen` | provider name → count |
| `drift` | pinned: turns answered by a provider outside the requested set; unpinned: number of provider changes between consecutive successful turns |
| `prompt_total`, `cached_total`, `cache_write_total`, `output_total` | token sums |
| `hit_ratio` | Σ cached / Σ prompt_total over turns ≥ 2 (turn 1 cannot hit) |
| `curve` | per-turn cached / prompt_total, rendered as a sparkline `▁▂▃▄▅▆▇█` |
| `cost` | summed `CostBreakdown`, or `None` |
| `billed_total` | Σ billed, or `None` |
| `effective_per_m_prompt` | (billed_total if available else cost.total) / prompt_total × 1e6 |
| `latency_p50_ms`, `latency_p95_ms` | with `max_tokens: 1` this is prefill time |
| `notes` | distinct notes across turns |

## Commands

| command | effects | output |
|---|---|---|
| `compare RUN_A RUN_B [--trace T] [--force]` | read-only | per-spec deltas between two runs of one trace; `invalid_input` across traces or protocols |
| `completion [SHELL] [--install] [--force]` | idempotent | completion script, or installed path and `changed` |
| `config show` | read-only | every setting with its effective value and source |
| `endpoints MODEL [--sort KEY]` | read-only | OpenRouter endpoints: tag, provider, quantization, context, prices, uptime, latency, throughput |
| `history [MODEL] [--trace T] [--since DURATION]` | read-only | one row per run and spec, plus a hit-rate sparkline per spec |
| `inspect TRACE [--conversation KEY]` | read-only | conversation list plus per-turn table of the selected conversation |
| `prices [MODEL...] [--update]` | idempotent | native model prices and their source |
| `probe TRACE SPEC... [--rungs] [--repeats] [--gap] [--ttl] [--warm] [--timeout] [--budget] [--yes] [--dry-run]` | non-idempotent, spends API credit, `confirm=True` | `{run_dir, run_hex, conversation, rungs, summaries, partial, changed}` on a real run; `--dry-run` returns `{estimate, total_usd, pre_check, upper_bound, runs_dir, requires_confirmation, partial: false, changed: false}` instead |
| `record --name N --upstream URL [--host] [--port] [--append] [--timeout DURATION]` | non-idempotent, runs until SIGINT or the timeout | `{trace, requests, conversations, changed}` |
| `replay TRACE --run SPEC... [--conversation] [--max-tokens] [--delay] [--strip-thinking] [--limit] [--warm] [--budget] [--yes] [--dry-run]` | non-idempotent, spends API credit, `confirm=True` | `{run_dir, conversation, turns, summaries, partial, changed}` on a real run; `--dry-run` returns `{estimate, total_usd, runs_dir, requires_confirmation, partial: false, changed: false}` instead |
| `report RUN [--format text\|md\|html\|json] [--output-file PATH]` | idempotent | summaries table and cache curves; `--output-file` writes the selected rendering, replacing the file (a report is derived from the run alone) |
| `scrub TRACE OUT [--output-file PATH] [--replace OLD=NEW]... [--user NAME]... [--turns N] [--allow-email ADDR]... [--force]` | idempotent | `{trace, out, output_file, entries_in/out/dropped, selection, bytes_in/out, user_names, rules, changed}` and the rules table |
| `sweep TRACE MODEL [--top N] [--sort KEY] [--zdr] [--budget USD] [--yes] [--dry-run]` | non-idempotent, spends API credit, `confirm=True` | probe summaries plus the recorded selection criteria, `partial` included; `--dry-run` returns the same shape as `probe --dry-run` |

A partial result (a failed request, a skipped rung, or a failed replay turn) still reaches stdout with `partial: true`; the call then exits non-zero with an `operation_failed` error on stderr whose `context` carries only `run_dir` and `run_hex`, not the document again. `--dry-run` prices the run and stops before the confirmation: nothing is sent, `changed` is `false`, and `requires_confirmation` reports whether the same call without `--dry-run` and without `--yes` would be gated in a non-interactive context, which for these three commands is always true because they always reach a request that spends credit; it does not describe whether this particular call could have prompted. `--yes` is accepted and ignored alongside it.

Settings: `targets_path` (`--targets`, `PROVIBENCH_TARGETS`), `traces_dir` (`PROVIBENCH_TRACES_DIR`, default `./traces`), `runs_dir` (`PROVIBENCH_RUNS_DIR`, default `./runs`). A non-empty `NO_INPUT` disables prompts.

## Known limits (v1)

- Replay measures infrastructure on the exact payload: cache depth, input-side cost, drift, prefill latency. It does not measure output quality, generation throughput, or how a provider's tool-call formatting affects the harness.
- A trace recorded on model A replayed on model B (with `--strip-thinking`) measures a slightly different payload; the report carries the note.
- OpenRouter's `native_tokens_prompt` is assumed to include cached tokens; a turn where `cached > prompt_total` gets a note instead of a negative input count.
