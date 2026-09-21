# promptreplay 0.2: first public release

Goal: publish promptreplay on GitHub and PyPI so that anyone running Claude Code or an OpenAI-compatible coding agent can measure, on their own recorded session, which provider actually caches their prompt and what it costs.

Owner: Damian. Orchestration: main session (Claude Code). Builder and explore: `acpc run deepseek-flash-ds` (API-billed, cheap) or a Codex-subscription entry when the slice needs stronger reasoning. Review: Opus via the Agent tool. Roles follow `~/ai/lab/knowledge/build-contract.md`: builders get repo and fixtures only; live runs that need API keys or real traces stay in the main session.

## Starting point (2026-09-14)

- One commit (`70b9cfb`, 2026-09-12), no remote, no LICENSE, no CI, no `status.md`.
- Built on the cli-design Python starter: `core/` is the starter's process interface almost verbatim, six unused starter modules dropped; `bench/` (1 330 lines) and `commands/` (1 080 lines) are the domain.
- 149 tests, ruff and pyright clean.
- One trace (`traces/acpc-cov-deepseek.jsonl`, 48 turns of Claude Code on DeepSeek Flash, native endpoint) and four replay runs across ~12 OpenRouter resellers for DeepSeek V4.1 Flash, V4 Flash 0731 and GLM 5.3 Flash.
- Anthropic Messages wire only; OpenAI chat-completions harnesses cannot be recorded or replayed.
- The existing trace is not publishable: it embeds the operator's global `CLAUDE.md` files, home paths, and `metadata.user_id`.

## Decisions

| Date | Decision | Why |
|---|---|---|
| 2026-09-14 | Keep the starter-based architecture; no rebuild | `core/` matches the starter; alignment with the current CLI standard is a delta check, done in slice 8 |
| 2026-09-14 | OpenAI chat-completions support via `convert` of an Anthropic trace, not a second recorder | One recording, two wire formats; keeps the sample trace single-source. Own converter, LiteLLM as reference only (`docs/research/anthropic-to-openai-converters.md`) |
| 2026-09-14 | Sample trace re-recorded on the public acpc repo from an isolated Claude Code home, plus a `scrub` command | Cheaper and safer than scrubbing 14 MB by hand; `scrub` is also a feature users need to share traces |
| 2026-09-14 | No public results leaderboard | Not worth maintaining contributions; the README carries our own measured table instead |
| 2026-09-14 | Native prices come from LiteLLM's `model_prices_and_context_window.json`; `targets.toml` prices are an override | DeepSeek has no pricing API; LiteLLM is the community-maintained table. Caveat recorded: it carries peak rates only, DeepSeek halves them off-peak |
| 2026-09-14 | Skip more native targets in 0.2 | Reach comes from wire formats, not from more `kind = "anthropic"` entries |
| 2026-09-14 | Default measurement is the probe (a few real turns, cold write + repeated warm reads per size rung), not the full replay | PoC in `docs/research/probe-poc.md`: same ranking as the full replay at ~1/7 of the cost, independent samples instead of 30 correlated turns, a size ladder instead of a curve. Full replay stays as `--full` |
| 2026-09-14 | Report shows cache, cost, prefill latency, generation TTFT and tok/s, reliability and drift by default; TTL behind `--ttl`; decode-KV probe and tool-call agreement move to 0.3 | Everything in the default costs nothing or a fraction of a cent extra; TTL costs wall time; the rest costs output tokens or needs thinking-block round-tripping |
| 2026-09-14 | 0.2 scope: slices 1, 3, 3b, 3c sweep, 4, 5, 5b, 6, 7, 8. Slice 2 (`convert`, chat-completions) moves to 0.3 | 0.2 measures harnesses on the Anthropic wire (Claude Code and what runs through it); a second wire format is its own error surface |
| 2026-09-14 | MIT license, CI from the starter, `requires-python >= 3.12` | A public repo without CI looks abandoned; 3.13-only would cut off most users |
| 2026-09-14 | Report in the terminal and as HTML, both from the same `run.json` | Terminal for daily use, HTML for a screenshot and a link |
| 2026-09-14 | Agent skill lives statically in the repo (`skills/promptreplay/SKILL.md`), linked from the README, no installer | Basics only: what the tool measures, how to record, probe, sweep and read a report; `--help` and `schema` carry the rest |
| 2026-09-14 | Headline models beyond DeepSeek V4.1 Flash decided at sweep time; name decided at the end, before the upload | Both depend on what the sweeps show |
| 2026-09-14 | Slice 3d (`sweep --top/--sort/--zdr` + availability pre-check) is in 0.2; the skill's headline recipe is "measure the N best endpoints" | The OpenRouter endpoint list already carries uptime, status, and (with a key) latency and throughput percentiles; the account's provider exclusions only show up as failed requests, hence the pre-check. An agent using the tool should not have to hand-pick a provider list |

## Slices

Each slice ends with: ruff, ruff format, pyright, pytest green; a review verdict; one commit. Order is the dependency order. Slice 2 stays in the plan for 0.3; 1 and 3a were done in parallel on 2026-09-14.

### 1. `scrub` + clean sample trace

- `promptreplay scrub TRACE OUT [--replace OLD=NEW]...`: rewrites a trace with `metadata` removed, home paths replaced (`/home/<user>` → `/home/user`), a built-in secret pattern list (API keys, bearer tokens, emails outside an allow-list) reported and masked; prints a report of every replacement count. Deterministic, byte-stable except for the replaced spans.
- Recording done 2026-09-14 (`traces/acpc-coverage-raw.jsonl`, 46 turns, 13,7 MB raw, 4,0 MB gzip; see `status.md`). Slice 1 scrubs it, keeps the first 30 main-conversation turns (prompt 20k → 93k tokens, Σ 1,8 M prompt tokens, 7,3 MB raw, 2,2 MB gzip; worst case 0,54 USD per provider at 0,30 USD/M with no cache, ~0,10 USD typical), and packages that as `src/promptreplay/data/sample.jsonl.gz`; `load_trace` accepts `.gz`. Replay cost grows quadratically with turns because every turn resends the whole history, and truncation is lossless since each turn is a self-contained request. The README quickstart uses `probe sample` (~0,17 USD worst case per provider on the default rungs). The full 46-turn trace stays local for our own headline runs.
- `replay sample ...` and `inspect sample` resolve the packaged trace by name.
- Acceptance: scan of the packaged trace finds no operator instruction files, no home paths, no identifiers; `uvx promptreplay inspect sample` works on a fresh install.

### 2. `convert` to chat-completions + OpenAI-kind replay

- `bench/convert.py`: Anthropic body → chat-completions body per the mapping spec in `docs/research/anthropic-to-openai-converters.md`; returns the body plus a loss report (dropped cache markers, thinking blocks, signatures, metadata, context management, `is_error` flags).
- `promptreplay convert TRACE OUT [--reasoning-content]` writes a second trace file with `wire = "chat-completions"` in each entry.
- `targets.toml` gains `kind = "openai"` (and `kind = "openrouter"` learns to send chat-completions when the trace wire is chat-completions); replay parses OpenAI usage (`prompt_tokens_details.cached_tokens`, DeepSeek `prompt_cache_hit_tokens`, OpenRouter `provider`).
- Acceptance: converted sample replays against `openrouter` and `deepseek` (OpenAI wire) with the same turn count; loss report appears in `run.json` and the report table notes.

### 3. Probe protocol: `bench/probe.py`, `probe`, nonce on `replay`

Why: measured 2026-09-14, a full replay resends 1,9 M tokens of the 30-turn sample to check 93 k new bytes, its per-turn hit ratio is 30 correlated samples of "did this request land on a warm replica", and every existing run was warm to an unknown degree (the 12:37 run inherited the 12:32 run's cache: Novita 28 % → 68 %; the first native replay hit the live session's cache from hours earlier). The PoC (`scripts/probe_poc.py`, `docs/research/probe-poc.md`) reproduces the ranking with a few real turns at ~1/7 of the cost.

Protocol, per run spec, sequential:

- Rung = a turn `k` of the main conversation. Cold: send turn `k` once. Warm: send turn `k+1` R times, `--gap` seconds apart (default 1 s, also before the first warm request).
- Nonce `promptreplay-probe:<run hex>:<k>` prepended to the first system block; same across specs, different per rung, so every cold request is truly cold. `--warm` disables it. Nonce and run hex recorded in `run.json`.
- Default rungs: the smallest, a middle and the largest turn of the trace (`--rungs 1,13,30` to override); default repeats 6 on the first rung, 2 on the others (`--repeats`). A cold request that fails skips its rung (reported as skipped, never as a miss). One retry with backoff on 429 and 503.
- Throughput request per rung: turn `k+1` once more, streamed, `max_tokens 256`, `temperature 0`, after the warm reads; gives TTFT, tok/s and the first 16 output tokens as a model fingerprint. `--no-throughput` skips it.
- `--ttl 60,300,900`: after the warm reads on the first rung, re-send turn `k+1` at those offsets. Off by default because it costs wall time.
- `max_tokens: 1` and `stream: false` on cold and warm requests as today; OpenRouter `/generation` enrichment as today (billed cost, `cache_discount`, native token counts, served provider).

Commands:

- `promptreplay probe TRACE SPEC... [--rungs] [--repeats] [--gap] [--ttl] [--no-throughput] [--warm] [--budget USD] [--yes]`: the protocol above; persists under `runs/<trace>/<timestamp>/` with `run.json` (`protocol = "probe"`, nonce, rungs, endpoints snapshot) and one jsonl per spec.
- `replay` keeps the full per-turn protocol (`protocol = "full"`), gains the nonce (`--warm` to disable) and `--limit`; it exists for the per-turn curve and the headline "your own session" story.
- Confirmation before any request shows the worst-case estimate per spec (tokens to be sent × listed input price, no cache), because weak resellers bill close to it; `--budget USD` refuses above it, `--yes` skips the prompt.
- Metrics per spec and rung (computed in `bench/summary.py`): hit rate = hits ÷ served warm attempts; cached fraction = mean over hits of cached ÷ cold prompt, capped at 1; hit sequence; cached on cold (contamination check); cold and warm latency medians; TTFT, tok/s, fingerprint; errors, retries; served provider, response model, prompt tokens vs the native or first spec (tokenization drift); billed vs computed cost when `/generation` is available. Per spec: hit rate over all warm attempts, eff $/M prompt = (1 − h)·input + h·cache_read with h = hit rate × cached fraction, price source.
- Acceptance: `probe` on the raw sample against `deepseek:deepseek-flash` and two OpenRouter resellers reproduces the PoC tables within sampling noise; two consecutive probes give the same per-rung cold `cached = 0`; `--warm` shows contamination; unit tests cover rung selection, nonce placement (byte-identical warm bodies), skip, retry, aggregation and the stream parser on fixtures, no network.

### 3b. `sweep`: probe over every endpoint

- `promptreplay sweep TRACE MODEL [--target openrouter] [--exclude tag]... [--include tag]...` expands to one probe spec per endpoint from `endpoints` (plus native targets that carry the model in `targets.toml`), then behaves like `probe`, including the estimate and `--budget`.
- Acceptance: `sweep sample deepseek/deepseek-v4.1-flash --json` produces one summary per endpoint; a 12-endpoint sweep of the sample stays under 2 USD worst case.

### 3d. `sweep` selection: `--top`, `--sort`, `--zdr`, availability pre-check

- The endpoint list already carries `status`, `uptime_last_{5m,30m,1d}`, `pricing`, `quantization`, and with an API key `latency_last_30m` and `throughput_last_30m` as p50/p75/p90/p99. `GET /api/v1/endpoints/zdr` lists the Zero Data Retention endpoints.
- Selection criteria (also stated in the skill): "best" means cheapest among the stable ones. Stability floor first: `status >= 0` and `uptime_last_1d >= 97 %`, otherwise the endpoint is dropped and named in a note (`--include` brings one back). Then the ranking key from `--sort`: `price` (default, prompt price ascending, ties broken by throughput p50 descending when the key is present, then uptime), `throughput` (p50 descending), `latency` (p50 ascending), `uptime` (1d descending); `throughput` and `latency` need the key and error without it. `--top N` keeps the N best after the availability check, `--zdr` intersects with the ZDR list before ranking. The selection is a pre-filter only; the report still orders rows by the measured effective $/M, so the measurement, not the listing, is the verdict.
- Availability pre-check: one `max_tokens 1` request on the smallest rung per candidate before the cut, so account-level exclusions (ignored providers, ZDR on the account) surface as "unavailable for this key" rows instead of eating a slot in `--top`. The pre-check tokens count in the estimate.
- Acceptance: `sweep sample deepseek/deepseek-v4.1-flash --top 5 --sort uptime` probes exactly five served endpoints; `--zdr` on the sample model yields only endpoints from the ZDR list; the run records the selection (sort, top, zdr, dropped endpoints with the reason) so `report` can show it offline.

### 4. Prices from LiteLLM

- `bench/prices.py`: fetch `model_prices_and_context_window.json` (cached under `$XDG_CACHE_HOME/promptreplay/`, refreshed on `promptreplay prices --update` or when older than 7 days), resolve a native target's model to `Prices`; `targets.toml` `prices` entries override.
- Report shows price source per run (`litellm`, `table`, `openrouter-endpoint`).
- Acceptance: `deepseek:deepseek-flash` replay costs without a `prices` table in `targets.toml`; unit tests use a fixture JSON, no network.

### 5. HTML report

- `report RUN` (terminal and `--json`) for a probe run: one line per provider with hit %, cached %, eff $/M next to the listed price, cold/warm prefill ms on the first rung, TTFT ms, tok/s, errors, drift (served provider, tokenization delta vs native, fingerprint mismatch); below it the per-rung table (prompt size, hit sequence, cold/warm ms, tok/s, TTL results when present). Full-replay runs keep today's per-turn table.
- `report RUN --format html --output-file PATH`: single self-contained file with the same two tables and one chart per run (probe: hit rate per rung per provider; full: cache curve per spec); light and dark; no external assets.
- Acceptance: opens offline; screenshot-friendly at 1200 px wide; the terminal report of the PoC-equivalent run fits 120 columns.

### 5b. Runs over time: `history` and `compare`

Providers change routing, quantization, cache config and prices from week to week, so the realistic use is a re-run every few days, not one benchmark. Runs already persist under `runs/<trace>/<timestamp>/`; this slice makes the time axis visible.

- `probe`/`sweep` store an `endpoints` snapshot (price, uptime, quantization, context length per endpoint at run time) in `run.json`, so price and endpoint drift are part of the record.
- `promptreplay history MODEL [--trace T] [--since 30d]`: one row per run per provider with date, hit %, eff $/M, tok/s, errors, listed price; a sparkline per provider across runs; `--json` for scripts. `promptreplay compare RUN_A RUN_B` prints the delta per provider.
- README shows the cron line (`promptreplay sweep sample MODEL --yes --json >> sweeps.ndjson`); the default probe is already the cheap repeat, no separate `--quick`.
- Acceptance: three runs a day apart render as one history table.

### 6. Release hygiene

- LICENSE (MIT), `CHANGELOG.md`, `status.md` kept current, GitHub Actions CI copied from the starter (lint, types, tests, build, wheel smoke), `requires-python >= 3.12` (verify nothing 3.13-only is used), README rewritten: one-paragraph why, the measured DeepSeek table, quickstart on the sample trace, then reference.
- PyPI name `promptreplay` confirmed free on 2026-09-14; GitHub `DamianPala/promptreplay` free.

### 7. cli-design alignment + agent skill

- Delta check of `core/` and the command specs against the current CLI Design Standard; fix drift.
- `skills/promptreplay/SKILL.md`, static in the repo and linked from the README: what the tool measures, how to record, probe, sweep and read a report, the basics only; `--help` and `schema` carry the rest. No installer.
- The skill carries agent recipes, not only command reference: the headline one is "pick the N best endpoints for a model and measure them" (`sweep TRACE MODEL --top N --sort ... [--zdr] --budget ... --yes --json`, then read the summaries and name the winner with its hit rate, effective $/M and TTFT). The recipe states the selection criteria in one paragraph: stability floor (status, uptime 1d), then the `--sort` key, `price` by default; pick `throughput` for interactive agents, `latency` when TTFT matters, `uptime` when reliability matters; the measured effective $/M in the report is the verdict, the listing only chooses candidates. A second recipe is "re-check one endpoint by hand" with `probe` on a label copied from a sweep row.

### 8. Publish 0.2.0

- Tag, PyPI upload, GitHub repo, post with the measured table.

## Metrics roadmap

The probe measures infrastructure on exact recorded payloads. Cache is the first metric, not the only one; the report and the name must not assume cache-only.

| Metric | Source | Status |
|---|---|---|
| Cache hit rate and cached fraction per size rung, hit sequence (replica warm-up) | probe usage / OpenRouter generation | 0.2 slice 3 (full-replay per-turn curve stays under `--full`) |
| Cost split, billed vs computed, eff $/M prompt weighted by hit rate | prices, generation | 0.2 slice 3 |
| Prefill latency cold vs warm at equal size, medians | wall clock | 0.2 slice 3 |
| Generation TTFT and tok/s, model fingerprint (first 16 tokens at temperature 0) | one streamed request per rung, `max_tokens 256` | 0.2 slice 3 |
| Provider drift: served vs requested, response model, tokenization delta vs native | responses, generation | 0.2 slice 3 |
| Reliability: HTTP errors by code, retries, endpoint uptime | responses, `endpoints` | 0.2 slice 3 and 5 |
| Cache TTL | `--ttl` re-reads on the first rung | 0.2 slice 3, opt-in (wall time) |
| Tool-call agreement: does the replayed turn's first tool call match the recorded assistant turn (name, argument keys) | replay with enough `max_tokens` on sampled turns, compared to the next recorded request's assistant message | 0.3 candidate; detects quantized or badly templated endpoints |
| Decode-KV reuse: does the provider cache the tokens it generated, so turn N+1 hits on prompt N plus output N | generate-and-continue on a few turns: request full output O at turn N, then send prompt N + O + the recorded next message and check cached ≥ len(prompt N + O); prompt coherence is irrelevant, only the byte prefix matters | 0.3 candidate. Replay with `max_tokens: 1` cannot see this: the recorded output is new input at N+1 for every provider, so replay is conservative for providers that do cache decode KV and exact for the rest; the ranking holds because the bytes are identical for all |
| Output quality | out of scope | never, by design |

## Open decisions

- Name: decided at the end of 0.2, before the PyPI upload; analysis and candidates in `docs/research/naming.md`. The rename is its own mechanical slice (env prefix, XDG paths, entry point, package dir) right before slice 8.
- Headline models for the README table beyond DeepSeek V4.1 Flash: decided when the sweeps run.

## Out of scope for 0.2

Slice 2 (`convert` to chat-completions, `kind = "openai"`; moved to 0.3), Responses API (`previous_response_id` chaining), output-quality measurement, more native targets, a public results site.
