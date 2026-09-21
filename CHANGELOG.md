# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] - 2026-09-20

The first published version. Everything below is relative to the unpublished `0.1.0`
scaffold: a proxy recorder, a byte-for-byte full replay and a per-turn table.

### Added

- `probe`: a size-rung cache probe as the default measurement — one cold write per rung
  carrying a nonce, repeated warm reads, and hit rate, prefix fraction, hit sequence, cold
  and warm latency and hit-weighted effective price per rung
- `--warm` on `probe` and `replay`: send no nonce and measure the cache as it already is
- Worst-case cost estimate printed before anything is sent, and `--budget` to refuse a run
  above it, on `probe`, `replay` and `sweep`
- `probe --ttl OFFSETS`: re-read the first served rung's cache at the given offsets and
  record when the read actually went out
- Streamed generation on each rung: time to first token, tokens per second and the model's
  first output tokens as a fingerprint; `--no-throughput` skips it
- `drift` column: served provider, response model and prompt tokenization delta against the
  reference endpoint
- `sweep TRACE MODEL`: list a model's OpenRouter endpoints, pin one probe endpoint to every
  provider tag, add an endpoint for each native target that carries the model, and probe them in
  one run ordered by the measured effective price
- `sweep` endpoint selection: a stability floor on endpoint status and one-day uptime,
  `--include`/`--exclude` by tag prefix, `--zdr`, `--sort price|uptime|throughput|latency`
  and `--top N`
- `sweep --min-uptime PERCENT` sets the one-day uptime floor an endpoint must clear (97 by
  default); the run records it as `sweep.uptime_floor` and the selection sentence names it
- `sweep --quantization VALUES` keeps only endpoints served at one of the named quantizations
  (`unknown` for endpoints the listing does not label); the candidates table gains a `quant`
  column and the run records the filter as `sweep.quantization`
- `sweep --pin TAG` always probes an endpoint alongside the `--top` ranking, past the
  stability floor and outside the ranking's own cut; the run records it as `sweep.pinned`
- `sweep` availability pre-check: one `max_tokens 1` request per ranked candidate, so an
  endpoint the account's settings exclude is named with the gateway's reason instead of
  failing three cold requests
- `sweep --parallel N` to probe several endpoints at once, recorded in the run's notes because
  it moves the latency numbers
- `report --format html --output-file PATH`: one self-contained page with the same tables, hit-rate bars per
  rung, effective prompt price bars and the full-replay cache curve, in dark mode, with no
  scripts and no external resources
- The HTML page is written for a person who opens it once: a title naming the model and the
  endpoint count, one sentence of method, the endpoint table captioned cheapest first with
  the prices in the order `in $/M`, `cache $/M`, `eff $/M`, `this trace $` under a price
  band, every header explained on hover, the `eff $/M` column drawn as one bar per endpoint
  right under the table, then a chart of the cached share per turn, the per-turn table
  folded away, and the caveats as
  sentences in a fixed order with drop reasons in words and the cache mode stated once;
  chart hovers and captions use the page's own words (turn, repeat request, the short
  `@tag` label), never the tool's (`rung`, `warm reads`, `h`)
- The text report separates its sections with blank lines, prefixes notes with the same short
  endpoint labels its tables use, states the cache mode once per run, and prints the session
  bill as one aligned line per endpoint
- `scrub`: write a shareable copy of a trace with home paths, the encoded Claude Code
  project directories, secrets and e-mail addresses removed, `--user` and `--replace` for
  what the patterns cannot know, and `--turns N` to keep a cheaper prefix
- Packaged sample trace — the first 30 turns of a real Claude Code session on DeepSeek V4.1
  Flash — accepted as `sample` by `inspect`, `replay` and `scrub`
- Traces load from gzipped files, and `scrub TRACE OUT` writes them
- Runs persist their protocol, nonce, prices and an endpoint snapshot, so `report`
  re-renders a run with no network, in the terminal, markdown or HTML
- The endpoint snapshot is per endpoint and dated: every probe and sweep records the listed
  input and cache-read price, quantization, context length, uptime 1d and status of the
  endpoint it pinned, so price and endpoint drift are part of the record
- `history [MODEL]`: every run of a model in the runs dir as one table — date, hit %,
  effective $/M, TTFT, tok/s, errors and the listed price of that week — with `--trace` and
  `--since`, a hit-rate sparkline per endpoint, and `--json` rows and series
- `compare RUN_A RUN_B`: the per-endpoint delta between two runs of one trace, with the listed
  price change taken from the two endpoint snapshots, `latest`/`previous` naming, and an
  `invalid_input` refusal across traces or protocols unless `--force`
- `--force` for scrub's OUT or `--output-file`, which are otherwise refused when the target
  exists; `report --output-file` replaces its file, since a report is derived from the run alone
- Native prices from LiteLLM's community table, cached under `$XDG_CACHE_HOME` and refetched
  weekly or on `provibench prices --update`; `targets.toml` `prices` entries override it, and
  every estimate names its price source (`targets`, `litellm` or `openrouter-endpoint`)
- `--dry-run` on `probe`, `sweep`, and `replay`: prices the run and stops there, sending
  nothing; the result carries the estimate, the runs directory, and `requires_confirmation`,
  and `--yes` alongside it is accepted and ignored
- `1st hit %` column: hit rate pooled over only each rung's first warm read, the only one an
  agent loop performs, next to the pooled `hit %` that flatters it
- Every probe and sweep persists the swept model's listed endpoint prices as `listing_prices`
  in `run.json`; an unpinned OpenRouter endpoint is repriced by what actually served it, from that
  listing where the run has one (one provider takes its listed rate as is, several combine by
  a prompt-token-weighted mean), or from rates fitted from the provider's own billed records
  on a run written before this field existed, rejected as noise past ten times the run's own
  worst-case listed rate; `priced_as` says which, on the summary and as a note under the
  table, and `listed_input`/`listed_cache_read`/`listed_source` carry the price `history` and
  `compare` treat as this run's own, never a fit
- `spend_usd` per endpoint and per run: OpenRouter's own bill where it reported one, else the
  usage priced at the listed rates; `probe` and `report` print `spent $X (worst case $Y)` as
  the closing line of the rendered tables, and `history` gets a `spend $` column
- `output_tokens` per endpoint, and a note when a read returned more than its `max_tokens: 1`
  budget (GMICloud and native DeepSeek both do)
- The availability pre-check's own requests persist to `precheck.jsonl`, kept out of every
  endpoint's own records; `probe`/`report` print `pre-check: N requests, $X`
- `rate_limited` next to `errors`: how many of an endpoint's requests came back HTTP 429, with a
  note when non-zero
- A session-projection footer line under the summary table: what a session shaped like the
  recorded trace would bill in prompt tokens, per endpoint, at its measured effective price
- `config init [--force]`: copies the packaged `targets.toml` to its effective user path, so
  `config show`'s `packaged` source has a file the user can actually edit
- A caveat when a cold write reports cached tokens despite the nonce making it a new prompt:
  `N of M cold writes reported cached tokens (the largest ... of ...)`, printed under the
  table and listed in the HTML page's Caveats; silent under `--warm`, where a cold write
  hitting the cache is the thing being measured

### Changed

- Every user-facing "spec" is now "endpoint": `probe`'s positional argument, table captions
  and columns, JSON fields (`summaries[].rungs[].spec` is now `endpoint`, and likewise across
  probe/sweep/replay/report/history/compare), help text and error messages. The persisted
  `run.json` `specs` key, per-endpoint JSONL `spec_label` field, and internal `RunSpec` name
  are unchanged
- A `--json` document written to a terminal is pretty-printed with two-space indent; off a
  terminal, and every NDJSON stream record everywhere, it stays the same compact single line
- `config show` names the packaged `targets.toml` with source `packaged` when no user file
  exists yet, instead of a default path nothing is at
- `inspect`'s text view drops `first_seq`/`last_seq` for a `prompt tokens` `first → last`
  column, and drops the `provider` column when no turn in the conversation carries one
  (`--json` is unchanged in both cases)
- Contract change: `prefix %` is now `cached %` everywhere it is shown (headers, captions,
  notes, HTML tooltips, docs), and the JSON fields it comes from are renamed to match --
  `prefix_fraction` to `cached_fraction`, `first_prefix_fraction` to `first_cached_fraction`
  (`ProbeSummary`/`RungSummary` and the probe/sweep/report/history/compare output schemas);
  persisted run files are unaffected
- `eff $/M` stays priced from the pooled reads (`h`), with `1st hit %` beside it as the
  cold-start bound: a long agent session runs in the steady state the later reads sample
  (a prefix admitted to the cache on its second sight, several replicas warm), while the
  first read after one write sees a single warm replica and a once-seen prefix. A probe
  that replays a few preceding turns before the measured read is the 0.3 follow-up

- `report --json`'s summaries array is named `summaries` for both a probe and a full-replay
  run, the name `sweep` and `probe --json` already used; the separate, always-empty
  `probe_summaries` field is gone
- `history MODEL` resolves a native target's runs through `targets.toml` aliases, the same
  way `sweep` matches a native target to `MODEL`
- The native price source `table` is renamed `targets` everywhere (`prices`, estimates, and
  run records); `prices --help` and the README document what `cache_write` prices
- Report now selects text, Markdown, or HTML with `--format` and writes one selected result to
  `--output-file`; scrub takes its trace product as positional OUT and uses the same flag for
  its result document. TTL summaries expose `offset_s`.
- The size-rung probe is the default measurement; full `replay` remains for the per-turn
  cache curve and the whole-session cost
- `report` reads a persisted run instead of re-summarising live state, and accepts a run
  directory, a trace name or `latest`
- Terminal, markdown and HTML reports build every cell from the same row, so the three
  cannot disagree on a number
- Endpoint labels are shortened under a caption instead of being clipped into identical rows;
  a sweep's native reference row keeps its label whole
- `sweep`'s selection sentence names `--include` ("the N endpoints named with --include")
  instead of claiming a ranking that never happened ("the N cheapest by listed price") when
  the endpoints were named rather than ranked
- A slash tag such as `@novita/fp8` parses as a provider pin; only `@preset/...` stays part
  of the model name
- The `--budget` refusal names the ways to fit: fewer endpoints or rungs, lower `--repeats`, or
  `--no-throughput`
- The packaged trace is addressed as `sample-trace` (was `sample`), so `inspect sample-trace`
  no longer reads like a subcommand
- The probe nonce now carries the endpoint as well as the rung
  (`probe_nonce(run_hex, rung, endpoint)`), so two OpenRouter tags of one provider probed in
  the same run no longer read back each other's cache writes
- A failed request's note carries the error body's `error_type` next to its status
  (`HTTP 502, provider_unavailable`), or `provider error, <error_type>` for a 2xx response
  that still carried an error payload
- The endpoint column's floor width is 32 (was 16), so a provider tag like
  `@sail-research/fp8` prints whole instead of clipped
- A single endpoint's own `target:model` prefix folds into the caption when its whole label
  does not fit the column, which until now only happened when two rows shared the prefix
- The HTML report names every price as an input-token price, marks a hit rate whose cold
  write already read back cached tokens, says where the cold (nonce) mode's marker sits, and
  the price bars' hover separates the run's cache share from the conditional `cached %`

### Fixed

- Providers that buffer a whole generation and flush it right after the first token no
  longer report meaningless tokens-per-second; the rung is noted as burst delivery and its
  `tok/s` is left empty
- Endpoints the account's settings refuse are named with the reason the gateway gave,
  instead of showing failed requests with no error
- Error bodies on streamed responses reach the recorded answer
- Streamed requests stop at `message_stop` and give up on an idle timeout
- The estimate prints line by line instead of escaping its newlines into one line
- Endpoints that come back untagged are reported instead of silently dropped
- A mistyped trace name fails before the endpoint listing is fetched
- A partial `probe`, `sweep`, or `replay` (a failed request, a skipped rung, or a failed
  turn) now writes its result document to stdout with a required `partial: true`, instead
  of only on success; the `operation_failed` error on stderr then names just the run
  (`run_dir`, and `run_hex` for `probe`/`sweep`) rather than repeating the whole document.
  A replay with a failed turn now also exits non-zero instead of `0`
- A `@provider` pin that matches no endpoint fails with `invalid_input` naming the tags
  that do exist, instead of reading as merely unpriced
- A drop reason naming an OpenRouter training-data restriction now says where to change it
  (the account's privacy setting)
- `endpoints` on an unknown model slug fails with a clean `not_found`, instead of printing
  OpenRouter's raw HTML 404 page
- `endpoints` reads the same default API key `sweep`'s listing does, so `latency_ms_30m`
  and `throughput_30m` are populated instead of `null` when a key is configured
- `tokens±N%` no longer fires when an endpoint's largest rung was skipped and its comparison
  fell back to a smaller one; a `token drift n/a (rung skipped)` note explains why instead
- The `drift` column renders its marker in full instead of eliding it, even past 120 columns
- A pin with a variant suffix (`@relace/fp4`) no longer reads as provider drift when Relace
  served it: the served provider is compared with the tag's provider part only, since a
  response names the provider and never the variant
- Cache-creation tokens on a cold write are no longer billed at both the input rate and the
  cache-write rate
- A served provider that billed nothing no longer prices as free; it falls back to the
  worst-case listed price instead
- The `spent $X (worst case $Y)` closing line no longer prints twice in human-mode output
- `probe` and `report` now print the same worst-case number for the same run
- The session-projection footer's prompt-token count reads `N k prompt tokens` under 1M and
  `N.N M prompt tokens` at or above it, instead of a small trace rounding to `0.0 M`
- `sweep`'s `candidates:` table and its `endpoints:` estimate table are separated by one
  blank line instead of running straight into each other

[Unreleased]: https://github.com/DamianPala/provibench/compare/0.2.0...main
[0.2.0]: https://github.com/DamianPala/provibench/releases/tag/0.2.0
