# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  reference spec
- `sweep TRACE MODEL`: list a model's OpenRouter endpoints, pin one probe spec to every
  provider tag, add a spec for each native target that carries the model, and probe them in
  one run ordered by the measured effective price
- `sweep` endpoint selection: a stability floor on endpoint status and one-day uptime,
  `--include`/`--exclude` by tag prefix, `--zdr`, `--sort price|uptime|throughput|latency`
  and `--top N`
- `sweep` availability pre-check: one `max_tokens 1` request per ranked candidate, so an
  endpoint the account's settings exclude is named with the gateway's reason instead of
  failing three cold requests
- `sweep --parallel N` to probe several specs at once, recorded in the run's notes because
  it moves the latency numbers
- `report --format html --output-file PATH`: one self-contained page with the same tables, hit-rate bars per
  rung, effective prompt price bars and the full-replay cache curve, in dark mode, with no
  scripts and no external resources
- `scrub`: write a shareable copy of a trace with home paths, the encoded Claude Code
  project directories, secrets and e-mail addresses removed, `--user` and `--replace` for
  what the patterns cannot know, and `--turns N` to keep a cheaper prefix
- Packaged sample trace — the first 30 turns of a real Claude Code session on DeepSeek V4.1
  Flash — accepted as `sample` by `inspect`, `replay` and `scrub`
- Traces load from gzipped files, and `scrub TRACE OUT` writes them
- Runs persist their protocol, nonce, prices and an endpoint snapshot, so `report`
  re-renders a run with no network, in the terminal, markdown or HTML
- The endpoint snapshot is per spec and dated: every probe and sweep records the listed
  input and cache-read price, quantization, context length, uptime 1d and status of the
  endpoint it pinned, so price and endpoint drift are part of the record
- `history [MODEL]`: every run of a model in the runs dir as one table — date, hit %,
  effective $/M, TTFT, tok/s, errors and the listed price of that week — with `--trace` and
  `--since`, a hit-rate sparkline per spec, and `--json` rows and series
- `compare RUN_A RUN_B`: the per-spec delta between two runs of one trace, with the listed
  price change taken from the two endpoint snapshots, `latest`/`previous` naming, and an
  `invalid_input` refusal across traces or protocols unless `--force`
- `--force` for scrub's OUT or `--output-file`, which are otherwise refused when the target
  exists; `report --output-file` replaces its file, since a report is derived from the run alone
- Native prices from LiteLLM's community table, cached under `$XDG_CACHE_HOME` and refetched
  weekly or on `provibench prices --update`; `targets.toml` `prices` entries override it, and
  every estimate names its price source (`table`, `litellm` or `openrouter-endpoint`)

### Changed

- Report now selects text, Markdown, or HTML with `--format` and writes one selected result to
  `--output-file`; scrub takes its trace product as positional OUT and uses the same flag for
  its result document. TTL summaries expose `offset_s`.
- The size-rung probe is the default measurement; full `replay` remains for the per-turn
  cache curve and the whole-session cost
- `report` reads a persisted run instead of re-summarising live state, and accepts a run
  directory, a trace name or `latest`
- Terminal, markdown and HTML reports build every cell from the same row, so the three
  cannot disagree on a number
- Spec labels are shortened under a caption instead of being clipped into identical rows;
  a sweep's native reference row keeps its label whole
- A slash tag such as `@novita/fp8` parses as a provider pin; only `@preset/...` stays part
  of the model name
- The `--budget` refusal names the ways to fit: fewer specs or rungs, lower `--repeats`, or
  `--no-throughput`

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

[Unreleased]: https://github.com/DamianPala/provibench/commits/main
