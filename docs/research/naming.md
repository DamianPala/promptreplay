# Naming

Status: open. Decide at the end of 0.2, before the PyPI upload. The name is baked into the env prefix (`PROVIBENCH_*`), the XDG config and cache paths, the entry point, and the package directory, so the rename is one mechanical slice; do it before the first public tag, not after.

## Why `provibench` is weak

- "provider" is generic: cloud providers, payment providers, healthcare providers. Nothing says LLM.
- "provi" also reads as provisioning.
- It says nothing about the mechanism (replay of a recorded session) or the subject (prompt cache, cost, latency per endpoint).
- Since 0.2 the tool reports more than cache (throughput, TTFT, reliability, later tool-call agreement), so a cache-specific name would be wrong too.

## What the name should carry

The stable, differentiating idea is: replay a real agent session, byte for byte, against many endpoints. Metrics change; replay does not. Prefer one word, lowercase, no hyphen, pronounceable, free on PyPI and GitHub.

## Candidates checked on 2026-09-14

PyPI checked with `GET https://pypi.org/pypi/<name>/json` (404 = free). GitHub checked with `gh search repos` where noted.

| name | PyPI | GitHub | notes |
|---|---|---|---|
| `promptreplay` | free | not checked | says what it does, no metric baked in. Favourite. |
| `replaybench` | free | three unrelated repos, one an "AI agent evaluation" platform | clear, but collides |
| `traceplay` | free | not checked | short, slightly vague |
| `replayllm` | free | not checked | descriptive, clunky |
| `cachereplay` | free | not checked | good if the tool stayed cache-only; it will not |
| `cacheprobe` | free | not checked | same limitation |
| `promptcache-bench` | free | not checked | descriptive, hyphen, generic "bench" |
| `prefixbench`, `tokenreplay`, `cachetrace`, `hitratio`, `cachedeck`, `prefixcache`, `sessionreplay`, `rerun-llm`, `llmrerun`, `traceroute-llm` | free | `tokenreplay` has unrelated repos | weaker fits; `sessionreplay` collides with the web-analytics term |
| `provibench` | free | free (`DamianPala/provibench` does not exist) | current name |
| `cachebench`, `promptcache`, `llmreplay`, `tracebench`, `agentreplay`, `reprompt`, `playback`, `agentbench` | taken | | |

## Before deciding

- Re-check PyPI and GitHub for the finalist on the day; names get taken.
- Check the npm registry too if a JS companion ever becomes likely.
- Try the CLI in a sentence: `promptreplay sweep sample deepseek/deepseek-v4.1-flash`, `PROMPTREPLAY_TRACES_DIR`, `~/.config/promptreplay/targets.toml`.
