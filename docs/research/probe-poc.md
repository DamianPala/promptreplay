# Probe protocol PoC (2026-09-14)

Question: can the per-provider cache measurement be done on a few real turns instead of replaying every turn? A 30-turn replay of the sample trace sends 1,90 M tokens of which only 93 k are new bytes (95 % of the spend re-checks the same prefix), and its hit ratio is 30 correlated samples of one question: did this request land on a warm replica.

## Protocol

Script: `scripts/probe_poc.py` (standalone, reuses `bench/`). Per run spec, sequentially:

- Rung = a 1-based turn `k` of the main conversation. Cold: send turn `k` once. Warm: send turn `k+1` R times, 1 s apart.
- Nonce `promptreplay-probe:<run hex>:<k>` prepended to the first system block, identical across specs, different per rung, so every cold request is truly cold and later rungs do not inherit earlier ones.
- `max_tokens: 1`, `stream: false`, provider pinning via `provider.only`.
- Rungs 1, 13, 30 of `traces/acpc-coverage-raw.jsonl` (prompt 20 k, 50 k, 93 k tokens), repeats 5, 2, 2.

Two passes, 10:3x UTC, DeepSeek V4.1 Flash: native API plus Novita, SiliconFlow, GMICloud via OpenRouter.

## Results

Warm attempts with a cache hit, both passes pooled (hit = `cache_read_input_tokens` > 0; on every hit the cached fraction of the cold prefix was 0,99 to 1,00):

| provider | hits / warm attempts | pass 1 sequences (rung 1, 13, 30) | pass 2 sequences |
|---|---|---|---|
| DeepSeek native | 18 / 18 | 1 1 1 1 1 · 1 1 · 1 1 | 1 1 1 1 1 · 1 1 · 1 1 |
| GMICloud (fp8) | 15 / 18 | 1 1 1 1 0 · 1 1 · 1 1 | 1 1 1 0 1 · 0 1 · 1 1 |
| Novita (fp8) | 9 / 18 | 0 0 1 1 1 · 0 0 · 0 1 | 0 1 1 1 1 · 0 1 · 0 0 |
| SiliconFlow (fp8) | 4 / 13 | 502 on cold, rung skipped · 0 0 · 0 1 | 0 0 1 0 1 · 0 1 · 0 0 |

Cost per pass, all four providers: 0,32 and 0,38 USD (tokens sent 2,2 M per pass; misses on the weak resellers are billed at full price, so the worst-case estimate is the honest one). A 30-turn full replay of the same four would be ~2,3 USD worst case per pass.

## What the numbers say

- Ranking is stable between passes and matches the earlier full-replay runs (DeepSeek ≫ GMICloud > Novita ≈ SiliconFlow). Novita and SiliconFlow swap order between passes: that is the noise floor of 9 samples per pass.
- The hit sequences show the mechanism: misses followed by hits on the same prefix are load-balanced replicas without prefix-aware routing. A live session warms replicas one by one, exactly as the sequence does; the full replay measured the same effect with 30 correlated, more expensive samples.
- No provider showed "caches at 20 k, fails at 93 k": whenever a request hit, it hit the whole prefix at every rung. The ladder stays because it costs one pair per rung and this is the cheapest way to keep checking.
- DeepSeek's 64-token cache blocks are visible (20 198-token prompt → 20 096 cached on the first warm hit).
- Latency needs more samples than cache does: single cold requests had outliers of 33 s (DeepSeek, rung 13, pass 2) and 40 s (SiliconFlow, rung 13, pass 1). Cold vs warm gain is visible on native DeepSeek (2,1 s → 1,4 s at 20 k) and noisy elsewhere.
- Frac just under 1,0 is expected: Claude Code moves the `cache_control` marker, so the last message of turn `k` is a block and in turn `k+1` a bare string; same bytes, slightly different tokenization tail.

## Consequences for the design

1. The probe replaces the full replay as the default measurement (`sweep`, `--quick`). Full replay remains as `--full` for the per-turn curve.
2. Summary metric per provider: hits ÷ warm attempts (with the per-hit fraction shown separately), not the mean of the per-hit fraction, which made Novita and SiliconFlow look equal at 0,67 while one had a skipped rung.
3. Repeats: rung 1 is the cheapest (26 k per warm request) and carries the routing statistic, so it gets the most repeats (6 to 8); larger rungs get 2 each. Cold latency needs its own repeats or a p50 over rungs.
4. A failed cold request skips the rung and is reported as such; a 502 on the cold write must not count as a miss.
5. Estimate shown before a run is the worst case (no cache), because weak resellers bill close to it.
