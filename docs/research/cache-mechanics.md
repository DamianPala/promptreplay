# Prompt-cache mechanics and `provibench`

## Answer

1. A replay with `max_tokens: 1` equals live cache accounting only when the serving stack caches input/prompt tokens but does not publish generated tokens for later prefix reuse.
2. It under-reports whenever decoded output is inserted into a reusable prefix cache: at turn N+1 the missing amount is, in principle, the output of turn N (subject to block/granularity and eviction).
3. vLLM, SGLang, TensorRT-LLM, and LMDeploy document or expose generated-block reuse, so replay can under-report by completed output blocks.
4. DeepSeek explicitly persists request output boundaries, and the supplied trace confirms this in production: `provibench trace, 2026-09-12`.
5. OpenAI and Anthropic document prompt-prefix caching, not completion caching; treat decoded-token reuse as unknown/not established.
6. Gemini, Moonshot/Kimi, Zhipu/GLM, and Qwen document context/prefix caching, but do not clearly specify whether generated tokens are admitted; treat as unknown unless measured.
7. The ranking is not automatically valid: the replay is byte-identical across providers, but providers can differ in decode-KV admission, granularity, TTL, routing, and cache policy.
8. If every provider is prompt-only, ranking is much more comparable, though thresholds, TTLs, and routing still matter.
9. For a decode-caching provider, a 6 000-token turn can make a replay miss roughly 6 000 live-cacheable tokens, not merely one token.

## Stack and provider matrix

| Stack/provider | Decode tokens cached? | Granularity / TTL | Source and confidence |
|---|---|---|---|
| vLLM automatic prefix caching | Yes, completed blocks. A block is added when full; its example shows prompt+generated blocks becoming reusable. | Block size is configurable by model/engine; cache is hash-keyed and evicted by ref-count/LRU-style policy when space is needed. No fixed TTL. | [vLLM prefix-cache design](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md), documented, high. |
| SGLang RadixAttention | Yes. The RadixAttention paper says it retains prompt and generation results in a radix tree instead of discarding them. | Radix nodes/tokens; eviction is memory-policy dependent, not a provider TTL. | [SGLang thesis chapter](https://www2.eecs.berkeley.edu/Pubs/TechRpts/Hold/798934743815c00b62ec6dbd1ff1338c.pdf), documented, high. |
| TensorRT-LLM KV cache reuse | Yes, when decode retention is configured/supported. Current docs explicitly distinguish prompt retention from `decode_retention_policy`. | Fixed-size blocks, power-of-two `tokens_per_block`, default 128 in legacy reuse docs; prioritized LRU/retention duration, otherwise memory pressure. | [KV cache system](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html), [reuse guide](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/kv-cache-reuse.md), documented, high. |
| LMDeploy/TurboMind | Yes, configurable. `cache_generation='auto'` indexes full generated blocks; `'all'` also indexes the terminal partial block for exact multi-turn resume; `'none'` disables generated blocks. | Full-block threshold is `cache_block_seq_len`; terminal partial block only in `all`; eviction/TTL not specified as a fixed time. | [LMDeploy config](https://github.com/InternLM/lmdeploy/blob/main/docs/en/inference/turbomind_config.md), documented, high. |
| Hugging Face TGI | Unknown as a general TGI serving guarantee. The current docs expose chunking and deployment, but no authoritative generated-block prefix-cache contract was found. | Unknown. | [TGI conceptual docs](https://huggingface.co/docs/text-generation-inference/en/conceptual/chunking), unknown, low. |
| DeepSeek API | Yes. Docs state each request produces cache units at end of user input and end of model output; the multi-turn example says the second request reuses the first request's prefix. | 64-token cache units are reported in current guidance; disk persistence also occurs at request boundaries and fixed intervals. Docs say idle retention is usually hours to days, not a strict TTL. | [DeepSeek context caching](https://api-docs.deepseek.com/guides/kv_cache/), plus `provibench trace, 2026-09-12`, documented/measured, high. |
| OpenAI API | Not documented as completion caching. Automatic caching is the longest previously computed input prefix; completion usage is reported separately. | Minimum 1 024 tokens, then 128-token increments; typically cleared after 5–10 minutes idle and within one hour of last use. | [OpenAI prompt caching](https://openai.com/index/api-prompt-caching/), prompt-only documented, decode reuse unknown. |
| Anthropic API | Not established. Explicit/automatic breakpoints cache the prompt through a marked block; docs do not say generated completion tokens are independently admitted. | 5-minute default or 1-hour option; cache write is 25% above base input for 5m and 2x for 1h; hits are discounted. | [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching), prompt-only documented, decode reuse unknown. |
| Google Gemini API | Unknown for decoded tokens. Implicit caching is automatic on Gemini 2.5+; explicit caching describes cached input content, not output admission. | Explicit cache has user-selected TTL; implicit-cache TTL/details are not promised in the cited docs. | [Gemini context caching](https://ai.google.dev/gemini-api/docs/caching), documented input caching, decode unknown. |
| Moonshot/Kimi API | Unknown. Kimi is stateless at the API message layer and recommends sending both user and assistant history; `prompt_cache_key` is recommended for session affinity, but decode admission is not specified. | `prompt_cache_key`/provider policy; no public fixed TTL found in cited docs. | [Kimi multi-turn guide](https://platform.kimi.ai/docs/guide/engage-in-multi-turn-conversations-using-kimi-api), [chat API](https://platform.kimi.com/docs/api/chat), unknown, medium-low. |
| Zhipu/GLM API | Unknown. Official context-cache docs expose cached-token usage, but do not state whether generated tokens are inserted. | Provider-defined; no fixed TTL/granularity established from the cited documentation. | [Zhipu context cache](https://docs.bigmodel.cn/cn/guide/capabilities/cache), documented cache metric, decode unknown. |
| Alibaba Qwen/Model Studio | Unknown. Qwen API is stateless; session cache/implicit cache can cache common conversation prefixes, but admission of decoded blocks is not specified. | Implicit common-prefix condition is at least 1 024 tokens in current QwenCloud docs; session-cache TTL not stated there. | [Alibaba multi-round conversation](https://help.aliyun.com/en/model-studio/multi-round-conversation), [Qwen context cache](https://docs.qwencloud.com/developer-guides/run-and-scale/context-cache), decode unknown. |
| OpenRouter | Not a serving stack. `native_tokens_cached` is the upstream provider's native cached-token count; `cache_discount` reports savings. OpenRouter says sticky routing improves reuse, but does not publish a complete provider-by-provider implicit-cache matrix. | Depends on selected upstream; session/sticky routing affects whether the warm cache is reached. | [Prompt caching](https://openrouter.ai/docs/guides/best-practices/prompt-caching), documented, high for semantics; upstream decode behavior remains provider-specific. |

## Measurement implications for `provibench`

* Cold starts should use a random nonce in a cacheable prefix, preferably the system prompt, and record the exact tokenized prompt. Do not rely on elapsed time alone: OpenAI and Anthropic have short documented windows, while DeepSeek and self-hosted LRU caches have different lifetimes.
* Keep a warm and a cold protocol separate. A nonce changes the prefix and measures cold prefill; removing it measures reuse. Never use a nonce in the stable prefix when the goal is a live-session comparison.
* Interpret thresholds carefully: OpenAI does not cache below 1 024 input tokens and rounds reuse in 128-token increments. DeepSeek uses 64-token-scale units. Engine block caches can make turn-2 counts jump at block boundaries rather than track exact output length.
* For vLLM, SGLang, TensorRT-LLM, and LMDeploy, `max_tokens: 1` can fail to fill the generated block that would be reusable in live generation. This creates a lower bound on live cache reuse, not a neutral replay.
* Add a generate-and-continue probe: request the recorded/full output O at turn N, then send prompt N + O + the recorded next message, and test whether cached tokens are at least the prefix length through O. Run at several O lengths and repeat after controlled idle periods. This directly tests decode admission, block completion, eviction, and routing.
* The probe must use deterministic settings where possible, preserve assistant reasoning/tool-call fields exactly, and distinguish “cache not admitted” from “request routed to a different worker”. For OpenRouter, pin a stable session/provider route where the API permits it.
* Published prior art is adjacent rather than identical. SGLang/RadixAttention and CachedAttention explicitly target multi-turn reuse of generation state. `Don't Break the Cache` (arXiv:2601.06007) evaluates long-horizon agent prompt caching across OpenAI, Anthropic, and Google, but reports cost/TTFT strategies rather than a decode-KV admission probe. Infron `prompt-cache-bench` measures provider/routing/cache behavior, but its public benchmark description does not establish that it tests full-output continuation.
* Therefore report both `cached_tokens` and a normalized prompt-only metric, and label any provider ranking as “under this replay protocol”. A live-session ranking needs the continuation probe or real live traces.

## Sources

* [vLLM automatic prefix-cache design](https://github.com/vllm-project/vllm/blob/main/docs/design/prefix_caching.md)
* [SGLang thesis, RadixAttention](https://www2.eecs.berkeley.edu/Pubs/TechRpts/Hold/798934743815c00b62ec6dbd1ff1338c.pdf)
* [TensorRT-LLM KV cache system](https://nvidia.github.io/TensorRT-LLM/features/kvcache.html)
* [TensorRT-LLM KV cache reuse](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/legacy/advanced/kv-cache-reuse.md)
* [LMDeploy TurboMind configuration](https://github.com/InternLM/lmdeploy/blob/main/docs/en/inference/turbomind_config.md)
* [DeepSeek context caching](https://api-docs.deepseek.com/guides/kv_cache/)
* [OpenAI prompt caching](https://openai.com/index/api-prompt-caching/)
* [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)
* [Google Gemini context caching](https://ai.google.dev/gemini-api/docs/caching)
* [Moonshot/Kimi multi-turn API](https://platform.kimi.ai/docs/guide/engage-in-multi-turn-conversations-using-kimi-api)
* [Zhipu context cache](https://docs.bigmodel.cn/cn/guide/capabilities/cache)
* [Alibaba Qwen multi-round conversation](https://help.aliyun.com/en/model-studio/multi-round-conversation)
* [QwenCloud context cache](https://docs.qwencloud.com/developer-guides/run-and-scale/context-cache)
* [OpenRouter prompt caching and sticky routing](https://openrouter.ai/docs/guides/best-practices/prompt-caching)
* [Don't Break the Cache, arXiv:2601.06007](https://arxiv.org/abs/2601.06007)
* [Infron prompt-cache-bench](https://github.com/InfronAI/prompt-cache-bench)
* [CachedAttention, arXiv:2403.19708](https://arxiv.org/abs/2403.19708)

