# Anthropic Messages to OpenAI Chat Completions

## Recommendation

Write a small, local converter in `promptreplay`, using LiteLLM's adapter as a reference rather than as a runtime dependency.
LiteLLM has the broadest implementation and a real Python entry point, but it is a large dependency and its adapter is coupled to LiteLLM types, provider detection, streaming wrappers, and parameter policy.
The trace needs deterministic replay, while the adapter intentionally changes behavior by target model, preserves some provider-specific fields, truncates long tool names, and may pass unknown fields through.
The small proxy projects are useful compatibility references, but they are HTTP servers, not stable importable libraries.
The local converter should preserve message order and exact text/tool JSON bytes, remove Anthropic-only cache markers from the OpenAI wire, and emit a note for every lossy decision.
Usage parsing should be a separate small function that accepts both standard OpenAI usage and provider extensions.
This avoids making cache results depend on a transitive gateway's release or serialization choices.

## Shape inventory of the recorded trace

Source: `traces/acpc-cov-deepseek.jsonl`, 49 JSONL entries, 13 909 441 bytes. Counts below are block/message occurrences across all entries, produced with `jq` probes, not by loading the trace into the report.

### Top-level keys

All 49 requests contain: `model`, `system`, `messages`, `tools`, `metadata`, `max_tokens`, `stream`, and `output_config`.
`thinking` and `context_management` occur in 48/49 requests. No request contains `stop_sequences`, `temperature`, or `top_p` as a present field; the sampled JSON represents those as null when queried with `//null`.

Observed values:

- `model`: `deepseek-flash`.
- `max_tokens`: 32 000; `stream`: true.
- `thinking`: `{"type":"adaptive"}`.
- `context_management`: `{"edits":[{"type":"clear_thinking_20251015","keep":"all"}]}`.
- `output_config`: usually `{"effort":"high"}`; one observed shape adds `format: {type: "json_schema", schema: {type: "object", properties: {title: {type: "string"}}, required: ["title"], additionalProperties: false}}`.
- `metadata`: only `user_id`, whose value is a JSON-encoded string containing device, account, and session identifiers.
- `system`: an array of text blocks. The trace includes both text with no `cache_control` and text with `{"type":"ephemeral"}`. A separate system string shape also occurs inside messages.
- `tools`: 28 named tools per request in the 48 requests with the normal tool set; every observed tool has `name`, `description`, and `input_schema` with JSON Schema type `object`. No tool has `cache_control`.

### Blocks and message variants

Block counts are: `thinking` 1 128, `tool_use` 1 365, `tool_result` 1 365, `text` 335, and scalar-string content 1 128.
The only block types in the trace are those five; there are no images, documents, redacted thinking blocks, server-tool blocks, or nested tool-result content arrays.

Message content shapes include:

- `system` with scalar string: 1 128; `system` with array containing text: 48.
- `user` with text array: 49 requests, with one text block in the first occurrence and two text blocks in the other 48.
- `user` with tool results: 984 messages with one result, 51 with two, and 93 with three.
- `assistant` with thinking plus one tool use: 841 messages; with thinking, one text, and one tool use: 143; with thinking and two tool uses: 51; with thinking and three tool uses: 46; with thinking, one text, and two tool uses: 47. These are content-list shapes; the tool-use multiplicity is significant because OpenAI represents parallel calls in one assistant `tool_calls` array.
- Every `thinking` block has keys `type`, `thinking`, and `signature`; signature length is 36. There are 305 empty-thinking blocks and 823 non-empty blocks. There is no `redacted_thinking` in the trace.
- Text blocks have variable text lengths. System and user text sometimes carry `cache_control: {type: ephemeral}`; tool definitions do not. Assistant text/cache combinations must be counted from the source per turn, not reconstructed from a normalized representation.
- Every `tool_result` has `tool_use_id` and string `content`. 519 omit `is_error`, 782 set `is_error: false`, and 64 set `is_error: true`. No observed result has array content.

For cache measurement, preserve turn order, every text byte, every tool name, every tool argument JSON value, and the exact sequence of messages. Anthropic `cache_control` has no standard OpenAI Chat Completions equivalent and must be removed or handled by a target-specific extension; retaining it in a generic OpenAI body can cause rejection or provider-specific behavior. Removing it does not justify concatenating or reformatting prompt text. A converter should record the number and locations of removed markers and whether thinking blocks were dropped.

## Candidate table

Dates and star counts are repository metadata visible on 2026-09-14 and are not stable API facts. “Importable” means the translation function can reasonably be imported without starting the server.

| Candidate | Language/license; maintenance signal | Importability and mapping assessment |
|---|---|---|
| LiteLLM `LiteLLMAnthropicMessagesAdapter` | Python, MIT. Large, active repository; exact stars and latest commit are repository metadata. | Importable class in `litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py`. Maps system into OpenAI system messages, `input_schema` to `function.parameters`, `tool_use` to assistant `tool_calls`, and string/list tool results to `role: tool`. It handles images/documents, thinking blocks, redacted thinking, tool-name truncation, stop sequences, metadata, output format, streaming, and usage. It maps non-Claude thinking to `reasoning_effort`, not universally to `reasoning_content`. Its source currently drops `reasoning_tokens` from Anthropic usage, and its model-specific cache handling is not appropriate for a byte-stable generic replay. Known issues include in-place mutation of input schemas (#34277), corrupted tool schema type (#30557), dropped/unsupported nested tool-result shapes (#22841/#22878), and missing reasoning-token accounting (#36376). |
| `empero-org/claude-code-proxy` | Python, MIT according to repository metadata; repository presents a recent full proxy implementation. | `request_converter.py` is a server project's internal module, not a published standalone package. It explicitly advertises full tool conversion, thinking stripping, images/documents, and cache-token handling, so it is a useful test oracle. The public feature description is not enough to establish deterministic serialization or exact handling of this trace's `is_error` cases. |
| `musistudio/llms` / `claude-code-router` | TypeScript/Node ecosystem, MIT router repository. `claude-code-router` has a large public community footprint (repository page showed about 2.9k stars and 811 issues on the checked date). | The router delegates conversion to the `@musistudio/llms` package. Its author describes a bidirectional `AnthropicTransformer`, but it is part of a provider gateway rather than a small Python library. It is relevant for provider-specific behavior, especially cache breakpoints and reasoning, not a suitable direct dependency for promptreplay. |
| `1rgs/claude-code-proxy` | Python, repository metadata identifies an open-source proxy; 42 commits and 44 issues were visible in the checked result. | Server-bound proxy using LiteLLM. It is a practical integration, not an independently versioned converter API. Its correctness inherits the provider and LiteLLM paths selected by the server. |
| `maxnowack/anthropic-proxy` | JavaScript, MIT. Small repository, 8 commits in the checked GitHub result. | `index.js` is a minimal OpenRouter proxy. It is useful as a compact reference but not a reusable library and not sufficient for thinking signatures, context management, structured output, or detailed usage extensions. |
| Bifrost (`maximhq/bifrost`) | Go, Apache-2.0 repository. Gateway with multiple provider adapters; not a Python package. | A substantial gateway with Anthropic/OpenAI compatibility, but the converter is embedded in request routing. An issue reports `cache_control` being silently stripped in an Anthropic-backed OpenRouter path (#3942), and another reports OpenAI-to-Anthropic tool translation failures (#3511). Those are direct warnings for cache-sensitive replay. |
| PyPI-only converter packages | No package found that is both clearly maintained and narrowly exposes a stable Anthropic Messages ↔ Chat Completions conversion API. | The useful implementations found are components of gateways or LiteLLM. Adding an unknown narrow package would not reduce the correctness or determinism work required for this trace. |

### Candidate-specific usage behavior

LiteLLM's reverse adapter reads `usage.prompt_tokens` and `completion_tokens`; cache-read input is taken from explicit cache fields or `prompt_tokens_details.cached_tokens`, and cache-write from explicit fields or `prompt_tokens_details.cache_creation_tokens` / `cache_write_tokens`. It computes Anthropic input as prompt minus cache-read minus cache-write, flooring at zero. It does not read `completion_tokens_details.reasoning_tokens` in the cited implementation. The adapter therefore does not preserve all usage information requested by this project.

The other proxies generally normalize the upstream response into Anthropic SSE/message usage for Claude Code rather than exposing a reusable usage parser. A local parser should additionally accept DeepSeek's `prompt_cache_hit_tokens` and `prompt_cache_miss_tokens`, and OpenRouter's separate `provider`/generation metadata. Treat `prompt_cache_hit_tokens` as cached prompt tokens and miss tokens as non-cached prompt tokens only when the provider documents those fields; do not infer cache writes from a missing field.

## Mapping spec for a local converter

The converter should deep-copy the input, build ordinary dictionaries in a fixed insertion order, and serialize tool arguments with one explicit policy (`json.dumps(..., separators=(',', ':'), ensure_ascii=False)` if the target accepts compact JSON). It must never mutate the recorded body. Stable output means stable semantics and JSON serialization; the HTTP client's final bytes should be tested separately.

| Anthropic shape | OpenAI Chat Completions output | Decision for promptreplay |
|---|---|---|
| `model` | `model` | Replace only with the target model, as replay already does. |
| `system` array of text blocks | One leading `{role: system, content: <concatenated text>}` or a content-part list | Preserve block order and text bytes. Concatenate only if the target requires a scalar; record removed cache markers. Do not include `cache_control` in generic OpenAI JSON. |
| `messages` scalar string content | Same role and string content | Preserve exactly. For the trace, scalar system messages are normalized to system messages. |
| user text blocks | User message content string when there is no image; otherwise content parts | Preserve text order and bytes. Do not merge adjacent messages. |
| assistant text blocks | Assistant `content`; keep text order | Keep text even when a thinking block is dropped. If cache markers force content-part form for a target extension, use that only in a target adapter. |
| `tool_use {id,name,input}` | Assistant `tool_calls: [{id,type:function,function:{name,arguments}}]` | Keep one call per block and source order. Compact JSON encoding of `input` is deterministic. Maintain a reversible ID map if the provider restricts IDs. |
| parallel tool uses | One assistant message with multiple `tool_calls` | Do not split calls into separate assistant messages. |
| `tool_result` string | `{role: tool, tool_call_id: tool_use_id, content: string}` | Preserve string bytes and ID. `is_error` is not a standard Chat Completions field; add a short deterministic error prefix only for a target that needs the distinction, and note the loss. Default: drop the flag and report its count. |
| `tool_result` array content | Tool content string made by concatenating text parts in order; non-text parts become deterministic JSON text or target image parts | Not present in this trace, but implement it. The requested policy is concatenated text. Never silently drop unknown parts. |
| `thinking` with text/signature | DeepSeek-style assistant `reasoning_content` containing thinking text | For a target explicitly declaring DeepSeek/vLLM-style reasoning, preserve non-empty thinking in order. For generic OpenAI, drop it and report block/byte counts. Signatures have no generic OpenAI field and are dropped. Empty thinking is dropped unless the target requires a placeholder. |
| `redacted_thinking` | No generic equivalent | Drop and report. Not present in this trace. |
| `tools[].input_schema` | `tools[].function.parameters` | Copy schema, preserve property order, description, required, and additionalProperties. Do not add Anthropic tool keys to the schema. Do not truncate names unless the target imposes a documented limit; if truncating, persist a reversible mapping and report it. |
| `cache_control` on system/text blocks | No standard field | Drop from generic OpenAI wire, count locations, and never replace it with invented prompt text. A provider-specific adapter may map it to an extension, but that is a different replay mode. |
| `metadata.user_id` | Optional provider metadata or dropped | Drop by default because it is Anthropic billing/cache metadata and contains an encoded identity. Report that it was dropped. Do not turn it into an arbitrary OpenAI `user` field without an explicit target policy. |
| `thinking: {type: adaptive}` | `reasoning_effort` where supported, or omitted | `output_config.effort: high` is the stronger target hint. For DeepSeek-style targets use target-specific reasoning behavior; for generic Chat Completions omit unsupported fields and record the omission. |
| `context_management` | No standard field | Drop and report. `clear_thinking_20251015` cannot be reproduced by Chat Completions. |
| `output_config.effort` | `reasoning_effort` where supported | Map only for targets that document it. Otherwise drop and report. |
| `output_config.format` JSON schema | `response_format: {type: json_schema, json_schema: ...}` | Implement because one request contains it. Preserve schema; derive a deterministic name such as `output` if OpenAI requires a name, and record that derivation. |
| `max_tokens` | `max_completion_tokens` preferred, fallback `max_tokens` | Choose per target capability, not both. Preserve the numeric value. |
| `stop_sequences` | `stop` | Implement even though absent in this trace. Preserve list order. |
| `temperature`, `top_p` | Same keys | Implement when present; absent keys remain absent. Do not emit nulls. |
| `stream` | `stream` | Replay currently needs non-streaming response parsing, so set false in the replay body; this is a replay policy, not a conversion rule. |

### Reverse usage parser

Return a normalized record with `prompt_total`, `cached`, `cache_write`, and `output_tokens`, plus raw usage and notes.

1. `prompt_total = prompt_tokens` when present; `output_tokens = completion_tokens`.
2. Prefer explicit provider cache fields if present. Otherwise use `prompt_tokens_details.cached_tokens`.
3. Recognize `prompt_cache_hit_tokens` as cached and `prompt_cache_miss_tokens` as non-cached only for DeepSeek-compatible responses. If both hit and miss exist, use their sum as a consistency check against prompt tokens.
4. Preserve `completion_tokens_details.reasoning_tokens` in raw/optional normalized usage even though it is not output text; do not add it to output twice.
5. Preserve OpenRouter `provider` from the response or generation record. Do not confuse it with an OpenAI usage field.
6. If cached exceeds prompt total, retain the raw values, clamp derived non-cached input to zero, and add a note. If a provider reports cache write separately, keep it separate; otherwise set it to null, not zero-by-assumption.

### Explicit losses to show in the run report

Report counts for dropped cache markers, thinking/redacted-thinking blocks, signatures, metadata, context-management edits, `is_error` flags, unsupported output configuration, tool-name rewrites, and any unknown content blocks. Also report whether the target used `reasoning_content`, `reasoning_effort`, or neither. This makes a cache comparison distinguish a provider cache result from a changed prompt.

## Sources

- LiteLLM adapter source: https://raw.githubusercontent.com/BerriAI/litellm/main/litellm/llms/anthropic/experimental_pass_through/adapters/transformation.py
- LiteLLM repository: https://github.com/BerriAI/litellm
- LiteLLM tool-schema mutation issue #34277: https://github.com/BerriAI/litellm/issues/34277
- LiteLLM tool-result loss/unsupported block issue #22878: https://github.com/BerriAI/litellm/issues/22878
- LiteLLM tool-result translation issue #22841: https://github.com/BerriAI/litellm/issues/22841
- LiteLLM tool-schema corruption issue #30557: https://github.com/BerriAI/litellm/issues/30557
- LiteLLM reasoning-token usage issue #36376: https://github.com/BerriAI/litellm/issues/36376
- empero Claude Code proxy: https://github.com/empero-org/claude-code-proxy
- 1rgs Claude Code proxy: https://github.com/1rgs/claude-code-proxy
- musistudio Claude Code Router: https://github.com/musistudio/claude-code-router
- musistudio transformer rationale: https://github.com/musistudio/claude-code-router/blob/main/blog/en/maybe-we-can-do-more-with-the-route.md
- maxnowack Anthropic proxy: https://github.com/maxnowack/anthropic-proxy
- Bifrost gateway: https://github.com/maximhq/bifrost
- Bifrost cache-control issue #3942: https://github.com/maximhq/bifrost/issues/3942
- Bifrost tool translation issue #3511: https://github.com/maximhq/bifrost/issues/3511
- Anthropic Messages API request types: https://docs.anthropic.com/en/api/messages
- OpenAI Chat Completions API reference: https://platform.openai.com/docs/api-reference/chat
- OpenRouter usage and generation metadata: https://openrouter.ai/docs/api-reference/overview
- DeepSeek API usage and caching: https://api-docs.deepseek.com/api/create-chat-completion
